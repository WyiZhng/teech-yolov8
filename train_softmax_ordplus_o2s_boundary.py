import argparse
import os
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from ord2seq_head import Ord2SeqOrdinalHead


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def map_ic4(icdas: int) -> int:
    if icdas <= 0: return 0
    if icdas <= 2: return 1
    if icdas <= 4: return 2
    return 3


class Icdas4RoiDataset(Dataset):
    def __init__(self, csv_path, img_root, img_size=256, expand=1.25, augment=True):
        self.df = pd.read_csv(csv_path)
        self.root = img_root
        self.expand = expand
        self.augment = augment
        self.tx = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(os.path.join(self.root, str(r["image_id"]))).convert("RGB")
        w_img, h_img = im.size

        x, y, w, h = float(r.get("gx", r.get("x"))), float(r.get("gy", r.get("y"))), \
                     float(r.get("gw", r.get("w"))), float(r.get("gh", r.get("h")))

        cx, cy = x + w / 2, y + h / 2
        w2, h2 = w * self.expand, h * self.expand
        x1, y1 = max(0, cx - w2 / 2), max(0, cy - h2 / 2)
        x2, y2 = min(w_img, cx + w2 / 2), min(h_img, cy + h2 / 2)
        crop = im.crop((x1, y1, x2, y2))

        if self.augment and random.random() < 0.5:
            crop = transforms.functional.hflip(crop)
        img = self.tx(crop)

        y4 = int(r["ic4"]) if "ic4" in r else map_ic4(int(r["icdas"]))
        return img, y4


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_outputs=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, x):
        return self.mlp(x)


class ResNet18SoftmaxOrdPlusBoundary(nn.Module):
    def __init__(self, pretrained=True, ord2seq_d_model=256, ord2seq_layers=2):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        feat_dim = m.fc.in_features
        self.fc_cls = nn.Linear(feat_dim, 4)
        
        self.ord_head = Ord2SeqOrdinalHead(
            in_features=feat_dim,
            num_classes=4,
            d_model=ord2seq_d_model,
            nhead=8,
            num_decoder_layers=ord2seq_layers,
            dim_feedforward=ord2seq_d_model * 4,
            dropout=0.1,
            use_masked_decision=True,
        )
        
        # 1. Global Gating (Dynamic OGAF)
        # feat(512) + p_s(4) + p_o(4) + H_s(1) + H_o(1) + M_s(1) + M_o(1) = 524
        self.gating = GatingNetwork(input_dim=feat_dim + 4 + 4 + 1 + 1 + 1 + 1, num_outputs=4)
        
        # 2. Boundary Router
        # feat(512) + p_g(4) + H_s(1) + H_o(1) + M_s(1) + M_o(1) + p0(1) + pA(1) + |p0-pA|(1) = 522
        self.router = GatingNetwork(input_dim=feat_dim + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1, num_outputs=1)
        
        # 3. 0/A Boundary Expert
        self.expert_0A = nn.Linear(feat_dim, 2)

    def compute_stats(self, p):
        entropy = -torch.sum(p * torch.log(p + 1e-8), dim=1, keepdim=True)
        top2_vals, _ = torch.topk(p, k=2, dim=1)
        margin = (top2_vals[:, 0] - top2_vals[:, 1]).unsqueeze(1)
        return entropy, margin

    def forward(self, x, labels=None):
        feat = self.backbone(x).flatten(1)
        logits_cls = self.fc_cls(feat)
        p_soft = torch.softmax(logits_cls, dim=1)
        
        o2s = self.ord_head(feat, labels=labels)
        p_ord = o2s["prob"]
        
        # Stats
        h_s, m_s = self.compute_stats(p_soft)
        h_o, m_o = self.compute_stats(p_ord)
        
        # --- Global Dynamic Fusion ---
        gating_input = torch.cat([feat, p_soft, p_ord, h_s, h_o, m_s, m_o], dim=1)
        alpha = torch.sigmoid(self.gating(gating_input)) # (B, 4)
        p_global = alpha * p_soft + (1.0 - alpha) * p_ord
        p_global = p_global / p_global.sum(dim=1, keepdim=True)
        
        # --- Boundary Router ---
        p0, pA = p_global[:, 0:1], p_global[:, 1:2]
        p0A_diff = torch.abs(p0 - pA)
        router_input = torch.cat([feat, p_global, h_s, h_o, m_s, m_o, p0, pA, p0A_diff], dim=1)
        r_0A = torch.sigmoid(self.router(router_input)) # (B, 1) - probability of being 0/A boundary
        
        # --- 0/A Boundary Expert ---
        logits_0A = self.expert_0A(feat) # (B, 2)
        p_0A_exp = torch.softmax(logits_0A, dim=1)
        
        # --- Boundary-Aware Correction ---
        # Only correct 0 and A classes.
        # p_final[0] = (1 - r*exp_A) * p_global[0] + r*exp_0 * p_global[1] ... No, simpler:
        # We redistribute the probability mass of (p_global[0] + p_global[1]) based on expert.
        p_0A_sum = p_global[:, 0:1] + p_global[:, 1:2]
        p_0_corr = p_0A_sum * p_0A_exp[:, 0:1]
        p_A_corr = p_0A_sum * p_0A_exp[:, 1:2]
        
        # Weighted blend of global and corrected based on router
        p_final = p_global.clone()
        p_final[:, 0:1] = (1.0 - r_0A) * p_global[:, 0:1] + r_0A * p_0_corr
        p_final[:, 1:2] = (1.0 - r_0A) * p_global[:, 1:2] + r_0A * p_A_corr
        
        out = {
            "logits_cls": logits_cls,
            "logits_0A": logits_0A,
            "p_soft": p_soft,
            "p_ord": p_ord,
            "p_global": p_global,
            "p_final": p_final,
            "r_0A": r_0A,
            "alpha": alpha,
            "loss_ord2seq": o2s.get("loss")
        }
        return out


def emd_loss(probs, labels, num_classes=4):
    onehot = F.one_hot(labels, num_classes=num_classes).float()
    cdf_p = torch.cumsum(probs, dim=1)
    cdf_t = torch.cumsum(onehot, dim=1)
    return ((cdf_p - cdf_t) ** 2).mean()


def qwk_numpy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    n, k = len(y_true), 4
    o = np.zeros((k, k))
    for t, p in zip(y_true, y_pred): o[t, p] += 1
    act_hist = np.bincount(y_true, minlength=k).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=k).astype(np.float64)
    e = np.outer(act_hist, pred_hist) / max(n, 1)
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            w[i, j] = ((i - j) ** 2) / float((k - 1) ** 2)
    num, den = (w * o).sum(), (w * e).sum()
    return 1.0 - num / den if den > 1e-12 else 0.0


def validate(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            p_final = out["p_final"]
            # Decision
            ge1 = 1.0 - p_final[:, 0]
            ge3 = p_final[:, 2] + p_final[:, 3]
            ge5 = p_final[:, 3]
            pred = torch.zeros_like(ge1, dtype=torch.long)
            pred = torch.where(ge1 >= 0.5, torch.ones_like(pred), pred)
            pred = torch.where(ge3 >= 0.5, torch.full_like(pred, 2), pred)
            pred = torch.where(ge5 >= 0.5, torch.full_like(pred, 3), pred)
            ys.append(labels.cpu().numpy())
            preds.append(pred.cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(preds)
    return float(np.mean(np.abs(y - p))), float(qwk_numpy(y, p))


def main(a):
    seed_everything(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_ds = Icdas4RoiDataset(a.train_csv, a.img_root_train, a.img_size, a.expand, augment=True)
    va_ds = Icdas4RoiDataset(a.val_csv, a.img_root_val, a.img_size, a.expand, augment=False)
    tr_loader = DataLoader(tr_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)

    model = ResNet18SoftmaxOrdPlusBoundary(pretrained=True).to(device)

    # Class weights
    labels_np = tr_ds.df["ic4"].values if "ic4" in tr_ds.df.columns else tr_ds.df["icdas"].map(map_ic4).values
    class_count = np.bincount(labels_np, minlength=4).astype(np.float32)
    class_weight = class_count.sum() / np.maximum(class_count, 1.0)
    class_weight[2] *= a.class_2_boost
    class_weight = class_weight / class_weight.mean()
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, device=device))
    
    # Expert 0/A Loss (Only for 0 and A samples)
    expert_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)

    best_score, best_mae, best_qwk = -1e9, 1e9, -1e9

    for epoch in range(a.epochs):
        model.train()
        run_loss = 0.0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs, labels=labels)
            
            p_final, p_soft, p_ord = out["p_final"], out["p_soft"], out["p_ord"]
            
            l_ce = ce_loss(out["logits_cls"], labels)
            l_ord = out["loss_ord2seq"]
            
            # Expert Loss: target is 0 or 1, others are ignored
            expert_labels = labels.clone()
            expert_labels = torch.where(expert_labels > 1, torch.full_like(expert_labels, -1), expert_labels)
            l_expert = expert_loss_fn(out["logits_0A"], expert_labels)
            
            # Global EMD
            l_emd = emd_loss(p_final, labels)
            if a.class_2_boost > 1.0:
                is_c2 = (labels == 2).float()
                sample_weights = 1.0 + (a.class_2_boost - 1.0) * is_c2
                onehot = F.one_hot(labels, 4).float()
                l_emd = (((torch.cumsum(p_final, 1) - torch.cumsum(onehot, 1))**2).mean(1) * sample_weights).mean()

            # Consistency Loss
            l_cons = F.l1_loss(p_soft, p_ord.detach())

            loss = a.l_ce * l_ce + a.l_ord * l_ord + a.l_emd * l_emd + a.l_cons * l_cons + a.l_exp * l_expert
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * imgs.size(0)

        scheduler.step()
        val_mae, val_qwk = validate(model, va_loader, device)
        score = val_qwk - 0.2 * val_mae
        if score > best_score:
            best_score, best_mae, best_qwk = score, val_mae, val_qwk
            torch.save({"model": model.state_dict(), "config": vars(a)}, a.out)
        print(f"Epoch {epoch+1} loss={run_loss/len(tr_ds):.4f} val_mae={val_mae:.4f} val_qwk={val_qwk:.4f} best_qwk={best_qwk:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--img_root_train", type=str, required=True)
    ap.add_argument("--img_root_val", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--expand", type=float, default=1.25)
    ap.add_argument("--class_2_boost", type=float, default=1.5)
    ap.add_argument("--l_ce", type=float, default=1.0)
    ap.add_argument("--l_ord", type=float, default=0.8)
    ap.add_argument("--l_emd", type=float, default=0.6)
    ap.add_argument("--l_cons", type=float, default=0.2)
    ap.add_argument("--l_exp", type=float, default=0.4, help="Loss weight for boundary expert")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", type=str, default="softmax_ordplus_o2s_boundary.pt")
    main(ap.parse_args())
