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


def map_ic4(icdas: int) -> int:
    if icdas <= 0:
        return 0
    if icdas <= 2:
        return 1
    if icdas <= 4:
        return 2
    return 3


class Icdas4RoiDataset(Dataset):
    def __init__(self, csv_path, img_root, img_size=256, expand=1.25, augment=True):
        self.df = pd.read_csv(csv_path)
        self.root = img_root
        self.expand = expand
        self.augment = augment
        self.tx = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(os.path.join(self.root, str(r["image_id"]))).convert("RGB")
        w_img, h_img = im.size

        if "gx" in r:
            x, y, w, h = float(r["gx"]), float(r["gy"]), float(r["gw"]), float(r["gh"])
        else:
            x, y, w, h = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])

        cx, cy = x + w / 2, y + h / 2
        w2, h2 = w * self.expand, h * self.expand
        x1 = max(0, cx - w2 / 2)
        y1 = max(0, cy - h2 / 2)
        x2 = min(w_img, cx + w2 / 2)
        y2 = min(h_img, cy + h2 / 2)
        crop = im.crop((x1, y1, x2, y2))

        if self.augment and random.random() < 0.5:
            crop = transforms.functional.hflip(crop)

        img = self.tx(crop)

        if "ic4" in r:
            y4 = int(r["ic4"])
        else:
            y4 = map_ic4(int(r["icdas"]))
        return img, y4


class ResNet18SoftmaxOrdPlus(nn.Module):
    def __init__(
        self,
        pretrained=True,
        ord_mode="ord2seq",
        ord2seq_d_model=256,
        ord2seq_layers=2,
    ):
        super().__init__()
        m = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        feat = m.fc.in_features
        self.fc_cls = nn.Linear(feat, 4)
        self.ord_mode = ord_mode
        if self.ord_mode == "cond":
            self.fc_ord = nn.Linear(feat, 3)
            self.ord_head = None
        elif self.ord_mode == "ord2seq":
            self.fc_ord = None
            self.ord_head = Ord2SeqOrdinalHead(
                in_features=feat,
                num_classes=4,
                d_model=ord2seq_d_model,
                nhead=8,
                num_decoder_layers=ord2seq_layers,
                dim_feedforward=ord2seq_d_model * 4,
                dropout=0.1,
                use_masked_decision=True,
            )
        else:
            raise ValueError(f"Unknown ord_mode: {self.ord_mode}")
        self.alpha_logits = nn.Parameter(torch.zeros(4))

    def forward(self, x, labels=None):
        feat = self.backbone(x).flatten(1)
        logits_cls = self.fc_cls(feat)
        out = {"logits_cls": logits_cls}
        if self.ord_mode == "cond":
            out["logits_ord"] = self.fc_ord(feat)
        else:
            o2s = self.ord_head(feat, labels=labels)
            out["p_ord"] = o2s["prob"]
            if "loss" in o2s:
                out["loss_ord2seq"] = o2s["loss"]
        return out


class SoftQWKLoss(nn.Module):
    def __init__(self, num_classes=4, eps=1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        w = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = ((i - j) ** 2) / float((num_classes - 1) ** 2)
        self.register_buffer("W", w)

    def forward(self, probs, labels):
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        o = y_onehot.transpose(0, 1) @ probs
        o = o / (o.sum() + self.eps)

        hist_true = y_onehot.sum(dim=0)
        hist_pred = probs.sum(dim=0)
        e = torch.outer(hist_true, hist_pred)
        e = e / (e.sum() + self.eps)

        num = (self.W * o).sum()
        den = (self.W * e).sum() + self.eps
        return num / den


def ord_probs_from_conditional(logits_ord):
    q = torch.sigmoid(logits_ord)
    ge1 = q[:, 0]
    ge2 = q[:, 0] * q[:, 1]
    ge3 = q[:, 0] * q[:, 1] * q[:, 2]

    p0 = 1.0 - ge1
    p1 = ge1 - ge2
    p2 = ge2 - ge3
    p3 = ge3
    p = torch.stack([p0, p1, p2, p3], dim=1)
    p = torch.clamp(p, min=1e-6)
    p = p / p.sum(dim=1, keepdim=True)
    return p, torch.stack([ge1, ge2, ge3], dim=1)


def cumulative_targets(labels):
    ge1 = (labels >= 1).float()
    ge2 = (labels >= 2).float()
    ge3 = (labels >= 3).float()
    return torch.stack([ge1, ge2, ge3], dim=1)


def emd_loss(probs, labels, num_classes=4):
    onehot = F.one_hot(labels, num_classes=num_classes).float()
    cdf_p = torch.cumsum(probs, dim=1)
    cdf_t = torch.cumsum(onehot, dim=1)
    return ((cdf_p - cdf_t) ** 2).mean()


def predict_ic4_from_probs(p_fuse, t=0.5):
    ge1 = 1.0 - p_fuse[:, 0]
    ge3 = p_fuse[:, 2] + p_fuse[:, 3]
    ge5 = p_fuse[:, 3]

    pred = torch.zeros_like(ge1, dtype=torch.long)
    pred = torch.where(ge1 >= t, torch.ones_like(pred), pred)
    pred = torch.where(ge3 >= t, torch.full_like(pred, 2), pred)
    pred = torch.where(ge5 >= t, torch.full_like(pred, 3), pred)
    return pred


def qwk_numpy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(y_true)
    k = 4
    o = np.zeros((k, k), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        o[t, p] += 1.0
    act_hist = np.bincount(y_true, minlength=k).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=k).astype(np.float64)
    e = np.outer(act_hist, pred_hist) / max(n, 1)

    w = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            w[i, j] = ((i - j) ** 2) / float((k - 1) ** 2)

    num = (w * o).sum()
    den = (w * e).sum()
    if den <= 1e-12:
        return 0.0
    return 1.0 - num / den


def cumulative_from_probs(p4):
    ge1 = 1.0 - p4[:, 0]
    ge3 = p4[:, 2] + p4[:, 3]
    ge5 = p4[:, 3]
    return torch.stack([ge1, ge3, ge5], dim=1)


def kd_kl_div(logits_student, probs_teacher, temperature=2.0):
    log_p_student = F.log_softmax(logits_student / temperature, dim=1)
    p_teacher = torch.clamp(probs_teacher, min=1e-6)
    p_teacher = p_teacher / p_teacher.sum(dim=1, keepdim=True)
    loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
    return loss * (temperature * temperature)


def validate(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            logits_cls = out["logits_cls"]
            p_soft = torch.softmax(logits_cls, dim=1)
            if model.ord_mode == "cond":
                p_ord, _ = ord_probs_from_conditional(out["logits_ord"])
            else:
                p_ord = out["p_ord"]
            alpha = torch.sigmoid(model.alpha_logits).unsqueeze(0)
            p_fuse = alpha * p_soft + (1.0 - alpha) * p_ord
            p_fuse = p_fuse / p_fuse.sum(dim=1, keepdim=True)
            pred = predict_ic4_from_probs(p_fuse, t=0.5)

            ys.append(labels.cpu().numpy())
            preds.append(pred.cpu().numpy())

    y = np.concatenate(ys)
    p = np.concatenate(preds)
    mae = float(np.mean(np.abs(y - p)))
    qwk = float(qwk_numpy(y, p))
    return mae, qwk


def main(a):
    seed_everything(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_ds = Icdas4RoiDataset(a.train_csv, a.img_root_train, a.img_size, a.expand, augment=True)
    va_ds = Icdas4RoiDataset(a.val_csv, a.img_root_val, a.img_size, a.expand, augment=False)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=a.bs,
        shuffle=True,
        num_workers=a.workers,
        pin_memory=True,
        drop_last=False,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=a.bs,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=True,
        drop_last=False,
    )

    model = ResNet18SoftmaxOrdPlus(
        pretrained=True,
        ord_mode=a.ord_mode,
        ord2seq_d_model=a.ord2seq_d_model,
        ord2seq_layers=a.ord2seq_layers,
    ).to(device)

    labels_np = tr_ds.df["ic4"].values if "ic4" in tr_ds.df.columns else tr_ds.df["icdas"].map(map_ic4).values
    class_count = np.bincount(labels_np, minlength=4).astype(np.float32)
    class_weight = class_count.sum() / np.maximum(class_count, 1.0)
    class_weight = class_weight / class_weight.mean()
    class_weight_t = torch.tensor(class_weight, device=device, dtype=torch.float32)

    ce_loss = nn.CrossEntropyLoss(weight=class_weight_t, label_smoothing=a.label_smoothing)
    bce_loss = nn.BCELoss()
    qwk_loss = SoftQWKLoss(num_classes=4).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)

    best_score = -1e9
    best_mae = 1e9
    best_qwk = -1e9

    for epoch in range(a.epochs):
        model.train()
        run_loss = 0.0

        for imgs, labels in tr_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(imgs, labels=labels if a.ord_mode == "ord2seq" else None)
            logits_cls = out["logits_cls"]
            p_soft = torch.softmax(logits_cls, dim=1)
            if a.ord_mode == "cond":
                p_ord, ge_ord = ord_probs_from_conditional(out["logits_ord"])
                loss_ord = bce_loss(ge_ord, cumulative_targets(labels))
            else:
                p_ord = out["p_ord"]
                ge_ord = cumulative_from_probs(p_ord)
                loss_ord = out["loss_ord2seq"]

            alpha = torch.sigmoid(model.alpha_logits).unsqueeze(0)
            p_fuse = alpha * p_soft + (1.0 - alpha) * p_ord
            p_fuse = p_fuse / p_fuse.sum(dim=1, keepdim=True)

            ge_soft = cumulative_from_probs(p_soft)

            loss_ce = ce_loss(logits_cls, labels)
            loss_emd = emd_loss(p_fuse, labels, num_classes=4)
            loss_qwk = qwk_loss(p_fuse, labels)
            loss_cons = F.l1_loss(ge_soft, ge_ord)
            loss_kd = kd_kl_div(logits_cls, p_ord.detach(), temperature=a.temperature)

            loss = (
                a.lambda_ce * loss_ce
                + a.lambda_ord * loss_ord
                + a.lambda_emd * loss_emd
                + a.lambda_qwk * loss_qwk
                + a.lambda_cons * loss_cons
                + a.lambda_kd * loss_kd
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * imgs.size(0)

        scheduler.step()

        tr_loss = run_loss / max(len(tr_ds), 1)
        val_mae, val_qwk = validate(model, va_loader, device)
        score = val_qwk - a.mae_tradeoff * val_mae

        if score > best_score:
            best_score = score
            best_mae = val_mae
            best_qwk = val_qwk
            ckpt = {
                "model": model.state_dict(),
                "config": vars(a),
                "best_val_mae": best_mae,
                "best_val_qwk": best_qwk,
                "best_val_score": best_score,
            }
            torch.save(ckpt, a.out)

        print(
            f"Epoch {epoch + 1}/{a.epochs} loss={tr_loss:.4f} "
            f"val_mae={val_mae:.4f} val_qwk={val_qwk:.4f} "
            f"best_mae={best_mae:.4f} best_qwk={best_qwk:.4f}"
        )

    print(f"Saved best model to: {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--img_root_train", type=str, required=True)
    ap.add_argument("--img_root_val", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--expand", type=float, default=1.25)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label_smoothing", type=float, default=0.02)

    ap.add_argument("--ord_mode", type=str, default="ord2seq", choices=["cond", "ord2seq"])
    ap.add_argument("--ord2seq_d_model", type=int, default=256)
    ap.add_argument("--ord2seq_layers", type=int, default=2)

    ap.add_argument("--lambda_ce", type=float, default=1.0)
    ap.add_argument("--lambda_ord", type=float, default=0.8)
    ap.add_argument("--lambda_emd", type=float, default=0.6)
    ap.add_argument("--lambda_qwk", type=float, default=0.25)
    ap.add_argument("--lambda_cons", type=float, default=0.2)
    ap.add_argument("--lambda_kd", type=float, default=0.25)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--mae_tradeoff", type=float, default=0.20)

    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", type=str, default="softmax_ordplus_icdas4.pt")
    args = ap.parse_args()
    main(args)
