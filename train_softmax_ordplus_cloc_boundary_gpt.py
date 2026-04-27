import argparse
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from roi_ordinal_baselines_common import load_cloc_loss_module


class _NoOpWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_custom_scalars(self, *args, **kwargs):
        return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def map_ic4(icdas: int) -> int:
    if icdas <= 0:
        return 0
    if icdas <= 2:
        return 1
    if icdas <= 4:
        return 2
    return 3


class Icdas4RoiDataset(Dataset):
    def __init__(self, csv_path: str, img_root: str, img_size: int = 256, expand: float = 1.25, augment: bool = True):
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

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        im = Image.open(os.path.join(self.root, str(r["image_id"]))).convert("RGB")
        w_img, h_img = im.size

        x = float(r.get("gx", r.get("x")))
        y = float(r.get("gy", r.get("y")))
        w = float(r.get("gw", r.get("w")))
        h = float(r.get("gh", r.get("h")))

        cx, cy = x + w / 2.0, y + h / 2.0
        w2, h2 = w * self.expand, h * self.expand
        x1, y1 = max(0.0, cx - w2 / 2.0), max(0.0, cy - h2 / 2.0)
        x2, y2 = min(float(w_img), cx + w2 / 2.0), min(float(h_img), cy + h2 / 2.0)
        crop = im.crop((x1, y1, x2, y2))

        if self.augment and random.random() < 0.5:
            crop = transforms.functional.hflip(crop)

        img = self.tx(crop)
        y4 = int(r["ic4"]) if "ic4" in r else map_ic4(int(r["icdas"]))
        return img, y4


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet18SoftmaxOrdPlusClocBoundary(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        gate_hidden: int = 128,
        router_hidden: int = 128,
        boundary_tau: float = 0.15,
        boundary_mass: float = 0.65,
    ):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        feat_dim = m.fc.in_features

        self.fc_cls = nn.Linear(feat_dim, 4)
        self.fc_ord = nn.Linear(feat_dim, 4)

        gate_in = feat_dim + 4 + 4 + 1 + 1 + 1 + 1
        self.gating = MLP(gate_in, gate_hidden, 4)

        router_in = feat_dim + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1
        self.router = MLP(router_in, router_hidden, 1)

        self.expert_0A = nn.Linear(feat_dim, 2)

        self.boundary_tau = boundary_tau
        self.boundary_mass = boundary_mass

    @staticmethod
    def compute_stats(p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entropy = -torch.sum(p * torch.log(p + 1e-8), dim=1, keepdim=True)
        top2_vals, top2_idx = torch.topk(p, k=2, dim=1)
        margin = (top2_vals[:, 0] - top2_vals[:, 1]).unsqueeze(1)
        return entropy, margin, top2_idx

    def build_0A_candidate_mask(self, p_global: torch.Tensor) -> torch.Tensor:
        _, _, top2_idx = self.compute_stats(p_global)
        has_0 = (top2_idx == 0).any(dim=1)
        has_A = (top2_idx == 1).any(dim=1)
        pair_mask = has_0 & has_A

        p0 = p_global[:, 0:1]
        pA = p_global[:, 1:2]
        close_mask = ((torch.abs(p0 - pA) < self.boundary_tau) & ((p0 + pA) > self.boundary_mass)).squeeze(1)

        candidate_mask = (pair_mask | close_mask).float().unsqueeze(1)
        return candidate_mask

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x).flatten(1)

        logits_cls = self.fc_cls(feat)
        p_soft = torch.softmax(logits_cls, dim=1)

        logits_ord = self.fc_ord(feat)
        p_ord = torch.softmax(logits_ord, dim=1)

        h_s, m_s, _ = self.compute_stats(p_soft)
        h_o, m_o, _ = self.compute_stats(p_ord)

        gate_in = torch.cat([feat, p_soft, p_ord, h_s, h_o, m_s, m_o], dim=1)
        alpha = torch.sigmoid(self.gating(gate_in))

        p_global = alpha * p_soft + (1.0 - alpha) * p_ord
        p_global = p_global / p_global.sum(dim=1, keepdim=True)

        p0 = p_global[:, 0:1]
        pA = p_global[:, 1:2]
        p0A_diff = torch.abs(p0 - pA)

        candidate_mask = self.build_0A_candidate_mask(p_global)

        router_in = torch.cat([feat, p_global, h_s, h_o, m_s, m_o, p0, pA, p0A_diff], dim=1)
        route_logits = self.router(router_in)
        route_prob = torch.sigmoid(route_logits) * candidate_mask

        logits_0A = self.expert_0A(feat)
        p_0A_exp = torch.softmax(logits_0A, dim=1)

        p_0A_sum = p_global[:, 0:1] + p_global[:, 1:2]
        p0_corr = p_0A_sum * p_0A_exp[:, 0:1]
        pA_corr = p_0A_sum * p_0A_exp[:, 1:2]

        p_final = p_global.clone()
        beta = route_prob
        p_final[:, 0:1] = (1.0 - beta) * p_global[:, 0:1] + beta * p0_corr
        p_final[:, 1:2] = (1.0 - beta) * p_global[:, 1:2] + beta * pA_corr
        p_final = p_final / p_final.sum(dim=1, keepdim=True)

        return {
            "feat": feat,
            "logits_cls": logits_cls,
            "logits_ord": logits_ord,
            "logits_0A": logits_0A,
            "p_soft": p_soft,
            "p_ord": p_ord,
            "p_global": p_global,
            "p_final": p_final,
            "alpha": alpha,
            "candidate_mask": candidate_mask,
            "route_logits": route_logits,
            "route_prob": route_prob,
        }


def emd_loss(probs: torch.Tensor, labels: torch.Tensor, num_classes: int = 4, sample_weights: torch.Tensor = None) -> torch.Tensor:
    onehot = F.one_hot(labels, num_classes=num_classes).float()
    cdf_p = torch.cumsum(probs, dim=1)
    cdf_t = torch.cumsum(onehot, dim=1)
    per_sample = ((cdf_p - cdf_t) ** 2).mean(dim=1)
    if sample_weights is not None:
        per_sample = per_sample * sample_weights
    return per_sample.mean()


def qwk_numpy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n, k = len(y_true), 4
    o = np.zeros((k, k))
    for t, p in zip(y_true, y_pred):
        o[t, p] += 1
    act_hist = np.bincount(y_true, minlength=k).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=k).astype(np.float64)
    e = np.outer(act_hist, pred_hist) / max(n, 1)
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            w[i, j] = ((i - j) ** 2) / float((k - 1) ** 2)
    num, den = (w * o).sum(), (w * e).sum()
    return 1.0 - num / den if den > 1e-12 else 0.0


def decode_prediction(p: torch.Tensor, mode: str = "ordinal") -> torch.Tensor:
    if mode == "argmax":
        return p.argmax(dim=1)

    ge1 = 1.0 - p[:, 0]
    ge3 = p[:, 2] + p[:, 3]
    ge5 = p[:, 3]
    pred = torch.zeros_like(ge1, dtype=torch.long)
    pred = torch.where(ge1 >= 0.5, torch.ones_like(pred), pred)
    pred = torch.where(ge3 >= 0.5, torch.full_like(pred, 2), pred)
    pred = torch.where(ge5 >= 0.5, torch.full_like(pred, 3), pred)
    return pred


@torch.no_grad()
def validate(model, loader, device, decode_mode: str = "ordinal"):
    model.eval()
    ys, preds = [], []
    route_mean = []
    cand_mean = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        p_final = out["p_final"]
        pred = decode_prediction(p_final, mode=decode_mode)

        ys.append(labels.cpu().numpy())
        preds.append(pred.cpu().numpy())
        route_mean.append(out["route_prob"].mean().item())
        cand_mean.append(out["candidate_mask"].mean().item())

    y = np.concatenate(ys)
    p = np.concatenate(preds)
    mae = float(np.mean(np.abs(y - p)))
    qwk = float(qwk_numpy(y, p))
    return mae, qwk, float(np.mean(route_mean)), float(np.mean(cand_mean))


def build_cloc_criterion(module, device, margin_mode: str):
    writer = _NoOpWriter()
    if margin_mode == "sm":
        return module.OrdinalContrastiveLoss_sm(
            n_classes=4,
            device=device,
            learnable_map=None,
            summaryWriter=writer,
        )
    return module.OrdinalContrastiveLoss_mm(
        n_classes=4,
        device=device,
        learnable_map=None,
        summaryWriter=writer,
    )


def main(a):
    seed_everything(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cloc_module = load_cloc_loss_module(".")

    tr_ds = Icdas4RoiDataset(a.train_csv, a.img_root_train, a.img_size, a.expand, augment=True)
    va_ds = Icdas4RoiDataset(a.val_csv, a.img_root_val, a.img_size, a.expand, augment=False)

    tr_loader = DataLoader(tr_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)

    model = ResNet18SoftmaxOrdPlusClocBoundary(
        pretrained=True,
        gate_hidden=a.gate_hidden,
        router_hidden=a.router_hidden,
        boundary_tau=a.boundary_tau,
        boundary_mass=a.boundary_mass,
    ).to(device)

    cloc_criterion = build_cloc_criterion(cloc_module, device, a.margin_mode)

    labels_np = tr_ds.df["ic4"].values if "ic4" in tr_ds.df.columns else tr_ds.df["icdas"].map(map_ic4).values
    class_count = np.bincount(labels_np, minlength=4).astype(np.float32)
    class_weight = class_count.sum() / np.maximum(class_count, 1.0)
    class_weight[2] *= a.class_2_boost
    class_weight = class_weight / class_weight.mean()
    class_weight_t = torch.tensor(class_weight, dtype=torch.float32, device=device)

    ce_cls_fn = nn.CrossEntropyLoss(weight=class_weight_t)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(cloc_criterion.parameters()), lr=a.lr, weight_decay=a.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)

    best_score = -1e9
    best_mae, best_qwk = 1e9, -1e9
    global_step = 0

    for epoch in range(a.epochs):
        model.train()
        run_loss = 0.0

        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)

            p_soft = out["p_soft"]
            p_ord = out["p_ord"]
            p_global = out["p_global"]
            p_final = out["p_final"]
            logits_cls = out["logits_cls"]
            logits_ord = out["logits_ord"]
            logits_0A = out["logits_0A"]
            candidate_mask = out["candidate_mask"]
            route_logits = out["route_logits"]

            l_ce_cls = ce_cls_fn(logits_cls, labels)
            l_ce_fuse = F.nll_loss(torch.log(p_final + 1e-8), labels, weight=class_weight_t)
            l_cloc_ce = ce_cls_fn(logits_ord, labels)
            l_cloc_reg = cloc_criterion(logits_ord, labels, global_step)
            l_ord = l_cloc_ce + a.cloc_weight * l_cloc_reg

            is_c2 = (labels == 2).float()
            sample_weights = 1.0 + (a.class_2_boost - 1.0) * is_c2
            l_emd = emd_loss(p_final, labels, sample_weights=sample_weights)

            l_cons = F.l1_loss(p_soft, p_ord.detach())

            valid_0A = labels <= 1
            safe_labels = labels.clone()
            safe_labels[~valid_0A] = 0
            correct_prob = p_global.gather(1, safe_labels.unsqueeze(1)).squeeze(1).detach()
            hardness = 1.0 - correct_prob

            expert_mask = valid_0A & ((candidate_mask.squeeze(1) > 0.5) | (hardness > a.expert_hard_thr))

            if expert_mask.any():
                exp_logits = logits_0A[expert_mask]
                exp_targets = labels[expert_mask]
                exp_ce = F.cross_entropy(exp_logits, exp_targets, reduction="none")
                exp_weights = 1.0 + a.expert_hard_boost * hardness[expert_mask]
                l_exp = (exp_ce * exp_weights).mean()
            else:
                l_exp = torch.zeros((), device=device)

            route_target = expert_mask.float().unsqueeze(1)
            route_valid = (candidate_mask > 0.5) | (route_target > 0.5)

            if route_valid.any():
                rv_logits = route_logits[route_valid]
                rv_target = route_target[route_valid]
                rv_bce = F.binary_cross_entropy_with_logits(rv_logits, rv_target, reduction="none")
                rv_weight = torch.where(
                    rv_target > 0.5,
                    torch.full_like(rv_target, a.route_pos_weight),
                    torch.ones_like(rv_target),
                )
                l_route = (rv_bce * rv_weight).mean()
            else:
                l_route = torch.zeros((), device=device)

            loss = (
                a.l_ce * l_ce_cls
                + a.l_fce * l_ce_fuse
                + a.l_ord * l_ord
                + a.l_emd * l_emd
                + a.l_cons * l_cons
                + a.l_exp * l_exp
                + a.l_route * l_route
            )

            optimizer.zero_grad()
            loss.backward()
            if a.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            optimizer.step()

            run_loss += loss.item() * imgs.size(0)
            global_step += 1

        scheduler.step()

        val_mae, val_qwk, route_avg, cand_avg = validate(model, va_loader, device, decode_mode=a.decode_mode)
        score = val_qwk - a.score_mae_weight * val_mae

        if score > best_score:
            best_score = score
            best_mae = val_mae
            best_qwk = val_qwk
            torch.save(
                {
                    "model": model.state_dict(),
                    "cloc_state": cloc_criterion.state_dict(),
                    "config": vars(a),
                    "best_mae": best_mae,
                    "best_qwk": best_qwk,
                },
                a.out,
            )

        print(
            f"Epoch {epoch + 1:03d} "
            f"loss={run_loss / len(tr_ds):.4f} "
            f"val_mae={val_mae:.4f} "
            f"val_qwk={val_qwk:.4f} "
            f"route_avg={route_avg:.4f} "
            f"cand_avg={cand_avg:.4f} "
            f"best_qwk={best_qwk:.4f}"
        )


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
    ap.add_argument("--gate_hidden", type=int, default=128)
    ap.add_argument("--router_hidden", type=int, default=128)
    ap.add_argument("--boundary_tau", type=float, default=0.15)
    ap.add_argument("--boundary_mass", type=float, default=0.65)
    ap.add_argument("--class_2_boost", type=float, default=1.5)
    ap.add_argument("--expert_hard_thr", type=float, default=0.25)
    ap.add_argument("--expert_hard_boost", type=float, default=1.0)
    ap.add_argument("--route_pos_weight", type=float, default=3.0)
    ap.add_argument("--cloc_weight", type=float, default=1.0)
    ap.add_argument("--margin_mode", type=str, default="mm", choices=["mm", "sm"])
    ap.add_argument("--l_ce", type=float, default=1.0)
    ap.add_argument("--l_fce", type=float, default=1.0)
    ap.add_argument("--l_ord", type=float, default=0.8)
    ap.add_argument("--l_emd", type=float, default=0.6)
    ap.add_argument("--l_cons", type=float, default=0.2)
    ap.add_argument("--l_exp", type=float, default=0.4)
    ap.add_argument("--l_route", type=float, default=0.2)
    ap.add_argument("--decode_mode", type=str, default="ordinal", choices=["ordinal", "argmax"])
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--score_mae_weight", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", type=str, default="softmax_ordplus_cloc_boundary.pt")
    main(ap.parse_args())
