import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def map_ic4(icdas):
    if icdas <= 0:
        return 0
    if icdas <= 2:
        return 1
    if icdas <= 4:
        return 2
    return 3


class Icdas4RoiDataset(Dataset):
    def __init__(self, csv_path, img_root, img_size=256, expand=1.25, augment=False):
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
        img = Image.open(os.path.join(self.root, r["image_id"])).convert("RGB")
        w_img, h_img = img.size

        if "gx" in r:
            x, y, w, h = float(r["gx"]), float(r["gy"]), float(r["gw"]), float(r["gh"])
        else:
            x, y, w, h = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])

        cx, cy = x + w / 2.0, y + h / 2.0
        w2, h2 = w * self.expand, h * self.expand
        x1 = max(0, cx - w2 / 2.0)
        y1 = max(0, cy - h2 / 2.0)
        x2 = min(w_img, cx + w2 / 2.0)
        y2 = min(h_img, cy + h2 / 2.0)
        crop = img.crop((x1, y1, x2, y2))
        if self.augment:
            crop = transforms.functional.hflip(crop)
        crop = self.tx(crop)

        if "ic4" in r:
            y4 = int(r["ic4"])
        else:
            y4 = map_ic4(int(r["icdas"]))

        return crop, y4, i


class ResNet18Coral(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        m = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.fc = nn.Linear(m.fc.in_features, num_classes - 1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.flatten(1)
        return self.fc(feat)


def safe_auc(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if (y_true == 1).sum() > 0 and (y_true == 0).sum() > 0:
        return float(roc_auc_score(y_true, scores))
    return float("nan")


def evaluate_split(csv_path, img_root, ckpt, out_csv, img_size=256, expand=1.25, bs=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = Icdas4RoiDataset(csv_path, img_root, img_size=img_size, expand=expand, augment=False)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    model = ResNet18Coral(num_classes=4).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    probs_all = []
    y_all = []
    idx_all = []

    with torch.no_grad():
        for imgs, labels, idxs in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()  # [B,3]
            probs_all.append(probs)
            y_all.append(labels.numpy())
            idx_all.append(idxs.numpy())

    probs = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, 3), dtype=np.float32)
    y_true = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,), dtype=np.int64)
    row_idx = np.concatenate(idx_all, axis=0) if idx_all else np.zeros((0,), dtype=np.int64)

    p_ge1 = probs[:, 0] if len(probs) else np.array([])
    p_ge3 = probs[:, 1] if len(probs) else np.array([])
    p_ge5 = probs[:, 2] if len(probs) else np.array([])

    pred_class = (probs > 0.5).sum(axis=1).astype(np.int64) if len(probs) else np.array([], dtype=np.int64)

    auc_ge1 = safe_auc((y_true >= 1).astype(int), p_ge1)
    auc_ge3 = safe_auc((y_true >= 2).astype(int), p_ge3)
    auc_ge5 = safe_auc((y_true >= 3).astype(int), p_ge5)

    mae = float(mean_absolute_error(y_true, pred_class)) if len(y_true) else float("nan")
    qwk = float(cohen_kappa_score(y_true, pred_class, weights="quadratic")) if len(y_true) else float("nan")
    cm = confusion_matrix(y_true, pred_class, labels=[0, 1, 2, 3]) if len(y_true) else np.zeros((4, 4), dtype=np.int64)

    src = pd.read_csv(csv_path)
    out_df = src.iloc[row_idx].copy() if len(row_idx) else src.iloc[:0].copy()
    out_df = out_df.reset_index(drop=True)
    out_df["roi_id"] = np.arange(len(out_df))
    out_df["gt_class"] = y_true
    out_df["pred_class"] = pred_class
    out_df["p_ge1"] = p_ge1
    out_df["p_ge3"] = p_ge3
    out_df["p_ge5"] = p_ge5

    keep_cols = ["image_id", "roi_id", "gt_class", "pred_class", "p_ge1", "p_ge3", "p_ge5"]
    out_df[keep_cols].to_csv(out_csv, index=False)

    print(f"Split CSV: {csv_path}")
    print(f"AUC(>=1/>=3/>=5)= {auc_ge1:.3f}/{auc_ge3:.3f}/{auc_ge5:.3f}")
    print(f"MAE={mae:.3f}  QWK={qwk:.3f}")
    print("Confusion Matrix (rows=GT 0/A/B/C, cols=Pred):")
    print(cm)
    print(f"Saved per-ROI predictions to: {out_csv}")

    return {
        "auc_ge1": auc_ge1,
        "auc_ge3": auc_ge3,
        "auc_ge5": auc_ge5,
        "mae": mae,
        "qwk": qwk,
    }


def main(a):
    if a.val_csv:
        evaluate_split(
            csv_path=a.val_csv,
            img_root=a.img_root,
            ckpt=a.ckpt,
            out_csv=a.out_val_csv,
            img_size=a.img_size,
            expand=a.expand,
            bs=a.bs,
        )

    if a.test_csv:
        evaluate_split(
            csv_path=a.test_csv,
            img_root=a.img_root,
            ckpt=a.ckpt,
            out_csv=a.out_test_csv,
            img_size=a.img_size,
            expand=a.expand,
            bs=a.bs,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", type=str, default="icdas4_val.csv")
    ap.add_argument("--test_csv", type=str, default="icdas4_test.csv")
    ap.add_argument("--img_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="coral_head_icdas4.pt")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--expand", type=float, default=1.25)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--out_val_csv", type=str, default="roi_val_icdas4_coral.csv")
    ap.add_argument("--out_test_csv", type=str, default="roi_test_icdas4_coral.csv")
    args = ap.parse_args()
    main(args)
