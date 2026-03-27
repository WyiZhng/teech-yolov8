import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix

from ord2seq_head import Ord2SeqOrdinalHead


def map_ic4(icdas):
    if icdas <= 0:
        return 0
    if icdas <= 2:
        return 1
    if icdas <= 4:
        return 2
    return 3


class ResNet18SoftmaxOrdPlus(nn.Module):
    def __init__(self, ord_mode="cond", ord2seq_d_model=256, ord2seq_layers=2):
        super().__init__()
        m = models.resnet18(weights=None)
        feat = m.fc.in_features
        self.backbone = nn.Sequential(*list(m.children())[:-1])
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

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        logits_cls = self.fc_cls(feat)
        out = {"logits_cls": logits_cls}
        if self.ord_mode == "cond":
            out["logits_ord"] = self.fc_ord(feat)
        else:
            o2s = self.ord_head(feat)
            out["p_ord"] = o2s["prob"]
        return out


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
    return p


def safe_auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if (y == 1).sum() > 0 and (y == 0).sum() > 0:
        return roc_auc_score(y, s)
    return float("nan")


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(a.val_csv)
    for c in ["image_id", "x", "y", "w", "h", "icdas"]:
        if c not in df.columns:
            raise SystemExit(f"CSV missing column: {c}")

    df["image_id"] = df["image_id"].astype(str).map(os.path.basename)
    df["ic4"] = df["icdas"].map(map_ic4)

    ckpt = torch.load(a.ckpt, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    if "ord_mode" in cfg:
        ord_mode = cfg["ord_mode"]
    elif any(k.startswith("ord_head.") for k in sd.keys()):
        ord_mode = "ord2seq"
    else:
        ord_mode = "cond"

    d_model = int(cfg.get("ord2seq_d_model", 256))
    n_layers = int(cfg.get("ord2seq_layers", 2))
    if ord_mode == "ord2seq":
        if "ord_head.feature_proj.weight" in sd:
            d_model = int(sd["ord_head.feature_proj.weight"].shape[0])
        layer_ids = set()
        p = "ord_head.decoder.layers."
        for k in sd.keys():
            if p in k:
                idx = k.split(p, 1)[1].split(".", 1)[0]
                if idx.isdigit():
                    layer_ids.add(int(idx))
        if layer_ids:
            n_layers = max(layer_ids) + 1

    model = ResNet18SoftmaxOrdPlus(ord_mode=ord_mode, ord2seq_d_model=d_model, ord2seq_layers=n_layers).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    tx = transforms.Compose(
        [
            transforms.Resize((a.img_size, a.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def crop_tensor(im_path, x, y, w, h, expand=1.25):
        with Image.open(im_path) as im:
            im = im.convert("RGB")
            w_img, h_img = im.size
            cx, cy = x + w / 2, y + h / 2
            w2, h2 = w * expand, h * expand
            x1 = max(0, cx - w2 / 2)
            y1 = max(0, cy - h2 / 2)
            x2 = min(w_img, cx + w2 / 2)
            y2 = min(h_img, cy + h2 / 2)
            return tx(im.crop((x1, y1, x2, y2)))

    ps_ge1, ps_ge3, ps_ge5 = [], [], []
    batch = []
    with torch.no_grad():
        for i, r in df.iterrows():
            img_path = os.path.join(a.img_root, r["image_id"])
            batch.append(
                crop_tensor(
                    img_path,
                    float(r["x"]),
                    float(r["y"]),
                    float(r["w"]),
                    float(r["h"]),
                    a.expand,
                )
            )
            if len(batch) == a.bs or i == len(df) - 1:
                x = torch.stack(batch).to(device)
                out = model(x)
                logits_cls = out["logits_cls"]
                p_soft = torch.softmax(logits_cls, dim=1)
                if model.ord_mode == "cond":
                    p_ord = ord_probs_from_conditional(out["logits_ord"])
                else:
                    p_ord = out["p_ord"]
                alpha = torch.sigmoid(model.alpha_logits).unsqueeze(0)
                p = alpha * p_soft + (1.0 - alpha) * p_ord
                p = p / p.sum(dim=1, keepdim=True)
                p = p.cpu().numpy()

                ge1 = 1.0 - p[:, 0]
                ge3 = p[:, 2] + p[:, 3]
                ge5 = p[:, 3]
                ps_ge1.extend(ge1.tolist())
                ps_ge3.extend(ge3.tolist())
                ps_ge5.extend(ge5.tolist())
                batch = []

    df["p_ge1"] = ps_ge1
    df["p_ge3"] = ps_ge3
    df["p_ge5"] = ps_ge5

    auc_ge1 = safe_auc(df["icdas"] >= 1, df["p_ge1"])
    auc_ge3 = safe_auc(df["icdas"] >= 3, df["p_ge3"])
    auc_ge5 = safe_auc(df["icdas"] >= 5, df["p_ge5"])

    def pred_ic4_row(r, t=0.5):
        if r["p_ge1"] < t:
            return 0
        if r["p_ge3"] < t:
            return 1
        if r["p_ge5"] < t:
            return 2
        return 3

    df["pred_ic4"] = df.apply(pred_ic4_row, axis=1)
    mae = float(np.mean(np.abs(df["pred_ic4"] - df["ic4"])))
    qwk = cohen_kappa_score(df["ic4"], df["pred_ic4"], weights="quadratic")
    cm = confusion_matrix(df["ic4"], df["pred_ic4"], labels=[0, 1, 2, 3])

    print(f"AUC(>=1/>=3/>=5)= {auc_ge1:.3f}/{auc_ge3:.3f}/{auc_ge5:.3f}")
    print(f"4-class MAE={mae:.3f} QWK={qwk:.3f}")
    print("Confusion Matrix (rows=GT 0/A/B/C, cols=Pred):")
    print(cm)

    if a.out_csv:
        df.to_csv(a.out_csv, index=False)
        print("Saved per-ROI preds to:", a.out_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--expand", type=float, default=1.25)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()
    main(args)
