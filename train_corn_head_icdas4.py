import argparse
import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def map_ic4(icdas):
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
        img = Image.open(os.path.join(self.root, str(r["image_id"]))).convert("RGB")
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
        if self.augment and random.random() < 0.5:
            crop = transforms.functional.hflip(crop)
        crop = self.tx(crop)

        if "ic4" in r:
            y4 = int(r["ic4"])
        else:
            y4 = map_ic4(int(r["icdas"]))
        return crop, y4


class ResNet18Corn(nn.Module):
    def __init__(self, pretrained=True, num_classes=4):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.output_layer = nn.Linear(m.fc.in_features, num_classes - 1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.flatten(1)
        logits = self.output_layer(feat)
        return logits


def loss_corn(logits, y_train, num_classes=4):
    sets = []
    for i in range(num_classes - 1):
        label_mask = y_train > (i - 1)
        label_tensor = (y_train[label_mask] > i).to(torch.float32)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.0

    for task_index, (train_examples, train_labels) in enumerate(sets):
        if train_labels.numel() < 1:
            continue

        num_examples += train_labels.numel()
        pred = logits[train_examples, task_index]

        loss = -torch.sum(
            F.logsigmoid(pred) * train_labels
            + (F.logsigmoid(pred) - pred) * (1.0 - train_labels)
        )
        losses += loss

    return losses / max(num_examples, 1)


@torch.no_grad()
def corn_decode(logits, threshold=0.5):
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    predict_levels = probas > threshold
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels.long(), probas


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    y_true = []
    y_pred = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = loss_corn(logits, labels, num_classes=4)
        pred, _ = corn_decode(logits)

        losses.append(loss.item())
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())

    val_loss = float(np.mean(losses)) if losses else 0.0
    mae = mean_absolute_error(y_true, y_pred) if y_true else float("nan")
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic") if y_true else float("nan")
    return val_loss, mae, qwk


def main(args):
    seed_everything(args.seed, deterministic=args.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_ds = Icdas4RoiDataset(args.train_csv, args.img_root_train, args.img_size, args.expand, augment=True)
    va_ds = Icdas4RoiDataset(args.val_csv, args.img_root_val, args.img_size, args.expand, augment=False)

    g = torch.Generator()
    g.manual_seed(args.seed)
    tr_loader = DataLoader(
        tr_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = ResNet18Corn(pretrained=True, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_losses = []

        for imgs, labels in tr_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = loss_corn(logits, labels, num_classes=4)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss, val_mae, val_qwk = evaluate(model, va_loader, device)

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_MAE={val_mae:.4f} "
            f"val_QWK={val_qwk:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.out)

    print(f"Best val_loss={best_val:.4f}")
    print(f"Saved best checkpoint to: {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="icdas4_train.csv")
    ap.add_argument("--val_csv", type=str, default="icdas4_val.csv")
    ap.add_argument("--img_root_train", type=str, required=True)
    ap.add_argument("--img_root_val", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--expand", type=float, default=1.25)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, default="corn_head_icdas4.pt")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()
    main(args)
