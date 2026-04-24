import importlib.util
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
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
        row = self.df.iloc[i]
        image_id = str(row["image_id"])
        img = Image.open(os.path.join(self.root, image_id)).convert("RGB")
        w_img, h_img = img.size

        if "gx" in row:
            x, y, w, h = float(row["gx"]), float(row["gy"]), float(row["gw"]), float(row["gh"])
        else:
            x, y, w, h = float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])

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

        if "ic4" in row:
            label = int(row["ic4"])
        else:
            label = map_ic4(int(row["icdas"]))
        return crop, label, image_id


class ResNet18RoiClassifier(nn.Module):
    def __init__(self, pretrained=True, num_classes=4):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x, return_features=False):
        features = self.backbone(x).flatten(1)
        logits = self.fc(features)
        if return_features:
            return features, logits
        return logits


def pairwise_order_regularization(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {tuple(features.shape)}")
    if features.shape[0] < 2:
        return features.new_zeros(())

    features = F.normalize(features, dim=1)
    labels = labels.reshape(-1, 1).float().to(features.device)

    feature_distance = torch.cdist(features.float(), features.float(), p=1.0)
    label_distance = torch.cdist(labels, labels, p=1.0)

    upper = torch.triu(torch.ones_like(feature_distance, dtype=torch.bool), diagonal=1)
    feature_distance = feature_distance[upper]
    label_distance = label_distance[upper]

    if label_distance.numel() == 0:
        return features.new_zeros(())

    weights_min = label_distance.min()
    weights_max = label_distance.max()
    if weights_max.item() == weights_min.item():
        if weights_max.item() == 0:
            weights = torch.zeros_like(label_distance)
        else:
            weights = label_distance / weights_max
    else:
        weights = (label_distance - weights_min) / weights_max

    return -(feature_distance * weights).mean()


def safe_auc(y_true, score) -> float:
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def compute_softmax_metrics(
    base_df: pd.DataFrame, probs: np.ndarray, y_true: np.ndarray
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = probs.argmax(axis=1).astype(int)

    p0 = probs[:, 0]
    p_a = probs[:, 1]
    p_b = probs[:, 2]
    p_c = probs[:, 3]
    p_ge1 = p_a + p_b + p_c
    p_ge3 = p_b + p_c
    p_ge5 = p_c

    out_df = base_df.reset_index(drop=True).copy()
    out_df["y_gt"] = y_true
    out_df["y_pred"] = y_pred
    out_df["p0"] = p0
    out_df["pA"] = p_a
    out_df["pB"] = p_b
    out_df["pC"] = p_c
    out_df["p_ge1"] = p_ge1
    out_df["p_ge3"] = p_ge3
    out_df["p_ge5"] = p_ge5

    pos_hits_top1 = 0
    pos_hits_top3 = 0
    n_pos_images = 0
    for _, group in out_df.groupby("image_id"):
        if not (group["y_gt"] >= 1).any():
            continue
        n_pos_images += 1
        sorted_group = group.sort_values("p_ge1", ascending=False)
        pos_hits_top1 += int((sorted_group.head(1)["y_gt"] >= 1).any())
        pos_hits_top3 += int((sorted_group.head(3)["y_gt"] >= 1).any())

    metrics = {
        "auc_ge1": safe_auc(y_true >= 1, p_ge1),
        "auc_ge3": safe_auc(y_true >= 2, p_ge3),
        "auc_ge5": safe_auc(y_true >= 3, p_ge5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "roi_acc": float((y_pred == y_true).mean()),
        "top1": float(pos_hits_top1 / max(n_pos_images, 1)),
        "top3": float(pos_hits_top3 / max(n_pos_images, 1)),
        "top1_hits": int(pos_hits_top1),
        "top3_hits": int(pos_hits_top3),
        "n_pos_images": int(n_pos_images),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
    }
    return metrics, out_df


@torch.no_grad()
def evaluate_softmax_classifier(model, loader, device, criterion=None):
    model.eval()
    losses = []
    all_probs = []
    all_y = []
    all_image_ids: List[str] = []

    for imgs, labels, image_ids in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        if criterion is not None:
            losses.append(float(criterion(logits, labels).item()))
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_y.append(labels.cpu().numpy())
        all_image_ids.extend(list(image_ids))

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    df = pd.DataFrame({"image_id": all_image_ids})
    metrics, pred_df = compute_softmax_metrics(df, probs, y_true)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics, pred_df


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_checkpoint_payload(ckpt_path: str, device: str):
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "model_state" in payload:
        return payload
    return {"model_state": payload}


def load_cloc_loss_module(repo_root: str):
    loss_path = os.path.join(repo_root, "model-test", "CLOC-main", "loss.py")
    spec = importlib.util.spec_from_file_location("cloc_official_loss", loss_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load CLOC loss from {loss_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_cloc_fixed_map(cloc_loss) -> List[List[float]]:
    with torch.no_grad():
        if hasattr(cloc_loss, "distances_ori"):
            distances = cloc_loss.distances_ori.detach().clone()
            if hasattr(cloc_loss, "mask_learnables"):
                distances[cloc_loss.mask_learnables] = cloc_loss.learnables.detach()
            margins = F.softplus(distances.float()).cpu().tolist()
            return [["fixed", float(v)] for v in margins]
        if hasattr(cloc_loss, "distance"):
            margin = float(F.softplus(cloc_loss.distance.detach().float()).cpu().item())
            return [["fixed", margin]]
    raise ValueError("Unsupported CLOC loss object; could not export learned margins.")
