import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# Import models
from train_softmax_ordplus_o2s_boundary import ResNet18SoftmaxOrdPlusBoundary, Icdas4RoiDataset, map_ic4
from train_softmax_ordplus_o2s_boundary_gpt import ResNet18SoftmaxOrdPlusBoundaryV2

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

def eval_model(model, loader, device, out_csv):
    model.eval()
    all_preds, all_gts, all_img_ids = [], [], []
    all_p_ge1, all_p_ge3, all_p_ge5 = [], [], []

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            out = model(imgs)
            p_final = out["p_final"]
            
            ge1 = 1.0 - p_final[:, 0]
            ge3 = p_final[:, 2] + p_final[:, 3]
            ge5 = p_final[:, 3]
            
            pred = torch.zeros_like(ge1, dtype=torch.long)
            pred = torch.where(ge1 >= 0.5, torch.ones_like(pred), pred)
            pred = torch.where(ge3 >= 0.5, torch.full_like(pred, 2), pred)
            pred = torch.where(ge5 >= 0.5, torch.full_like(pred, 3), pred)

            all_preds.append(pred.cpu().numpy())
            all_gts.append(labels.numpy())
            all_p_ge1.append(ge1.cpu().numpy())
            all_p_ge3.append(ge3.cpu().numpy())
            all_p_ge5.append(ge5.cpu().numpy())
            
            batch_indices = range(i * loader.batch_size, min((i + 1) * loader.batch_size, len(loader.dataset)))
            all_img_ids.extend([loader.dataset.df.iloc[idx]["image_id"] for idx in batch_indices])

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_gts)
    p_ge1 = np.concatenate(all_p_ge1)
    p_ge3 = np.concatenate(all_p_ge3)
    p_ge5 = np.concatenate(all_p_ge5)

    mae = np.mean(np.abs(y_true - y_pred))
    qwk = qwk_numpy(y_true, y_pred)
    
    def safe_auc(y_true_bin, y_prob):
        if len(np.unique(y_true_bin)) < 2: return 0.5
        return roc_auc_score(y_true_bin, y_prob)

    auc1 = safe_auc(y_true >= 1, p_ge1)
    auc3 = safe_auc(y_true >= 2, p_ge3)
    auc5 = safe_auc(y_true >= 3, p_ge5)

    results_df = pd.DataFrame({
        "image_id": all_img_ids,
        "ic4": y_true,
        "pred_ic4": y_pred,
        "p_ge1": p_ge1,
        "p_ge3": p_ge3,
        "p_ge5": p_ge5
    })
    results_df.to_csv(out_csv, index=False)
    
    return mae, qwk, auc1, auc3, auc5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, default=None)
    ap.add_argument("--img_root_val", type=str, required=True)
    ap.add_argument("--img_root_test", type=str, default=None)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--model_type", type=str, choices=["standard", "gpt"], required=True)
    ap.add_argument("--out_val_csv", type=str, required=True)
    ap.add_argument("--out_test_csv", type=str, required=True)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(a.ckpt, map_location="cpu")
    
    if a.model_type == "standard":
        model = ResNet18SoftmaxOrdPlusBoundary(pretrained=False)
    else:
        model = ResNet18SoftmaxOrdPlusBoundaryV2(pretrained=False)
        
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    val_ds = Icdas4RoiDataset(a.val_csv, a.img_root_val, augment=False)
    val_loader = DataLoader(val_ds, batch_size=a.bs, shuffle=False, num_workers=4)
    
    print(f"\n--- Evaluating Val: {a.val_csv} ---")
    v_mae, v_qwk, v_auc1, v_auc3, v_auc5 = eval_model(model, val_loader, device, a.out_val_csv)
    print(f"Val MAE: {v_mae:.4f}, QWK: {v_qwk:.4f}")
    print(f"Val AUC: >=1: {v_auc1:.4f}, >=3: {v_auc3:.4f}, >=5: {v_auc5:.4f}")

    if a.test_csv and a.img_root_test:
        test_ds = Icdas4RoiDataset(a.test_csv, a.img_root_test, augment=False)
        test_loader = DataLoader(test_ds, batch_size=a.bs, shuffle=False, num_workers=4)
        print(f"\n--- Evaluating Test: {a.test_csv} ---")
        t_mae, t_qwk, t_auc1, t_auc3, t_auc5 = eval_model(model, test_loader, device, a.out_test_csv)
        print(f"Test MAE: {t_mae:.4f}, QWK: {t_qwk:.4f}")
        print(f"Test AUC: >=1: {t_auc1:.4f}, >=3: {t_auc3:.4f}, >=5: {t_auc5:.4f}")

if __name__ == "__main__":
    main()
