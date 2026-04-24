import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from roi_ordinal_baselines_common import (
    Icdas4RoiDataset,
    ResNet18RoiClassifier,
    ensure_parent_dir,
    evaluate_softmax_classifier,
    load_checkpoint_payload,
)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = Icdas4RoiDataset(args.roi_csv, args.img_root, args.img_size, args.expand, augment=False)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    payload = load_checkpoint_payload(args.ckpt, device)
    model = ResNet18RoiClassifier(pretrained=False, num_classes=4).to(device)
    model.load_state_dict(payload["model_state"])

    metrics, pred_df = evaluate_softmax_classifier(model, loader, device)
    base_df = pd.read_csv(args.roi_csv).reset_index(drop=True)
    out_df = pd.concat([base_df, pred_df.drop(columns=["image_id"])], axis=1)

    print(f"AUC(>=1/>=3/>=5)= {metrics['auc_ge1']:.3f}/{metrics['auc_ge3']:.3f}/{metrics['auc_ge5']:.3f}")
    print(f"MAE={metrics['mae']:.3f}  QWK={metrics['qwk']:.3f}  ROI_ACC={metrics['roi_acc']:.3f}")
    print(
        f"Top-1@p_ge1: {metrics['top1_hits']}/{metrics['n_pos_images']} = {metrics['top1']:.3f}\n"
        f"Top-3@p_ge1: {metrics['top3_hits']}/{metrics['n_pos_images']} = {metrics['top3']:.3f}"
    )
    print("Confusion Matrix (rows=GT 0/A/B/C, cols=Pred):")
    print(metrics["confusion_matrix"])

    if args.out_csv:
        ensure_parent_dir(args.out_csv)
        out_df.to_csv(args.out_csv, index=False)
        print(f"Saved per-ROI preds to: {args.out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi_csv", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--expand", type=float, default=1.25)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out_csv", type=str, default="")
    args = parser.parse_args()
    main(args)
