import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from roi_ordinal_baselines_common import (
    Icdas4RoiDataset,
    ResNet18RoiClassifier,
    ensure_parent_dir,
    pairwise_order_regularization,
    seed_everything,
    seed_worker,
)


@torch.no_grad()
def evaluate_order(model, loader, device, ce_criterion, order_weight):
    model.eval()
    losses = []
    all_probs = []
    all_y = []
    all_image_ids = []

    for imgs, labels, image_ids in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        features, logits = model(imgs, return_features=True)
        ce_loss = ce_criterion(logits, labels)
        order_loss = pairwise_order_regularization(features, labels)
        loss = ce_loss + order_weight * order_loss

        losses.append(float(loss.item()))
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_y.append(labels.cpu().numpy())
        all_image_ids.extend(list(image_ids))

    import numpy as np
    import pandas as pd

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    base_df = pd.DataFrame({"image_id": all_image_ids})
    from roi_ordinal_baselines_common import compute_softmax_metrics

    metrics, _ = compute_softmax_metrics(base_df, probs, y_true)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def main(args):
    seed_everything(args.seed, deterministic=args.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = Icdas4RoiDataset(args.train_csv, args.img_root_train, args.img_size, args.expand, augment=True)
    val_ds = Icdas4RoiDataset(args.val_csv, args.img_root_val, args.img_size, args.expand, augment=False)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = ResNet18RoiClassifier(pretrained=True, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    ensure_parent_dir(args.out)

    for epoch in range(args.epochs):
        model.train()
        running_total = 0.0
        running_ce = 0.0
        running_order = 0.0
        n_seen = 0

        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            features, logits = model(imgs, return_features=True)
            ce_loss = ce_criterion(logits, labels)
            order_loss = pairwise_order_regularization(features, labels)
            loss = ce_loss + args.order_weight * order_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = imgs.size(0)
            n_seen += batch_size
            running_total += float(loss.item()) * batch_size
            running_ce += float(ce_loss.item()) * batch_size
            running_order += float(order_loss.item()) * batch_size

        train_total = running_total / max(n_seen, 1)
        train_ce = running_ce / max(n_seen, 1)
        train_order = running_order / max(n_seen, 1)
        val_metrics = evaluate_order(model, val_loader, device, ce_criterion, args.order_weight)

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_total:.4f} "
            f"train_ce={train_ce:.4f} "
            f"train_order={train_order:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_MAE={val_metrics['mae']:.4f} "
            f"val_QWK={val_metrics['qwk']:.4f} "
            f"val_ACC={val_metrics['roi_acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "method": "order",
                    "order_weight": args.order_weight,
                    "args": vars(args),
                    "best_val_loss": best_val_loss,
                },
                args.out,
            )

    print(f"Best val_loss={best_val_loss:.4f}")
    print(f"Saved best checkpoint to: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="icdas4_train.csv")
    parser.add_argument("--val_csv", type=str, default="icdas4_val.csv")
    parser.add_argument("--img_root_train", type=str, required=True)
    parser.add_argument("--img_root_val", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--expand", type=float, default=1.25)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--order_weight", type=float, default=0.1)
    parser.add_argument("--out", type=str, default="order_head_icdas4.pt")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    main(args)
