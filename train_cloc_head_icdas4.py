import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from roi_ordinal_baselines_common import (
    Icdas4RoiDataset,
    ResNet18RoiClassifier,
    ensure_parent_dir,
    export_cloc_fixed_map,
    load_cloc_loss_module,
    seed_everything,
    seed_worker,
)


class _NoOpWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_custom_scalars(self, *args, **kwargs):
        return None


@torch.no_grad()
def evaluate_cloc(model, loader, device, ce_criterion, cloc_criterion, cloc_weight):
    model.eval()
    losses = []
    all_probs = []
    all_y = []
    all_image_ids = []

    for imgs, labels, image_ids in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        ce_loss = ce_criterion(logits, labels)
        cloc_loss = cloc_criterion(logits, labels)
        loss = ce_loss + cloc_weight * cloc_loss

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


def build_cloc_criterion(module, args, device, learnable_map=None):
    writer = _NoOpWriter()
    if args.margin_mode == "sm":
        return module.OrdinalContrastiveLoss_sm(
            n_classes=4,
            device=device,
            learnable_map=learnable_map,
            summaryWriter=writer,
        )
    return module.OrdinalContrastiveLoss_mm(
        n_classes=4,
        device=device,
        learnable_map=learnable_map,
        summaryWriter=writer,
    )


def main(args):
    seed_everything(args.seed, deterministic=args.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cloc_module = load_cloc_loss_module(".")

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
    ce_criterion = nn.CrossEntropyLoss()

    phase1_epochs = args.phase1_epochs if args.phase1_epochs >= 0 else args.epochs // 2
    phase2_epochs = args.phase2_epochs if args.phase2_epochs >= 0 else args.epochs - phase1_epochs
    if phase1_epochs + phase2_epochs <= 0:
        raise ValueError("Total training epochs must be positive.")

    phase1_criterion = build_cloc_criterion(cloc_module, args, device, learnable_map=None)
    phase1_params = list(model.parameters())
    if list(phase1_criterion.parameters()):
        phase1_params += list(phase1_criterion.parameters())
    optimizer = torch.optim.Adam(phase1_params, lr=args.lr)

    best_val_loss = float("inf")
    best_fixed_map = None
    ensure_parent_dir(args.out)
    global_step = 0

    total_epochs = phase1_epochs + phase2_epochs
    for epoch in range(total_epochs):
        in_phase1 = epoch < phase1_epochs
        if epoch == phase1_epochs and phase2_epochs > 0:
            best_fixed_map = export_cloc_fixed_map(phase1_criterion)
            phase2_criterion = build_cloc_criterion(cloc_module, args, device, learnable_map=best_fixed_map)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        current_criterion = phase1_criterion if in_phase1 else phase2_criterion
        phase_name = "phase1" if in_phase1 else "phase2"

        model.train()
        running_total = 0.0
        running_ce = 0.0
        running_cloc = 0.0
        n_seen = 0

        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            ce_loss = ce_criterion(logits, labels)
            cloc_loss = current_criterion(logits, labels, global_step)
            loss = ce_loss + args.cloc_weight * cloc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = imgs.size(0)
            n_seen += batch_size
            running_total += float(loss.item()) * batch_size
            running_ce += float(ce_loss.item()) * batch_size
            running_cloc += float(cloc_loss.item()) * batch_size
            global_step += 1

        if not in_phase1 and best_fixed_map is None:
            best_fixed_map = export_cloc_fixed_map(phase1_criterion)

        train_total = running_total / max(n_seen, 1)
        train_ce = running_ce / max(n_seen, 1)
        train_cloc = running_cloc / max(n_seen, 1)
        val_metrics = evaluate_cloc(model, val_loader, device, ce_criterion, current_criterion, args.cloc_weight)
        fixed_map_to_save = export_cloc_fixed_map(current_criterion if not in_phase1 else phase1_criterion)

        print(
            f"Epoch {epoch + 1}/{total_epochs} "
            f"{phase_name} "
            f"train_loss={train_total:.4f} "
            f"train_ce={train_ce:.4f} "
            f"train_cloc={train_cloc:.4f} "
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
                    "method": "cloc",
                    "margin_mode": args.margin_mode,
                    "cloc_weight": args.cloc_weight,
                    "phase1_epochs": phase1_epochs,
                    "phase2_epochs": phase2_epochs,
                    "fixed_margin_map": fixed_map_to_save,
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
    parser.add_argument("--phase1_epochs", type=int, default=-1)
    parser.add_argument("--phase2_epochs", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--cloc_weight", type=float, default=1.0)
    parser.add_argument("--margin_mode", type=str, default="mm", choices=["mm", "sm"])
    parser.add_argument("--out", type=str, default="cloc_head_icdas4.pt")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    main(args)
