#!/usr/bin/env python3
"""Run low-threshold YOLOv8 inference on selected dataset splits and export predictions.

Usage (example):
    python tools/export_low_thresh_predictions.py \
        --model runs/detect/public-2class/v8s_bin_1280_ab2_tail/weights/best.pt \
        --data datasets/mini-dental-binary.yaml \
        --splits train val \
        --imgsz 1280 --conf 0.2 --iou 0.6 --max-det 300 \
        --output runs/detect/public-2class/v8s_bin_1280_ab2_tail/low_thresh_predictions
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export low-threshold YOLOv8 predictions for recall analysis.")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained YOLOv8 weights (e.g., best.pt).")
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML describing train/val splits.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Dataset splits to evaluate.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold (low for recall).")
    parser.add_argument("--iou", type=float, default=0.6, help="IOU threshold for NMS.")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per image.")
    parser.add_argument("--batch", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--device", default=None, help="Computation device, e.g. 0 or 'cpu'.")
    parser.add_argument("--output", type=Path, required=True, help="Directory to store exported CSV files.")
    parser.add_argument("--augment", action="store_true", help="Enable inference-time augmentation.")
    return parser.parse_args()


def load_dataset_paths(data_cfg: Path) -> Tuple[Dict[str, Iterable[Path]], Path]:
    with data_cfg.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    base_dir = data_cfg.parent
    dataset_root = (base_dir / data.get("path", "")).resolve()

    split_map: Dict[str, Iterable[Path]] = {}
    if "train" in data:
        split_map["train"] = [dataset_root / Path(data["train"])]

    val_key = None
    for candidate in ("val", "valid", "validation"):
        if candidate in data:
            val_key = candidate
            break
    if val_key:
        split_map["val"] = [dataset_root / Path(data[val_key])]

    test_key = None
    for candidate in ("test", "testing"):
        if candidate in data:
            test_key = candidate
            break
    if test_key:
        split_map["test"] = [dataset_root / Path(data[test_key])]

    return split_map, dataset_root


def get_image_id(image_path: Path, sources: List[Path], dataset_root: Path) -> str:
    image_path = image_path.resolve()
    for base in sources:
        try:
            return image_path.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    try:
        return image_path.relative_to(dataset_root).as_posix()
    except ValueError:
        return image_path.name


def export_split(
    model: YOLO,
    split_name: str,
    sources: Iterable[Path],
    dataset_root: Path,
    out_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    batch: int,
    device,
    augment: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{split_name}_predictions.csv"
    rows_written = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_id", "x", "y", "w", "h", "yolo_score"])

        for source in sources:
            source = source.resolve()
            if not source.exists():
                raise FileNotFoundError(f"Source path not found for split '{split_name}': {source}")

            predictor = model.predict(
                source=str(source),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                batch=batch,
                device=device,
                augment=augment,
                stream=True,
                verbose=False,
            )

            for result in predictor:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                image_id = get_image_id(Path(result.path), [source], dataset_root)
                xyxy = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

                for (x1, y1, x2, y2), score in zip(xyxy, scores):
                    w = x2 - x1
                    h = y2 - y1
                    writer.writerow([
                        image_id,
                        f"{float(x1):.4f}",
                        f"{float(y1):.4f}",
                        f"{float(w):.4f}",
                        f"{float(h):.4f}",
                        f"{float(score):.6f}",
                    ])
                    rows_written += 1

    print(f"[{split_name}] Wrote {rows_written} predictions -> {csv_path}")
    return csv_path


def main() -> None:
    args = parse_args()

    split_map, dataset_root = load_dataset_paths(args.data)
    requested_splits = []
    for split in args.splits:
        key = split.lower()
        if key not in split_map:
            available = ", ".join(split_map.keys()) or "none"
            raise ValueError(f"Split {split!r} not found in dataset YAML. Available: {available}.")
        requested_splits.append(key)

    print(f"Loading model from {args.model}")
    model = YOLO(str(args.model))

    for split in requested_splits:
        export_split(
            model=model,
            split_name=split,
            sources=list(split_map[split]),
            dataset_root=dataset_root,
            out_dir=args.output,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            batch=args.batch,
            device=args.device,
            augment=args.augment,
        )


if __name__ == "__main__":
    main()
