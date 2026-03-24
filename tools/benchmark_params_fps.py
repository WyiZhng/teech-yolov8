import csv
import glob
import os
import time

import torch

# Compat shim: some legacy checkpoints reference ultralytics.utils.loss.DFLoss.
# It is not needed during pure inference, but must exist for unpickling.
import ultralytics.utils.loss as _uloss


if not hasattr(_uloss, "DFLoss"):
    class DFLoss(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            return torch.tensor(0.0)


    _uloss.DFLoss = DFLoss

from ultralytics import YOLO


def find_run_dir(base_dir: str) -> str:
    cands = sorted(glob.glob(os.path.join(base_dir, "*epcoh")))
    if not cands:
        cands = sorted(glob.glob(os.path.join(base_dir, "*")))
    if not cands:
        raise FileNotFoundError(f"No run directory found under {base_dir}")
    return cands[0]


def benchmark_weight(weight_path: str, imgsz: int, bs: int, warmup: int, iters: int, device: str):
    y = YOLO(weight_path)
    model = y.model.to(device).eval()

    params = sum(p.numel() for p in model.parameters())
    x = torch.randn(bs, 3, imgsz, imgsz, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    sec_per_img = dt / (iters * bs)
    return {
        "weight": os.path.basename(weight_path),
        "params": int(params),
        "params_M": round(params / 1e6, 6),
        "imgsz": imgsz,
        "batch": bs,
        "device": device,
        "forward_ms_per_img": round(sec_per_img * 1000.0, 6),
        "fps": round(1.0 / sec_per_img, 6),
    }


def main():
    base_dir = "/data/HZNU_ZWY/zwy_project/ultralytics-8.1.35/runs/detect/public-2class"
    run_dir = find_run_dir(base_dir)
    weights_dir = os.path.join(run_dir, "weights")
    out_csv = os.path.join(run_dir, "params_fps.csv")

    weights = sorted(glob.glob(os.path.join(weights_dir, "*.pt")))
    if not weights:
        raise FileNotFoundError(f"No .pt files found in {weights_dir}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    imgsz = 640
    bs = 1
    warmup = 30
    iters = 100

    print("run_dir=", run_dir)
    print("device=", device)

    rows = []
    for w in weights:
        row = benchmark_weight(w, imgsz=imgsz, bs=bs, warmup=warmup, iters=iters, device=device)
        rows.append(row)
        print(f"{row['weight']}: params={row['params_M']}M, ms={row['forward_ms_per_img']}, fps={row['fps']}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("saved=", out_csv)


if __name__ == "__main__":
    main()
