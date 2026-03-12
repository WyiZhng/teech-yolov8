import os
import argparse
import pandas as pd

def gather_preds(labels_dir, out_csv):
    rows = []
    if not os.path.exists(labels_dir):
        print(f"Labels directory not found: {labels_dir}")
        return

    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue
        image_id = fname.replace(".txt", ".jpg")
        path = os.path.join(labels_dir, fname)
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                cls, x_c, y_c, w_n, h_n, conf = parts
                rows.append({
                    "image_id": image_id,
                    "cls_pred": int(cls),
                    "x_c": float(x_c),
                    "y_c": float(y_c),
                    "w_n": float(w_n),
                    "h_n": float(h_n),
                    "score": float(conf),  # YOLO 输出的 conf（objectness*class_prob）
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv, "rows:", len(df))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", type=str,
                    default="runs/predict/y8_icdas4_val_pred/labels")
    ap.add_argument("--out_csv", type=str,
                    default="pred_val_icdas4_yolo4cls.csv")
    args = ap.parse_args()

    gather_preds(args.labels_dir, args.out_csv)
