import argparse
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--x_col", default="x")
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--w_col", default="w")
    ap.add_argument("--h_col", default="h")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    need = ["image_id", args.x_col, args.y_col, args.w_col, args.h_col]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {args.in_csv}. Got: {df.columns.tolist()}")

    # 读图拿宽高（缓存）
    try:
        from PIL import Image
    except Exception:
        raise RuntimeError("PIL(Pillow) not available. Please install pillow or rewrite using cv2.")

    size_cache = {}

    def get_wh(img_name: str):
        if img_name in size_cache:
            return size_cache[img_name]
        p = os.path.join(args.img_root, img_name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        with Image.open(p) as im:
            W, H = im.size
        size_cache[img_name] = (W, H)
        return W, H

    x = df[args.x_col].astype(float).values
    y = df[args.y_col].astype(float).values
    w = df[args.w_col].astype(float).values
    h = df[args.h_col].astype(float).values

    x_c_list, y_c_list, w_n_list, h_n_list = [], [], [], []

    for img, xi, yi, wi, hi in zip(df["image_id"].values, x, y, w, h):
        W, H = get_wh(img)
        x_c = (xi + wi / 2.0) / W
        y_c = (yi + hi / 2.0) / H
        w_n = wi / W
        h_n = hi / H
        x_c_list.append(x_c)
        y_c_list.append(y_c)
        w_n_list.append(w_n)
        h_n_list.append(h_n)

    df["x_c"] = x_c_list
    df["y_c"] = y_c_list
    df["w_n"] = w_n_list
    df["h_n"] = h_n_list

    # 保留原列 + 新列都行；eval脚本只要能找到 x_c,y_c,w_n,h_n 和 cls_col
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv} rows={len(df)}")

if __name__ == "__main__":
    main()
