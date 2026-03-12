import os
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image

def make_labels_for_split(csv_path, images_dir, labels_dir,
                          img_ext=".jpg", col_name_cls="icdas4"):
    os.makedirs(labels_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # 按 image_id 分组
    for img_id, g in tqdm(df.groupby("image_id"), desc=f"Processing {os.path.basename(csv_path)}"):
        img_path = os.path.join(images_dir, img_id)
        if not os.path.exists(img_path):
            # 尝试找一下不带路径的
            img_path = os.path.join(images_dir, os.path.basename(img_id))
            if not os.path.exists(img_path):
                print(f"[WARN] image not found: {img_path}")
                continue

        # 读取图像尺寸
        try:
            with Image.open(img_path) as img:
                W, H = img.size
        except Exception as e:
            print(f"[ERR] Cannot open {img_path}: {e}")
            continue

        lines = []
        for _, r in g.iterrows():
            cls = int(r[col_name_cls])          # 0,1,2,3
            gx, gy, gw, gh = float(r["gx"]), float(r["gy"]), float(r["gw"]), float(r["gh"])

            # 检查坐标系：
            # 如果 gy=20, gh=300，显然 gy 是 top-left，否则 center=20 意味着 top=-130
            # 因此这里统一按 Top-Left 处理，转为 Center
            x_c = (gx + gw / 2) / W
            y_c = (gy + gh / 2) / H
            w_n = gw / W
            h_n = gh / H

            # 限制在 [0, 1]
            x_c = max(0, min(1, x_c))
            y_c = max(0, min(1, y_c))
            w_n = max(0, min(1, w_n))
            h_n = max(0, min(1, h_n))

            # YOLO txt 一行: cls cx cy w h
            lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        # 写 label 文件
        label_name = os.path.splitext(os.path.basename(img_id))[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        with open(label_path, "w") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="icdas4_train.csv")
    parser.add_argument("--val_csv",   type=str, default="icdas4_val.csv")
    parser.add_argument("--test_csv",  type=str, default="icdas4_test.csv")
    parser.add_argument("--img_root",  type=str, default="datasets/icdas_yolo")
    parser.add_argument("--out_root",  type=str, default="datasets/icdas_yolo_icdas4")
    parser.add_argument("--cls_col",   type=str, default="icdas4")
    args = parser.parse_args()

    # 建立目录结构并处理图片链接
    for split in ["train", "val", "test"]:
        # 目标目录
        split_dir = os.path.join(args.out_root, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Labels 目录
        labels_dir = os.path.join(split_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)

        # Images 目录 (使用软链接)
        src_img_dir = os.path.abspath(os.path.join(args.img_root, split, "images"))
        dst_img_dir = os.path.join(split_dir, "images")
        
        if not os.path.exists(dst_img_dir):
            if os.path.exists(src_img_dir):
                print(f"Symlinking {src_img_dir} -> {dst_img_dir}")
                os.symlink(src_img_dir, dst_img_dir)
            else:
                print(f"[WARN] Source image dir not found: {src_img_dir}")
                os.makedirs(dst_img_dir, exist_ok=True) # Fallback
        else:
            print(f"[INFO] Images dir already exists: {dst_img_dir}")

    # 生成 labels
    print("Generating Train Labels...")
    make_labels_for_split(
        args.train_csv,
        os.path.join(args.out_root, "train", "images"),
        os.path.join(args.out_root, "train", "labels"),
        col_name_cls=args.cls_col
    )

    print("Generating Val Labels...")
    make_labels_for_split(
        args.val_csv,
        os.path.join(args.out_root, "val", "images"),
        os.path.join(args.out_root, "val", "labels"),
        col_name_cls=args.cls_col
    )

    print("Generating Test Labels...")
    make_labels_for_split(
        args.test_csv,
        os.path.join(args.out_root, "test", "images"),
        os.path.join(args.out_root, "test", "labels"),
        col_name_cls=args.cls_col
    )
