#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_annotations.py

可视化数据集中的标注（支持 YOLO .txt 与 COCO .json）
- 在图片上绘制方框与类别名
- 生成一个 index.html 方便逐张预览
- 导出 annotations.csv（每个框一行）与 class_summary.csv（类别统计）

用法示例：
1) YOLO(txt) 标注：
   python visualize_annotations.py \
     --images /path/to/images \
     --labels /path/to/labels \            # 可省略，若与 images 同目录且同名 .txt
     --classes /path/to/classes.txt \      # 可选：每行一个类别名
     --out out_yolo

2) COCO(json) 标注：
   python visualize_annotations.py \
     --images /path/to/images \
     --coco /path/to/instances.json \
     --out out_coco

可选参数：
  --limit 100          # 仅处理前 100 张
  --line-thickness 2   # 框线粗细
  --font-size 16       # 文本字号
  --score-thresh 0.0   # 仅绘制分数>=阈值（COCO 中常见）

作者：你友好的小工具 🧰
"""

import os
import json
import csv
import argparse
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTS

def load_class_names(classes_path: Optional[str], fallback_num: int = 0) -> Dict[int, str]:
    """
    读取类别名文件（每行一个）。若无则返回空映射（显示id）。
    """
    if not classes_path:
        return {}
    names = {}
    with open(classes_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            name = line.strip()
            if name:
                names[i] = name
    return names

def color_for_label(label: str) -> Tuple[int, int, int]:
    """
    基于标签名得到稳定颜色（哈希到RGB）。
    """
    h = abs(hash(label))  # 稳定到会话内
    r = 50 + (h % 206)
    g = 50 + ((h // 7) % 206)
    b = 50 + ((h // 13) % 206)
    return (r, g, b)

from PIL import ImageDraw, ImageFont

def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """兼容 Pillow 版本的文本尺寸测量：优先 textbbox，其次 font.getbbox，最后 textsize。"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)  # Pillow >= 8.0
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        try:
            bbox = font.getbbox(text)  # Pillow >= 8.0
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            return draw.textsize(text, font=font)  # 旧版 Pillow 兜底

def draw_box(draw: ImageDraw.ImageDraw, bbox, label: str, line_thickness: int, font: ImageFont.ImageFont):
    x1, y1, x2, y2 = bbox
    color = color_for_label(label)
    for t in range(line_thickness):
        draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color, width=1)
    tw, th = measure_text(draw, label, font)
    pad = 2
    box = [x1, max(0, y1 - th - 2*pad), x1 + tw + 2*pad, y1]
    draw.rectangle(box, fill=color)
    draw.text((x1 + pad, box[1] + pad), label, fill=(255, 255, 255), font=font)

def yolo_line_to_bbox(line: str, W: int, H: int) -> Tuple[int, Tuple[int,int,int,int], Optional[float]]:
    """
    支持 YOLO 格式：class cx cy w h [可选: score]
    所有坐标/尺寸为相对值(0~1) -> 转换到像素
    返回: class_id, (xmin,ymin,xmax,ymax), score
    """
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"YOLO标注非法: {line}")
    cls = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:5])
    # 将相对坐标转换到像素
    bx = cx * W
    by = cy * H
    bw = w * W
    bh = h * H
    x1 = max(0, int(round(bx - bw / 2)))
    y1 = max(0, int(round(by - bh / 2)))
    x2 = min(W - 1, int(round(bx + bw / 2)))
    y2 = min(H - 1, int(round(by + bh / 2)))
    score = float(parts[5]) if len(parts) >= 6 else None
    return cls, (x1, y1, x2, y2), score

def try_parse_possible_voc(line: str) -> Optional[Tuple[int, Tuple[int,int,int,int], Optional[float]]]:
    """
    某些数据集的 .txt 不是YOLO，而是: class x1 y1 x2 y2 [score]
    我们尝试兼容；若不符合，则返回 None。
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    # 判断像素坐标是否看起来像整数/像素
    try:
        cls = int(float(parts[0]))
        x1, y1, x2, y2 = map(float, parts[1:5])
        if max(x1, y1, x2, y2) <= 1.0:
            # 更像 YOLO
            return None
        return cls, (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))), float(parts[5]) if len(parts) >= 6 else None
    except Exception:
        return None

def find_label_path_for_image(image_path: str, labels_dir: Optional[str]) -> Optional[str]:
    base = os.path.splitext(os.path.basename(image_path))[0]
    if labels_dir:
        p = os.path.join(labels_dir, base + ".txt")
        return p if os.path.exists(p) else None
    # 尝试同目录
    p = os.path.join(os.path.dirname(image_path), base + ".txt")
    return p if os.path.exists(p) else None

def load_font(font_size: int) -> ImageFont.ImageFont:
    # 尝试加载常见字体，失败则用默认字体
    for name in ["arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, font_size)
        except Exception:
            continue
    return ImageFont.load_default()

def process_yolo(images_dir: str, labels_dir: Optional[str], classes_path: Optional[str],
                 out_dir: str, limit: Optional[int], line_thickness: int,
                 font_size: int, score_thresh: float):
    class_names = load_class_names(classes_path)
    os.makedirs(out_dir, exist_ok=True)
    out_img_dir = os.path.join(out_dir, "annotated")
    os.makedirs(out_img_dir, exist_ok=True)

    font = load_font(font_size)

    # gather images
    images = []
    for root, _, files in os.walk(images_dir):
        for fn in files:
            if is_image_file(fn):
                images.append(os.path.join(root, fn))
    images.sort()
    if limit is not None:
        images = images[:limit]

    ann_csv_path = os.path.join(out_dir, "annotations.csv")
    cls_csv_path = os.path.join(out_dir, "class_summary.csv")
    html_path = os.path.join(out_dir, "index.html")

    class_counter = Counter()
    rows = []  # for annotations.csv

    for idx, img_path in enumerate(images, 1):
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            draw = ImageDraw.Draw(im)

            label_path = find_label_path_for_image(img_path, labels_dir)
            boxes_this_img = 0
            if label_path and os.path.exists(label_path):
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parsed_voc = try_parse_possible_voc(line)
                        if parsed_voc is not None:
                            cls, bbox, score = parsed_voc
                        else:
                            cls, bbox, score = yolo_line_to_bbox(line, W, H)
                        if (score is not None) and (score < score_thresh):
                            continue
                        name = class_names.get(cls, str(cls))
                        draw_box(draw, bbox, name, line_thickness, font)
                        x1,y1,x2,y2 = bbox
                        rows.append([os.path.basename(img_path), cls, name, x1,y1,x2,y2, score if score is not None else ""])
                        class_counter[name] += 1
                        boxes_this_img += 1
            # save annotated image
            out_name = os.path.splitext(os.path.basename(img_path))[0] + "_ann.jpg"
            save_path = os.path.join(out_img_dir, out_name)
            im.save(save_path, quality=95)

    # write CSVs
    with open(ann_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image","class_id","class_name","xmin","ymin","xmax","ymax","score"])
        w.writerows(rows)

    with open(cls_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_name","count"])
        for name, cnt in class_counter.most_common():
            w.writerow([name, cnt])

    # html index
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>YOLO 标注预览</title></head><body>\n")
        f.write("<h1>YOLO 标注预览</h1>\n")
        f.write("<p>图片总数: {}；共 {} 个框。</p>\n".format(len(images), sum(class_counter.values())))
        f.write("<ul>\n")
        for name, cnt in class_counter.most_common():
            f.write(f"<li>{name}: {cnt}</li>\n")
        f.write("</ul>\n")
        f.write("<hr/>\n")
        for root, _, files in os.walk(out_img_dir):
            for fn in sorted(files):
                fpath = os.path.join("annotated", fn)
                f.write(f"<div><img src='{fpath}' style='max-width:100%;'/><p>{fn}</p></div><hr/>\n")
        f.write("</body></html>\n")

    print(f"[完成] 输出目录: {out_dir}")
    print(f"  - 标注图: {out_img_dir}")
    print(f"  - CSV: {ann_csv_path}, {cls_csv_path}")
    print(f"  - HTML: {html_path}")

def process_coco(images_dir: str, coco_json_path: str, out_dir: str, limit: Optional[int],
                 line_thickness: int, font_size: int, score_thresh: float):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 建索引
    id_to_image = {img["id"]: img for img in coco.get("images", [])}
    id_to_cat = {cat["id"]: cat for cat in coco.get("categories", [])}
    img_to_anns = defaultdict(list)
    for ann in coco.get("annotations", []):
        img_to_anns[ann["image_id"]].append(ann)

    # 准备输出
    os.makedirs(out_dir, exist_ok=True)
    out_img_dir = os.path.join(out_dir, "annotated")
    os.makedirs(out_img_dir, exist_ok=True)
    font = load_font(font_size)

    image_items = list(id_to_image.items())
    image_items.sort(key=lambda kv: kv[0])
    if limit is not None:
        image_items = image_items[:limit]

    ann_csv_path = os.path.join(out_dir, "annotations.csv")
    cls_csv_path = os.path.join(out_dir, "class_summary.csv")
    html_path = os.path.join(out_dir, "index.html")

    class_counter = Counter()
    rows = []

    for img_id, img in image_items:
        file_name = img.get("file_name")
        if not file_name:
            continue
        img_path = os.path.join(images_dir, file_name)
        if not os.path.exists(img_path):
            # 尝试仅基名匹配
            base = os.path.basename(file_name)
            alt = os.path.join(images_dir, base)
            if os.path.exists(alt):
                img_path = alt
            else:
                # 跳过找不到的图
                continue

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            draw = ImageDraw.Draw(im)
            W, H = im.size

            for ann in img_to_anns.get(img_id, []):
                bbox = ann.get("bbox")  # [x,y,w,h]
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                x1 = max(0, int(round(x)))
                y1 = max(0, int(round(y)))
                x2 = min(W - 1, int(round(x + w)))
                y2 = min(H - 1, int(round(y + h)))
                cid = ann.get("category_id")
                cname = id_to_cat.get(cid, {}).get("name", str(cid))
                score = ann.get("score", None)
                if (score is not None) and (score < score_thresh):
                    continue
                draw_box(draw, (x1, y1, x2, y2), cname, line_thickness, font)
                rows.append([os.path.basename(file_name), cid, cname, x1, y1, x2, y2, score if score is not None else ""])
                class_counter[cname] += 1

            out_name = os.path.splitext(os.path.basename(file_name))[0] + "_ann.jpg"
            save_path = os.path.join(out_img_dir, out_name)
            im.save(save_path, quality=95)

    # write CSVs
    with open(ann_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image","class_id","class_name","xmin","ymin","xmax","ymax","score"])
        w.writerows(rows)

    with open(cls_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_name","count"])
        for name, cnt in class_counter.most_common():
            w.writerow([name, cnt])

    # html
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>COCO 标注预览</title></head><body>\n")
        f.write("<h1>COCO 标注预览</h1>\n")
        f.write("<p>图片总数(处理): {}；共 {} 个框。</p>\n".format(len(image_items), sum(class_counter.values())))
        f.write("<ul>\n")
        for name, cnt in class_counter.most_common():
            f.write(f"<li>{name}: {cnt}</li>\n")
        f.write("</ul>\n")
        f.write("<hr/>\n")
        for root, _, files in os.walk(out_img_dir):
            for fn in sorted(files):
                fpath = os.path.join("annotated", fn)
                f.write(f"<div><img src='{fpath}' style='max-width:100%;'/><p>{fn}</p></div><hr/>\n")
        f.write("</body></html>\n")

    print(f"[完成] 输出目录: {out_dir}")
    print(f"  - 标注图: {out_img_dir}")
    print(f"  - CSV: {ann_csv_path}, {cls_csv_path}")
    print(f"  - HTML: {html_path}")

def main():
    ap = argparse.ArgumentParser(description="将标注框可视化到图片上（支持 YOLO .txt 与 COCO .json）")
    ap.add_argument("--images", required=True, help="图片根目录")
    ap.add_argument("--labels", default=None, help="(YOLO) 标签目录；若与图片同目录，可省略")
    ap.add_argument("--classes", default=None, help="(YOLO) 类别名文件（每行一个）")
    ap.add_argument("--coco", default=None, help="COCO 标注 json（例如 instances_train.json）")
    ap.add_argument("--out", default="out_vis", help="输出目录")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 张图片")
    ap.add_argument("--line-thickness", type=int, default=2, help="框线粗细")
    ap.add_argument("--font-size", type=int, default=16, help="文本字号")
    ap.add_argument("--score-thresh", type=float, default=0.0, help="分数阈值（COCO/预测文件常见）")
    args = ap.parse_args()

    if args.coco:
        process_coco(args.images, args.coco, args.out, args.limit, args.line_thickness, args.font_size, args.score_thresh)
    else:
        process_yolo(args.images, args.labels, args.classes, args.out, args.limit, args.line_thickness, args.font_size, args.score_thresh)

if __name__ == "__main__":
    main()
