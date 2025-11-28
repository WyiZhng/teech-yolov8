# tools/remap_icdas_multiclass_to_single.py
import os, shutil

SRC="datasets/icdas_yolo"           # 你现在的多类数据
DST="datasets/icdas_yolo_caries"    # 输出单类数据

os.makedirs(DST, exist_ok=True)
for sp in ["train","val","test"]:
    os.makedirs(f"{DST}/{sp}/images", exist_ok=True)
    os.makedirs(f"{DST}/{sp}/labels", exist_ok=True)
    # 复制图片
    for f in os.listdir(f"{SRC}/{sp}/images"):
        shutil.copy2(f"{SRC}/{sp}/images/{f}", f"{DST}/{sp}/images/{f}")
    # 处理标注：仅保留 1..6，映射为 0；丢弃 0（无龋）
    for f in os.listdir(f"{SRC}/{sp}/labels"):
        src = f"{SRC}/{sp}/labels/{f}"
        dst = f"{DST}/{sp}/labels/{f}"
        out = []
        if os.path.exists(src):
            for ln in open(src):
                p = ln.strip().split()
                if len(p)!=5: continue
                c, cx, cy, w, h = p
                ci = int(float(c))
                if 1 <= ci <= 6:
                    out.append(f"0 {cx} {cy} {w} {h}")  # 映射为单类 caries=0
                # ci==0（无龋）直接丢弃
        open(dst, "w").write("\n".join(out))  # 可能为空文件 -> 表示该图“无目标”
print("done -> datasets/icdas_yolo_caries")
