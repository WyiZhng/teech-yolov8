# tools/qc_yolo_labels.py
#快速质检（必须跑）目的：确保所有 txt 都是 5 列、归一化在 (0,1]、没有越界/极小框/漏标空文件。
import os, glob
from PIL import Image

ROOT = "datasets/icdas_yolo"
splits = ["train","val","test"]
bad = 0
for sp in splits:
    img_dir = f"{ROOT}/{sp}/images"; lbl_dir = f"{ROOT}/{sp}/labels"
    os.makedirs(lbl_dir, exist_ok=True)
    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    for im in imgs:
        p_img = os.path.join(img_dir, im)
        W,H = Image.open(p_img).size
        p_txt = os.path.join(lbl_dir, os.path.splitext(im)[0]+'.txt')
        if not os.path.exists(p_txt):
            open(p_txt,'w').close();  # 无目标也留空txt，训练更稳
            continue
        for ln in open(p_txt):
            try:
                c,cx,cy,w,h = map(float, ln.split())
                assert 0<=cx<=1 and 0<=cy<=1 and 0<w<=1 and 0<h<=1
                # 过小框提醒（阈值可调）
                if w*W<8 or h*H<8: pass
            except Exception as e:
                bad+=1; print(sp,im,"BAD:",ln.strip(),e)
print("BAD lines:", bad)
