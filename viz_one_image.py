# viz_one_image.py
import pandas as pd, os
from PIL import Image, ImageDraw

IMG = '/data/HZNU_ZWY/zwy_project/ultralytics-8.1.35/VOCdevkit/val/images/24.jpg'
PRED= 'runs/detect/public-2class/v8s_bin_1280_ab2_tail/low_thresh_predictions_icdas_conf1e-4/val_predictions.csv'
GT  = 'icdas_strong_labels_val.csv'  # 你前面已按split过滤好的那份

pred = pd.read_csv(PRED); pred['image_id']=pred['image_id'].apply(os.path.basename)
gt   = pd.read_csv(GT);   gt['image_id']=gt['image_id'].apply(os.path.basename)

img_id = os.path.basename(IMG)
p = pred[pred['image_id']==img_id].sort_values('yolo_score', ascending=False).head(50)
g = gt[gt['image_id']==img_id]  # icdas>=1可再过滤

im = Image.open(IMG).convert('RGB')
draw = ImageDraw.Draw(im)

# 预测：蓝色
for _,r in p.iterrows():
    x1,y1 = r['x'], r['y']
    x2,y2 = x1+r['w'], y1+r['h']
    draw.rectangle((x1,y1,x2,y2), outline=(0,0,255), width=2)

# GT：红色
for _,r in g.iterrows():
    x1,y1 = r['gx'], r['gy']
    x2,y2 = x1+r['gw'], y1+r['gh']
    draw.rectangle((x1,y1,x2,y2), outline=(255,0,0), width=3)

im.save('debug_vis.jpg'); print('saved debug_vis.jpg')
