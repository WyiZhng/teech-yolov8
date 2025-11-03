# list_missing_images.py
import pandas as pd, os, glob, sys
split = sys.argv[1]  # 'val' 或 'test'
pred_csv = f'runs/detect/public-2class/v8s_bin_1280_ab2_tail/low_thresh_predictions_icdas_conf0.001/{split}_predictions.csv'
img_dir  = f'/data/HZNU_ZWY/zwy_project/ultralytics-8.1.35/VOCdevkit/{split}/images'

pred = pd.read_csv(pred_csv)
pred['image_id'] = pred['image_id'].astype(str).apply(os.path.basename)

imgs = []
for e in ('*.jpg','*.jpeg','*.png','*.bmp'):
    imgs += glob.glob(os.path.join(img_dir, e))
imgs = sorted(os.path.basename(x) for x in imgs)
covered = set(pred['image_id'])

missing = [x for x in imgs if x not in covered]
print(f'{split} total={len(imgs)} covered={len(covered)} missing={len(missing)}')
for m in missing[:50]:
    print('[MISSING]', m)
