# filter_csv.py
import os, pandas as pd, glob

def keep_existing(csv_in, img_root, csv_out):
    df = pd.read_csv(csv_in)
    exist = df['image_id'].apply(lambda x: os.path.exists(os.path.join(img_root, str(x))))
    df[exist].to_csv(csv_out, index=False)
    print(f"kept {exist.sum()}/{len(df)} rows -> {csv_out}")

# 仅保留 train 根目录下能找到的样本
keep_existing('roi_training_gt_strong.train.csv', 'VOCdevkit/train/images',
              'roi_training_gt_strong.train.in_root.csv')

# 仅保留 val 根目录下能找到的样本
keep_existing('roi_training_gt_strong.val.csv',   'VOCdevkit/val/images',
              'roi_training_gt_strong.val.in_root.csv')
