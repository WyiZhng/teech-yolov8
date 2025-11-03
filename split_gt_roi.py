# split_gt_roi.py
import pandas as pd, numpy as np

df = pd.read_csv('roi_training_gt_strong.train.csv')
# 只保留我们需要的列（防止意外多列）
base_cols = ['image_id','x','y','w','h','gt_icdas'] \
           + [f'y_ge{k}' for k in range(1,7)] \
           + [f'mask_ge{k}' for k in range(1,7)]
df = df[base_cols].copy()

# 按 image_id 分组 8:2 切分
rng = np.random.default_rng(2025)
imgs = df['image_id'].unique()
rng.shuffle(imgs)
n_val = max(1, int(0.2*len(imgs)))
val_imgs = set(imgs[:n_val])

train = df[~df['image_id'].isin(val_imgs)].reset_index(drop=True)
val   = df[ df['image_id'].isin(val_imgs)].reset_index(drop=True)

train.to_csv('roi_training_gt_strong.train.csv', index=False)
val.to_csv('roi_training_gt_strong.val.csv', index=False)

print('train images:', train['image_id'].nunique(),
      'val images:', val['image_id'].nunique(),
      'train rois:', len(train), 'val rois:', len(val))
