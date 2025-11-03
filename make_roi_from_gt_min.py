import pandas as pd
df = pd.read_csv('icdas_strong_labels.csv')
df = df.rename(columns={'gx':'x','gy':'y','gw':'w','gh':'h','icdas':'gt_icdas'})
for k in range(1, 7):
    df[f'y_ge{k}'] = (df['gt_icdas'] >= k).astype(int)
    df[f'mask_ge{k}'] = 1
cols = ['image_id','x','y','w','h','gt_icdas'] + [f'y_ge{k}' for k in range(1,7)] + [f'mask_ge{k}' for k in range(1,7)]
df[cols].to_csv('roi_training_gt_strong.train.csv', index=False)
#超短版（~10 行 pandas）：把 icdas_strong_labels.csv 直接展开成序位标签与 mask，立刻可喂训练。