import pandas as pd, os
for split in ['val','test']:
    p = 'runs/detect/public-2class/v8s_bin_1280_ab2_tail/low_thresh_predictions_icdas_conf0.001/{}_predictions.csv'.format(split)
    df = pd.read_csv(p)
    df['image_id'] = df['image_id'].astype(str).apply(os.path.basename)
    df.to_csv(p, index=False)
    print(split, 'images:', df['image_id'].nunique(), 'cands:', len(df), 'avg/image:', len(df)/max(1,df['image_id'].nunique()))