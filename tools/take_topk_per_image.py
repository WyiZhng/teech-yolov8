# tools/take_topk_per_image.py
import argparse, pandas as pd
ap=argparse.ArgumentParser()
ap.add_argument('--in_csv', required=True)
ap.add_argument('--out_csv', required=True)
ap.add_argument('--k', type=int, default=400)
a=ap.parse_args()

df=pd.read_csv(a.in_csv)
df=df.sort_values(['image_id','yolo_score'], ascending=[True,False]) \
     .groupby('image_id').head(a.k).reset_index(drop=True)
df.to_csv(a.out_csv, index=False)
print("Wrote:", a.out_csv, "rows:", len(df))
