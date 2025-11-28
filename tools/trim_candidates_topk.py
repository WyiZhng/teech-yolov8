import pandas as pd, argparse
ap=argparse.ArgumentParser()
ap.add_argument('--pred_csv', required=True)
ap.add_argument('--out_csv',  required=True)
ap.add_argument('--topk', type=int, default=300)
a=ap.parse_args()

df = pd.read_csv(a.pred_csv)
if 'yolo_score' not in df.columns:
    raise SystemExit("yolo_score 列缺失")
out = df.sort_values(['image_id','yolo_score'], ascending=[True,False]) \
        .groupby('image_id').head(a.topk).reset_index(drop=True)
out.to_csv(a.out_csv, index=False)
print("wrote:", a.out_csv, " rows:", len(out))
