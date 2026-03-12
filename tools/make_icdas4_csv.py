# tools/make_icdas4_csv.py
import pandas as pd, argparse, os

def norm(s): return os.path.basename(str(s)).strip()

ap=argparse.ArgumentParser()
ap.add_argument('--in_csv', required=True)
ap.add_argument('--out_csv', required=True)
a=ap.parse_args()

df = pd.read_csv(a.in_csv)

# 归一化文件名
df['image_id'] = df['image_id'].astype(str).apply(norm)

# 4级映射：0 / A=1-2 / B=3-4 / C=5-6
def map_ic4(x):
    if x<=0: return 0
    if x<=2: return 1
    if x<=4: return 2
    return 3
df['ic4'] = df['icdas'].apply(map_ic4)

# 复制一份坐标到 x,y,w,h（训练脚本需要）
need = {'gx','gy','gw','gh'}
missing = [c for c in need if c not in df.columns]
if missing:
    raise SystemExit(f"缺少列 {missing}，请先确认输入CSV包含 gx,gy,gw,gh")

df['x'] = df['gx']; df['y'] = df['gy']; df['w'] = df['gw']; df['h'] = df['gh']

# 三个有效阈值：≥1(A+), ≥3(B+), ≥5(C+)
df['y_ge1'] = (df['icdas']>=1).astype(int)
df['y_ge2'] = 0
df['y_ge3'] = (df['icdas']>=3).astype(int)
df['y_ge4'] = 0
df['y_ge5'] = (df['icdas']>=5).astype(int)
df['y_ge6'] = 0

# 掩码：只监督 1/3/5
df['mask_ge1']=1; df['mask_ge2']=0
df['mask_ge3']=1; df['mask_ge4']=0
df['mask_ge5']=1; df['mask_ge6']=0

cols = ['image_id','x','y','w','h','gx','gy','gw','gh','icdas','ic4',
        'y_ge1','y_ge2','y_ge3','y_ge4','y_ge5','y_ge6',
        'mask_ge1','mask_ge2','mask_ge3','mask_ge4','mask_ge5','mask_ge6']
df[cols].to_csv(a.out_csv, index=False)
print("wrote:", a.out_csv, "rows:", len(df))
# # tools/make_icdas4_csv.py
# import pandas as pd, argparse
# ap=argparse.ArgumentParser()
# ap.add_argument('--in_csv', required=True)
# ap.add_argument('--out_csv', required=True)
# a=ap.parse_args()

# df = pd.read_csv(a.in_csv)
# # 4级标签：0,1,2,3
# def map_ic4(x):
#     if x<=0: return 0
#     if x<=2: return 1  # A:1-2
#     if x<=4: return 2  # B:3-4
#     return 3           # C:5-6
# df['ic4'] = df['icdas'].apply(map_ic4)

# # 三个有效阈值：≥1(A+), ≥3(B+), ≥5(C+)
# df['y_ge1'] = (df['icdas']>=1).astype(int)  # A+
# df['y_ge2'] = 0
# df['y_ge3'] = (df['icdas']>=3).astype(int)  # B+
# df['y_ge4'] = 0
# df['y_ge5'] = (df['icdas']>=5).astype(int)  # C+
# df['y_ge6'] = 0

# # mask：只监督 1/3/5
# df['mask_ge1']=1; df['mask_ge2']=0
# df['mask_ge3']=1; df['mask_ge4']=0
# df['mask_ge5']=1; df['mask_ge6']=0

# keep = ['image_id','gx','gy','gw','gh','icdas','ic4',
#         'y_ge1','y_ge2','y_ge3','y_ge4','y_ge5','y_ge6',
#         'mask_ge1','mask_ge2','mask_ge3','mask_ge4','mask_ge5','mask_ge6']
# df[keep].to_csv(a.out_csv, index=False)
# print("wrote:", a.out_csv, "rows:", len(df))
