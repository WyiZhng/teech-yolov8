# tools/make_enriched_topk.py
import os, argparse, pandas as pd, numpy as np, torch
from PIL import Image
from torchvision import transforms, models

def norm_id(s): return os.path.basename(str(s)).strip()

class OrdinalHead(torch.nn.Module):
    def __init__(self, out_dims=6):
        super().__init__()
        m = models.resnet18(weights=None); feat=m.fc.in_features; m.fc=torch.nn.Identity()
        self.backbone=m; self.head=torch.nn.Linear(feat, out_dims)
    def forward(self,x): return self.head(self.backbone(x))

def main(a):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv(a.pred_csv); df['image_id']=df['image_id'].apply(norm_id)

    # 预处理
    tx = transforms.Compose([
        transforms.Resize((a.img_size,a.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    model=OrdinalHead().to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()

    # 计算 p_ge1
    def crop(image_id,x,y,w,h,expand=1.25):
        p=os.path.join(a.img_root,image_id)
        with Image.open(p) as im:
            im=im.convert('RGB'); W,H=im.size
            cx,cy=x+w/2,y+h/2; w2,h2=w*expand,h*expand
            x1=max(0,cx-w2/2); y1=max(0,cy-h2/2); x2=min(W,cx+w2/2); y2=min(H,cy+h2/2)
            return tx(im.crop((x1,y1,x2,y2)))

    ps=[]; batch=[]
    with torch.no_grad():
        bmeta=[]
        for i,r in df.iterrows():
            batch.append(crop(r['image_id'], float(r['x']),float(r['y']),float(r['w']),float(r['h']), a.expand))
            if len(batch)==a.bs or i==len(df)-1:
                x=torch.stack(batch).to(device)
                z=model(x); p=torch.sigmoid(z)[:,0].cpu().numpy().tolist()
                ps.extend(p); batch=[]

    df['p_ge1']=ps
    # 备份原 YOLO 分，计算融合分
    df['yolo_score_raw'] = df['yolo_score']
    s = df['yolo_score_raw'].clip(1e-9,1.0).values
    p = df['p_ge1'].clip(1e-9,1.0).values

    mode=a.mode.lower()
    if mode=='alpha':
        fused = a.alpha*s + (1-a.alpha)*p
    elif mode=='gamma':
        fused = s * (p ** a.gamma)
    elif mode=='sqrt':
        fused = np.sqrt(s*p)
    elif mode=='ponly':
        fused = p
    else:
        fused = a.alpha*s + (1-a.alpha)*p
    df['score_fused']=fused

    df.to_csv(a.out_csv, index=False)
    print("wrote:", a.out_csv, "rows:", len(df))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--pred_csv', required=True)   # 你的 topK CSV（未NMS）
    ap.add_argument('--img_root', required=True)   # 对应 images 根目录
    ap.add_argument('--ckpt', required=True)       # 序位头权重
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--expand', type=float, default=1.25)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--mode', choices=['alpha','gamma','sqrt','ponly'], default='alpha')
    ap.add_argument('--alpha', type=float, default=0.3)
    ap.add_argument('--gamma', type=float, default=0.8)
    a=ap.parse_args(); main(a)
