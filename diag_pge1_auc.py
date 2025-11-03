import os, argparse, pandas as pd, numpy as np, torch
from PIL import Image
from torchvision import transforms, models

class OrdinalHead(torch.nn.Module):
    def __init__(self, out_dims=6):
        super().__init__()
        m = models.resnet18(weights=None); feat=m.fc.in_features; m.fc=torch.nn.Identity()
        self.backbone=m; self.head=torch.nn.Linear(feat,out_dims)
    def forward(self,x): return self.head(self.backbone(x))

def iou_xywh(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ax2,ay2=ax+aw,ay+ah; bx2,by2=bx+bw,by+bh
    ix1,iy1=max(ax,bx),max(ay,by); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=aw*ah + bw*bh - inter + 1e-6
    return inter/ua

def main(a):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    # load
    pred = pd.read_csv(a.pred_csv); pred['image_id']=pred['image_id'].apply(os.path.basename)
    gt   = pd.read_csv(a.gt_csv);   gt['image_id']=gt['image_id'].apply(os.path.basename)
    gt   = gt[gt['icdas']>=1].copy()

    # model
    model=OrdinalHead().to(device); model.load_state_dict(torch.load(a.ckpt, map_location=device)); model.eval()
    tx = transforms.Compose([
        transforms.Resize((a.img_size,a.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    # p_ge1
    def crop(image_id,x,y,w,h,expand=1.25):
        p=os.path.join(a.img_root,image_id); im=Image.open(p).convert('RGB'); W,H=im.size
        cx,cy=x+w/2,y+h/2; w2,h2=w*expand,h*expand
        x1=max(0,cx-w2/2); y1=max(0,cy-h2/2); x2=min(W,cx+w2/2); y2=min(H,cy+h2/2)
        return tx(im.crop((x1,y1,x2,y2)))
    ps=[]; metas=[]
    with torch.no_grad():
        batch=[]; metas=[]
        for i,r in pred.iterrows():
            batch.append(crop(r['image_id'], r['x'],r['y'],r['w'],r['h'], a.expand))
            metas.append((r['image_id'], r['x'],r['y'],r['w'],r['h']))
            if len(batch)==a.bs or i==len(pred)-1:
                x=torch.stack(batch).to(device)
                z=model(x); p=torch.sigmoid(z)[:,0].cpu().numpy().tolist()
                ps += p; batch=[]
    pred['p_ge1']=ps

    # 打命中标签
    y=[]
    for _,r in pred.iterrows():
        gimg = gt[gt['image_id']==r['image_id']]
        hit=False
        for _,g in gimg.iterrows():
            if iou_xywh((r['x'],r['y'],r['w'],r['h']), (g['gx'],g['gy'],g['gw'],g['gh']))>=0.5:
                hit=True; break
        y.append(int(hit))
    pred['hit']=y

    # AUC & PR-AUC
    from sklearn.metrics import roc_auc_score, average_precision_score
    y_true = pred['hit'].values
    scores = pred['p_ge1'].values
    auc = roc_auc_score(y_true, scores) if y_true.sum()>0 else float('nan')
    ap  = average_precision_score(y_true, scores) if y_true.sum()>0 else float('nan')

    # Top-1 命中率（按 p_ge1 排序，逐图取第一框）
    top_hits=0; total_imgs=pred['image_id'].nunique()
    for img,g in pred.groupby('image_id'):
        g=g.sort_values('p_ge1', ascending=False).head(1)
        top_hits += int(g['hit'].iloc[0]==1)
    top1 = top_hits/max(1,total_imgs)

    print(f"AUC={auc:.3f}  AP={ap:.3f}  Top1@p_ge1={top1:.3f}  (imgs={total_imgs}, pos={y_true.sum()})")

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--pred_csv', required=True)
    ap.add_argument('--gt_csv', required=True)
    ap.add_argument('--img_root', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--expand', type=float, default=1.25)
    ap.add_argument('--bs', type=int, default=128)
    a=ap.parse_args(); main(a)
