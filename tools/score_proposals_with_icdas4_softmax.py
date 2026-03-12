# tools/score_proposals_with_icdas4_softmax.py
import os, argparse, pandas as pd, numpy as np, torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

def norm_id(s): return os.path.basename(str(s)).strip()

class ResNet18Softmax4(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        # 推理时不需要预训练权重，因为会加载 checkpoint
        m = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # 去掉原 fc
        self.fc = nn.Linear(m.fc.in_features, 4)

    def forward(self, x):
        feat = self.backbone(x)   # [B,512,1,1]
        feat = feat.flatten(1)    # [B,512]
        logits = self.fc(feat)    # [B,4]
        return logits

def main(a):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv(a.pred_csv)
    df['image_id'] = df['image_id'].astype(str).apply(norm_id)
    if 'yolo_score_raw' not in df.columns:
        df['yolo_score_raw'] = df['yolo_score']

    # 加载模型
    model = ResNet18Softmax4(pretrained=False).to(device)
    sd = torch.load(a.ckpt, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    tx = transforms.Compose([
        transforms.Resize((a.img_size,a.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    def crop_tensor(p,x,y,w,h, expand=1.25):
        with Image.open(p) as im:
            im = im.convert('RGB'); W,H = im.size
            cx,cy = x+w/2, y+h/2; w2,h2 = w*expand, h*expand
            x1=max(0,cx-w2/2); y1=max(0,cy-h2/2); x2=min(W,cx+w2/2); y2=min(H,cy+h2/2)
            return tx(im.crop((x1,y1,x2,y2)))

    # 若坐标是归一化，自动还原（w/h的95分位<=2 视为归一化）
    wh95 = max(df['w'].quantile(0.95), df['h'].quantile(0.95))
    if wh95 <= 2.0:
        size_cache={}
        xs,ys,ws,hs=[],[],[],[]
        for _,r in df.iterrows():
            im = r['image_id']
            if im not in size_cache:
                from PIL import Image as _I
                W,H = _I.open(os.path.join(a.img_root, im)).size
                size_cache[im]=(W,H)
            W,H=size_cache[im]
            xs.append(float(r['x'])*W); ys.append(float(r['y'])*H)
            ws.append(float(r['w'])*W); hs.append(float(r['h'])*H)
        df['x'],df['y'],df['w'],df['h']=xs,ys,ws,hs

    p1s,p3s,p5s=[],[],[]
    with torch.no_grad():
        batch=[]; 
        def flush():
            nonlocal p1s,p3s,p5s,batch
            if not batch: return
            x=torch.stack(batch).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1) # [B, 4]
            
            # p0, pA, pB, pC = probs[:,0], probs[:,1], probs[:,2], probs[:,3]
            pA = probs[:, 1]
            pB = probs[:, 2]
            pC = probs[:, 3]
            
            # p_ge1 = pA + pB + pC
            # p_ge3 = pB + pC
            # p_ge5 = pC
            
            p_ge1 = pA + pB + pC
            p_ge3 = pB + pC
            p_ge5 = pC
            
            p1s += p_ge1.cpu().numpy().tolist()
            p3s += p_ge3.cpu().numpy().tolist()
            p5s += p_ge5.cpu().numpy().tolist()
            batch=[]

        for i,r in df.iterrows():
            ip = os.path.join(a.img_root, r['image_id'])
            batch.append(crop_tensor(ip, float(r['x']),float(r['y']),float(r['w']),float(r['h']), a.expand))
            if len(batch)==a.bs: flush()
        flush()

    df['p_ge1']=p1s; df['p_ge3']=p3s; df['p_ge5']=p5s
    df.to_csv(a.out_csv, index=False)
    print("Wrote:", a.out_csv, "rows:", len(df))

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_csv', required=True)   # 候选csv
    ap.add_argument('--img_root', required=True)   # 图像根目录
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--expand', type=float, default=1.25)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--out_csv', required=True)
    a = ap.parse_args(); main(a)
