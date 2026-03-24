# train_ordinal_head_min.py
import argparse, os, math, random
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from ord2seq_head import Ord2SeqOrdinalHead


def map_ic4(icdas):
    if icdas <= 0:
        return 0
    if icdas <= 2:
        return 1
    if icdas <= 4:
        return 2
    return 3


def map_label_to_num_classes(raw_label, num_classes):
    raw_label = int(raw_label)
    if num_classes == 4:
        return map_ic4(raw_label)
    return max(0, min(num_classes - 1, raw_label))

class ROIDataset(Dataset):
    def __init__(self, csv_path, img_root, img_size=256, expand=1.15, augment=True, ordinal_dims=6, num_classes=4):
        self.df = pd.read_csv(csv_path)
        self.root = img_root
        self.size = img_size
        self.expand = expand
        self.augment = augment
        self.ordinal_dims = ordinal_dims
        self.num_classes = num_classes
        self.tx = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(os.path.join(self.root, r['image_id'])).convert('RGB')
        W,H = im.size
        if 'gx' in r:
            x,y,w,h = float(r['gx']), float(r['gy']), float(r['gw']), float(r['gh'])
        else:
            x,y,w,h = float(r['x']), float(r['y']), float(r['w']), float(r['h'])
        # 扩边
        cx, cy = x+w/2, y+h/2
        w2, h2 = w*self.expand, h*self.expand
        x1 = max(0, cx - w2/2); y1 = max(0, cy - h2/2)
        x2 = min(W, cx + w2/2); y2 = min(H, cy + h2/2)
        crop = im.crop((x1,y1,x2,y2))
        if self.augment and random.random()<0.5:
            crop = transforms.functional.hflip(crop)
        x = self.tx(crop)
        y_ge_list, m_ge_list = [], []
        for k in range(1, self.ordinal_dims + 1):
            yk = r[f'y_ge{k}'] if f'y_ge{k}' in r else 0
            mk = r[f'mask_ge{k}'] if f'mask_ge{k}' in r else 0
            yk = 0 if (isinstance(yk, float) and math.isnan(yk)) else int(yk)
            mk = 0 if (isinstance(mk, float) and math.isnan(mk)) else int(mk)
            y_ge_list.append(yk)
            m_ge_list.append(mk)
        y_ge = torch.tensor(y_ge_list, dtype=torch.float32)
        m_ge = torch.tensor(m_ge_list, dtype=torch.float32)

        if 'ic4' in r:
            cls = int(r['ic4'])
        elif 'gt_icdas' in r:
            cls = map_label_to_num_classes(r['gt_icdas'], self.num_classes)
        elif 'icdas' in r:
            cls = map_label_to_num_classes(r['icdas'], self.num_classes)
        else:
            # Fallback: infer class by counting positive thresholds.
            usable = min(self.num_classes - 1, self.ordinal_dims)
            cls = int(sum(y_ge_list[:usable]))

        return x, y_ge, m_ge, torch.tensor(cls, dtype=torch.long)

class OrdinalHead(nn.Module):
    def __init__(
        self,
        pretrained=True,
        out_dims=6,
        head_type='masked',
        num_classes=4,
        ord2seq_d_model=256,
        ord2seq_layers=2,
    ):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        feat = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.head_type = head_type
        if self.head_type == 'masked':
            self.head = nn.Linear(feat, out_dims)
        elif self.head_type == 'ord2seq':
            self.ord_head = Ord2SeqOrdinalHead(
                in_features=feat,
                num_classes=num_classes,
                d_model=ord2seq_d_model,
                nhead=8,
                num_decoder_layers=ord2seq_layers,
                dim_feedforward=ord2seq_d_model * 4,
                dropout=0.1,
                use_masked_decision=True,
            )
        else:
            raise ValueError(f'Unknown head_type: {self.head_type}')

    def forward(self, x, labels=None):
        feat = self.backbone(x)
        if self.head_type == 'masked':
            return {'logits': self.head(feat)}
        return self.ord_head(feat, labels=labels)

def loss_masked_ordinal(z, y, m, lambda_mono=0.1):
    p = torch.sigmoid(z)
    bce = F.binary_cross_entropy(p, y, reduction='none')
    bce = (bce*m).sum(dim=1) / (m.sum(dim=1)+1e-6)
    # 单调正则：z_{k+1} >= z_k
    mono = F.softplus(z[:,1:] - z[:,:-1]).mean(dim=1)
    return (bce + lambda_mono*mono).mean()

@torch.no_grad()
def evaluate(model, loader, device, head_type, lambda_mono):
    model.eval(); losses=[]
    for x,y,m,cls in loader:
        x,y,m,cls = x.to(device), y.to(device), m.to(device), cls.to(device)
        if head_type == 'masked':
            z = model(x)['logits']
            losses.append(loss_masked_ordinal(z, y, m, lambda_mono=lambda_mono).item())
        else:
            out = model(x, labels=cls)
            losses.append(out['loss'].item())
    return sum(losses)/len(losses)

def main(a):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OrdinalHead(
        pretrained=True,
        out_dims=a.ordinal_dims,
        head_type=a.head_type,
        num_classes=a.num_classes,
        ord2seq_d_model=a.ord2seq_d_model,
        ord2seq_layers=a.ord2seq_layers,
    ).to(device)
    tr = ROIDataset(
        a.train_csv,
        a.img_root_train,
        a.img_size,
        a.expand,
        augment=True,
        ordinal_dims=a.ordinal_dims,
        num_classes=a.num_classes,
    )
    va = ROIDataset(
        a.val_csv,
        a.img_root_val,
        a.img_size,
        a.expand,
        augment=False,
        ordinal_dims=a.ordinal_dims,
        num_classes=a.num_classes,
    )
    tr_loader = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(va, batch_size=a.bs, shuffle=False, num_workers=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    best = 1e9

    if a.head_type == 'ord2seq' and abs(a.lambda_mono) > 1e-12:
        print('Info: head_type=ord2seq 时不使用 lambda_mono（仅 masked 模式使用）。')

    for ep in range(a.epochs):
        model.train()
        for x,y,m,cls in tr_loader:
            x,y,m,cls = x.to(device), y.to(device), m.to(device), cls.to(device)
            if a.head_type == 'masked':
                z = model(x)['logits']
                loss = loss_masked_ordinal(z, y, m, lambda_mono=a.lambda_mono)
            else:
                out = model(x, labels=cls)
                loss = out['loss']
            opt.zero_grad(); loss.backward(); opt.step()
        val_loss = evaluate(model, va_loader, device, a.head_type, a.lambda_mono)
        print(f'Epoch {ep+1}/{a.epochs}  val_loss={val_loss:.4f}')
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), a.out)
    print('best val_loss:', best)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--val_csv',   required=True)
    ap.add_argument('--img_root_train', required=True)
    ap.add_argument('--img_root_val',   required=True)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--expand', type=float, default=1.15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lambda_mono', type=float, default=0.1)
    ap.add_argument('--head_type', '--head-type', dest='head_type', type=str, default='masked', choices=['masked', 'ord2seq'])
    ap.add_argument('--num_classes', '--num-classes', dest='num_classes', type=int, default=4)
    ap.add_argument('--ordinal_dims', '--ordinal-dims', dest='ordinal_dims', type=int, default=6)
    ap.add_argument('--ord2seq_d_model', '--ord2seq-d-model', dest='ord2seq_d_model', type=int, default=256)
    ap.add_argument('--ord2seq_layers', '--ord2seq-layers', dest='ord2seq_layers', type=int, default=2)
    ap.add_argument('--out', default='ordinal_head_resnet18.pt')
    a = ap.parse_args()
    main(a)
