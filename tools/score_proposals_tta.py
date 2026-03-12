import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torchvision.models as models

# Define the Model (Must match training)
class OrdinalHead(nn.Module):
    def __init__(self, num_classes=4, feature_dim=512): # ResNet18 default
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        # Replace fc
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Ordinal output: K-1 binary classifiers
        # For 4 classes (0,1,2,3), we need 3 outputs: >=1, >=2, >=3
        # But wait, your previous context mentioned p_ge1, p_ge3, p_ge5? 
        # Let's assume standard ordinal: output dim = num_classes - 1
        self.head = nn.Linear(num_ftrs, num_classes - 1)
        
    def forward(self, x):
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits

class ProposalDataset(Dataset):
    def __init__(self, csv_file, img_root, img_size=256, expand=1.25):
        self.df = pd.read_csv(csv_file)
        self.img_root = img_root
        self.img_size = img_size
        self.expand = expand
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row['image_id'])
        
        try:
            image = Image.open(img_path).convert('RGB')
            W, H = image.size
            
            # Crop ROI
            x, y, w, h = row['x'], row['y'], row['w'], row['h']
            
            # Expand
            cx, cy = x + w/2, y + h/2
            w_new = w * self.expand
            h_new = h * self.expand
            x1 = max(0, int(cx - w_new/2))
            y1 = max(0, int(cy - h_new/2))
            x2 = min(W, int(cx + w_new/2))
            y2 = min(H, int(cy + h_new/2))
            
            roi = image.crop((x1, y1, x2, y2))
            
            # TTA: Original and Flip
            roi_flip = roi.transpose(Image.FLIP_LEFT_RIGHT)
            
            return self.transform(roi), self.transform(roi_flip)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy
            dummy = torch.zeros(3, self.img_size, self.img_size)
            return dummy, dummy

def score_proposals_tta(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    # Note: We need to know exactly how many outputs the model has.
    # Based on previous context (p_ge1, p_ge3, p_ge5), it seems to be 3 outputs.
    # BUT the checkpoint says 6 outputs! This means it was trained with 6 classes (>=1..>=6)
    # even if we only care about 1, 3, 5.
    model = OrdinalHead(num_classes=7) # 7 classes -> 6 outputs (>=1 to >=6)
    
    # Load weights
    ckpt = torch.load(args.ckpt, map_location=device)
    # Handle state dict key mismatch if any
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
        
    model.to(device)
    model.eval()
    
    dataset = ProposalDataset(args.pred_csv, args.img_root, args.img_size, args.expand)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=4)
    
    p_ge_list = []
    
    print("Scoring with TTA...")
    with torch.no_grad():
        for imgs, imgs_flip in tqdm(dataloader):
            imgs = imgs.to(device)
            imgs_flip = imgs_flip.to(device)
            
            # Forward pass
            logits = model(imgs)
            logits_flip = model(imgs_flip)
            
            # Sigmoid
            probs = torch.sigmoid(logits)
            probs_flip = torch.sigmoid(logits_flip)
            
            # Average probabilities (TTA)
            probs_avg = (probs + probs_flip) / 2.0
            
            p_ge_list.append(probs_avg.cpu().numpy())
            
    p_ge_all = np.concatenate(p_ge_list, axis=0)
    
    # Save results
    df = pd.read_csv(args.pred_csv)
    
    # Assuming model outputs 6 values corresponding to >=1, >=2, >=3, >=4, >=5, >=6
    # We only need p_ge1 (idx 0), p_ge3 (idx 2), p_ge5 (idx 4)
    
    df['p_ge1'] = p_ge_all[:, 0]
    df['p_ge3'] = p_ge_all[:, 2]
    df['p_ge5'] = p_ge_all[:, 4]
    
    # Backup raw score if not exists
    if 'yolo_score_raw' not in df.columns:
        df['yolo_score_raw'] = df['yolo_score']
        
    df.to_csv(args.out_csv, index=False)
    print(f"Saved TTA results to {args.out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--expand", type=float, default=1.25)
    parser.add_argument("--bs", type=int, default=64)
    
    args = parser.parse_args()
    score_proposals_tta(args)
