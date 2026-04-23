import pandas as pd
import sys

def calc_topk(csv_path):
    df = pd.read_csv(csv_path)
    if 'gt_class' in df.columns:
        y_gt_col = 'gt_class'
    elif 'ic4' in df.columns:
        y_gt_col = 'ic4'
    else:
        # try to find anything that looks like gt
        y_gt_col = [c for c in df.columns if 'gt' in c or 'ic' in c][0]
    
    top1_hit, top3_hit = 0, 0
    
    # Identify positive images (at least one ROI with GT >= 1)
    pos_imgs = df.groupby('image_id')[y_gt_col].apply(lambda x: (x >= 1).any())
    n_pos_imgs = pos_imgs.sum()
    n_imgs = df['image_id'].nunique()
    
    for img_id, g in df.groupby('image_id'):
        if not (g[y_gt_col] >= 1).any():
            continue
        
        g_sorted = g.sort_values('p_ge1', ascending=False)
        if (g_sorted.head(1)[y_gt_col] >= 1).any():
            top1_hit += 1
        if (g_sorted.head(3)[y_gt_col] >= 1).any():
            top3_hit += 1
            
    print(f"File: {csv_path}")
    print(f"Total Images: {n_imgs}")
    print(f"Positive Images (GT>=1): {n_pos_imgs}")
    if n_pos_imgs > 0:
        print(f"Top-1: {top1_hit}/{n_pos_imgs} = {top1_hit/n_pos_imgs:.3f}")
        print(f"Top-3: {top3_hit}/{n_pos_imgs} = {top3_hit/n_pos_imgs:.3f}")
    else:
        print("No positive images found.")

if __name__ == "__main__":
    for path in sys.argv[1:]:
        calc_topk(path)
