import pandas as pd, os
def prop_recall(pred_csv, gt_csv, iou_thr=0.5):
    pred = pd.read_csv(pred_csv); pred['image_id'] = pred['image_id'].astype(str).apply(os.path.basename)
    gt   = pd.read_csv(gt_csv);   gt['image_id']   = gt['image_id'].astype(str).apply(os.path.basename)
    gt = gt[gt['icdas']>=1].copy()
    def iou_xywh(a,b):
        ax,ay,aw,ah=a; bx,by,bw,bh=b
        ax2,ay2=ax+aw,ay+ah; bx2,by2=bx+bw,by+bh
        ix1,iy1=max(ax,bx),max(ay,by); ix2,iy2=min(ax2,bx2),min(ay2,by2)
        iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
        ua=aw*ah+bw*bh-inter+1e-6; return inter/ua
    hits=0; total=len(gt)
    for img,g in gt.groupby('image_id'):
        cand = pred[pred['image_id']==img]
        for _,r in g.iterrows():
            gbox=(r['gx'],r['gy'],r['gw'],r['gh'])
            ok=False
            for _,c in cand.iterrows():
                cbox=(c['x'],c['y'],c['w'],c['h'])
                if iou_xywh(gbox,cbox)>=iou_thr:
                    ok=True; break
            hits+=int(ok)
    return hits, total, hits/max(total,1)

val_pred  = 'runs/detect/public-2class/v8s_bin_1280_ab2_tail/low_thresh_predictions_icdas_conf1e-4/val_predictions.csv'
test_pred = 'runs/detect/public-2class/v8s_bin_1280_ab2_tail/low_thresh_predictions_icdas_conf1e-4/test_predictions.csv'

print('VAL prop recall@0.5:', prop_recall(val_pred,  'icdas_strong_labels_val.csv'))
print('TEST prop recall@0.5:', prop_recall(test_pred, 'icdas_strong_labels_test.csv'))
print('VAL prop recall@0.3:', prop_recall(val_pred,  'icdas_strong_labels_val.csv', 0.3))
print('TEST prop recall@0.3:', prop_recall(test_pred, 'icdas_strong_labels_test.csv',0.3))
