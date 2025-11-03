import pandas as pd, os, glob

GT = '/data/HZNU_ZWY/zwy_project/ultralytics-8.1.35/icdas_strong_labels.csv'
VAL_DIR  = '/data/HZNU_ZWY/zwy_project/ultralytics-8.1.35/VOCdevkit/val/images'
TEST_DIR = '/data/HZNU_ZWY/zwy_project/ultralytics-8.1.35/VOCdevkit/test/images'

gt = pd.read_csv(GT)
gt['image_id'] = gt['image_id'].astype(str).apply(os.path.basename)

def list_imgs(folder):
    files=[]
    for e in ('*.jpg','*.jpeg','*.png','*.bmp'):
        files += glob.glob(os.path.join(folder, e))
    return set(os.path.basename(x) for x in files)

val_set  = list_imgs(VAL_DIR)
test_set = list_imgs(TEST_DIR)

gt_val  = gt[gt['image_id'].isin(val_set)].copy()
gt_test = gt[gt['image_id'].isin(test_set)].copy()
gt_val.to_csv('icdas_strong_labels_val.csv', index=False)
gt_test.to_csv('icdas_strong_labels_test.csv', index=False)

print('GT-val imgs:', gt_val['image_id'].nunique(), 'pos boxes:', (gt_val['icdas']>=1).sum())
print('GT-test imgs:', gt_test['image_id'].nunique(), 'pos boxes:', (gt_test['icdas']>=1).sum())
