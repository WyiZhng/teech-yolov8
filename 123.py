import pandas as pd, os
pred = pd.read_csv('/data/HZNU_ZWY/zwy_project/ultralytics-8.1.35/runs/detect/public-2class/v8s_bin_1280_ab2_tail/low_thresh_predictions_icdas_conf0.01/test_predictions.csv')
pred['image_id_b'] = pred['image_id'].astype(str).apply(os.path.basename)
print('images:', pred['image_id_b'].nunique(),
      'candidates:', len(pred),
      'avg per image:', len(pred)/max(1,pred['image_id_b'].nunique()))
print(pred[['yolo_score','w','h']].quantile([0.5,0.9,0.95]))
print(pred['w'].quantile([0.5,0.95]))