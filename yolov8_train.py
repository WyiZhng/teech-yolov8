import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 使用第二块GPU

import random
import warnings

import numpy as np
import torch
from ultralytics import YOLO


def set_seed(seed: int = 42) -> None:
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


warnings.filterwarnings('ignore')
if __name__ == '__main__':
  set_seed(42)
  model = YOLO('runs/detect/v8s_bin_1280_ab2_tail/weights/best.pt')  # model = YOLO('yolov8s.pt')  # 从头开始训练
  results = model.train(
    data='datasets/mini-dental-binary.yaml',  #数据集配置文件的路径
    imgsz=1536, epochs=40, patience=15,
    # 学习率降一档，防止破坏已学到的高召回
    optimizer='SGD', lr0=0.003, lrf=0.003, momentum=0.937, weight_decay=5e-4,
    # —— 关强增广，轻微色彩即可 ——
    mosaic=0.0, mixup=0.0, copy_paste=0.0,
    degrees=0.0, translate=0.05, scale=0.10, shear=0.0, perspective=0.0,
    hsv_h=0.015, hsv_s=0.70, hsv_v=0.40,
    close_mosaic=0,
    batch=-1, amp=True, cache=True, workers=8,
    seed=42, deterministic=True,
    project='runs/detect', name='v8s_bin_1536_ab2'
)
  YOLO('runs/detect/v8s_bin_1536_ab2/weights/best.pt').val(
    data='datasets/mini-dental-binary.yaml', imgsz=1280, save_json=True, plots=True, augment=True
)