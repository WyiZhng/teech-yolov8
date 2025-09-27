import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
  model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
  model.load('best.pt')  #注释则不加载
  results = model.train(
    data='datasets/mini-dental-binary.yaml',  #数据集配置文件的路径
    epochs=200,  #训练轮次总数
    batch=16,  #批量大小，即单次输入多少图片训练
    seed=42,  #固定随机种子，确保结果可复现
    deterministic=True,  #确保所有可控阶段的确定性行为
    imgsz=640,  #训练图像尺寸
    workers=4,  #加载数据的工作线程数
    device=0,  #指定训练的计算设备，无nvidia显卡则改为 'cpu'
    optimizer='Adam',  #训练使用优化器，可选 auto,SGD,Adam,AdamW 等
    amp= True,  #True 或者 False, 解释为：自动混合精度(AMP) 训练
    cache=False , # True 在内存中缓存数据集图像，服务器推荐开启
    save_period = 30
)