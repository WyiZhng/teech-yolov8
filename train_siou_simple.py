#!/usr/bin/env python3
"""
简化的YOLOv8训练脚本，使用SIoU损失函数
直接使用本地修改的YOLO实现，避免模块导入冲突
"""

import warnings
warnings.filterwarnings('ignore')
import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# 添加本地路径
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# 直接导入需要的模块
from ultralytics_local.models.yolo.detect import DetectionTrainer
from ultralytics_local.models.yolo.model import YOLO as BaseYOLO
from ultralytics_local.utils.loss import v8DetectionLoss
from ultralytics_local.cfg import get_cfg

class SIoUYOLO(BaseYOLO):
    """使用SIoU损失函数的YOLO模型"""
    
    def __init__(self, model='yolov8n.pt', task=None, verbose=True):
        super().__init__(model=model, task=task, verbose=verbose)
    
    def train(self, **kwargs):
        """训练模型，强制使用SIoU损失函数"""
        # 设置box_loss参数
        kwargs['box_loss'] = 'siou'
        
        # 调用父类的训练方法
        return super().train(**kwargs)

def main():
    """主训练函数"""
    print("="*60)
    print("开始使用SIoU损失函数训练YOLOv8模型")
    print("="*60)
    
    try:
        # 创建模型实例
        model = SIoUYOLO('yolov8n.pt')
        
        # 训练配置
        train_args = {
            'data': 'ultralytics_local/cfg/datasets/mini-dental.yaml',
            'epochs': 50,
            'batch': 16,
            'imgsz': 640,
            'workers': 4,
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'optimizer': 'Adam',
            'amp': True,
            'cache': False,
            'save_period': 10,
            'project': 'runs/train_siou',
            'name': 'yolov8n_siou_dental',
            'exist_ok': True,
            'verbose': True,
            'box_loss': 'siou'  # 明确指定使用SIoU损失函数
        }
        
        print(f"训练配置:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")
        print()
        
        # 开始训练
        results = model.train(**train_args)
        
        print("\n" + "="*60)
        print("训练完成！使用了SIoU损失函数优化小目标检测")
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()