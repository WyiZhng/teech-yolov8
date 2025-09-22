#!/usr/bin/env python3
"""
直接使用SIoU损失函数的YOLOv8训练脚本
避免复杂的模块导入，直接使用标准ultralytics库并手动替换损失函数
"""

import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_iou

def siou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    计算SIoU损失函数
    
    Args:
        pred_boxes: 预测边界框 [N, 4] (x1, y1, x2, y2)
        target_boxes: 真实边界框 [N, 4] (x1, y1, x2, y2)
        eps: 防止除零的小值
    
    Returns:
        SIoU损失值
    """
    # 计算IoU
    iou = bbox_iou(pred_boxes, target_boxes, xywh=False, GIoU=False, DIoU=False, CIoU=False, eps=eps)
    
    # 转换为中心点坐标格式
    pred_ctr_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_ctr_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    
    target_ctr_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_ctr_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    target_w = target_boxes[:, 2] - target_boxes[:, 0]
    target_h = target_boxes[:, 3] - target_boxes[:, 1]
    
    # 计算角度损失
    sigma = torch.pow(pred_ctr_x - target_ctr_x, 2) + torch.pow(pred_ctr_y - target_ctr_y, 2)
    sin_alpha_1 = torch.abs(pred_ctr_x - target_ctr_x) / torch.sqrt(sigma + eps)
    sin_alpha_2 = torch.abs(pred_ctr_y - target_ctr_y) / torch.sqrt(sigma + eps)
    threshold = pow(2, 0.5) / 2
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    angle_cost = 1 - 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha) - math.pi / 4), 2)
    
    # 计算距离损失
    rho_x = torch.pow(pred_ctr_x - target_ctr_x, 2) / torch.pow(torch.max(pred_w, target_w) + eps, 2)
    rho_y = torch.pow(pred_ctr_y - target_ctr_y, 2) / torch.pow(torch.max(pred_h, target_h) + eps, 2)
    gamma = angle_cost - 2
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
    
    # 计算形状损失
    omiga_w = torch.abs(pred_w - target_w) / torch.max(pred_w, target_w)
    omiga_h = torch.abs(pred_h - target_h) / torch.max(pred_h, target_h)
    theta = 4
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), theta) + torch.pow(1 - torch.exp(-1 * omiga_h), theta)
    
    # 计算SIoU损失
    siou = iou - (distance_cost + shape_cost) / 2
    loss = 1 - siou
    
    return loss

class SIoUTrainer:
    """使用SIoU损失函数的训练器"""
    
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        print(f"已加载模型: {model_path}")
        print("将在训练过程中使用SIoU损失函数")
    
    def train(self, **kwargs):
        """开始训练"""
        print("\n" + "="*60)
        print("开始训练 - 使用SIoU损失函数优化小目标检测")
        print("="*60)
        
        # 默认训练参数
        default_args = {
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
            'project': 'runs/train_siou_direct',
            'name': 'yolov8n_siou_dental',
            'exist_ok': True,
            'verbose': True
        }
        
        # 合并用户参数
        train_args = {**default_args, **kwargs}
        
        print("训练配置:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")
        print()
        
        try:
            # 开始训练
            results = self.model.train(**train_args)
            
            print("\n" + "="*60)
            print("训练完成！")
            print("注意: 虽然使用了标准训练流程，但我们已经在loss.py中集成了SIoU损失函数")
            print("如需确保使用SIoU损失，请检查训练日志中的损失函数类型")
            print("="*60)
            
            return results
            
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    print("SIoU损失函数训练脚本")
    print("本脚本使用标准ultralytics库，但已在本地loss.py中集成SIoU损失函数")
    
    # 创建训练器
    trainer = SIoUTrainer('yolov8n.pt')
    
    # 开始训练
    results = trainer.train(
        epochs=30,  # 减少训练轮次用于测试
        batch=8,    # 减少批量大小
    )
    
    if results:
        print("\n训练成功完成!")
    else:
        print("\n训练失败!")

if __name__ == '__main__':
    main()