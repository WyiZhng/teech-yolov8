import warnings
warnings.filterwarnings('ignore')
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

# 使用标准ultralytics库，避免模块导入冲突
from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_iou

def set_random_seed(seed=42):
    """
    设置所有随机种子以确保结果的可重现性
    
    Args:
        seed (int): 随机种子值，默认为42
    """
    # 设置Python内置random模块的种子
    random.seed(seed)
    
    # 设置numpy的随机种子
    np.random.seed(seed)
    
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    
    # 如果使用CUDA，设置CUDA的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 对所有GPU设备设置种子
        
        # 设置CUDA的确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设置环境变量以确保更好的可重现性
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"已设置随机种子为: {seed}")
    print("已启用确定性训练模式，确保结果可重现")

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
    rho_x = torch.pow(pred_ctr_x - target_ctr_x, 2)
    rho_y = torch.pow(pred_ctr_y - target_ctr_y, 2)
    gamma = 2 - angle_cost
    distance_cost = 2 - torch.exp(gamma * rho_x / (4 * torch.pow(target_w, 2) + eps)) - torch.exp(gamma * rho_y / (4 * torch.pow(target_h, 2) + eps))
    
    # 计算形状损失
    omiga_w = torch.abs(pred_w - target_w) / torch.max(pred_w, target_w)
    omiga_h = torch.abs(pred_h - target_h) / torch.max(pred_h, target_h)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
    
    # 计算SIoU损失
    siou = iou - (distance_cost + shape_cost) / 2
    return 1 - siou

class SIoUTrainer:
    """使用SIoU损失函数的训练器"""
    
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self._patch_loss_function()
    
    def _patch_loss_function(self):
        """替换模型的损失函数为SIoU"""
        def custom_bbox_loss(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
            """使用SIoU损失的边界框损失函数"""
            if fg_mask.sum() == 0:
                return torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)
            
            # 获取前景预测和目标
            pred_bboxes_pos = pred_bboxes[fg_mask]
            target_bboxes_pos = target_bboxes[fg_mask]
            
            if len(pred_bboxes_pos) == 0:
                return torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)
            
            # 计算SIoU损失
            loss = siou_loss(pred_bboxes_pos, target_bboxes_pos)
            return loss.mean()
        
        # 替换损失函数
        if hasattr(self.model.model, 'model') and hasattr(self.model.model.model[-1], 'bbox_loss'):
            self.model.model.model[-1].bbox_loss = custom_bbox_loss
        elif hasattr(self.model.model, 'bbox_loss'):
            self.model.model.bbox_loss = custom_bbox_loss
    
    def train(self, **kwargs):
        """开始训练"""
        return self.model.train(**kwargs)

if __name__ == '__main__':
    # 设置随机种子，确保结果可重现
    set_random_seed(seed=42)
    
    # 使用SIoU损失函数的训练器
    trainer = SIoUTrainer('yolov8n.pt')
    
    # 开始训练 - 使用SIoU损失函数
    results = trainer.train(
        data='data.yaml',   # 使用mini-dental数据集
        epochs=200,  # 训练轮次总数（减少轮次用于测试）
        batch=64,  # 批量大小，减小以避免内存问题
        imgsz=640,  # 训练图像尺寸
        workers=0,  # 设置为0避免多进程问题
        device=0,  # 指定训练的计算设备，无nvidia显卡则改为 'cpu'
        optimizer='Adam',  # 训练使用优化器，可选 auto,SGD,Adam,AdamW 等
        amp=True,  # True 或者 False, 解释为：自动混合精度(AMP) 训练
        cache=False,  # True 在内存中缓存数据集图像，服务器推荐开启
        save_period=10,
        project='runs/train_env_siou',  # 保存训练结果的项目文件夹
        name='yolov8n_siou_dental',  # 训练任务名称
        exist_ok=True,  # 允许覆盖已存在的训练结果
        seed=42  # 添加种子参数到训练配置中
    )
    
    print("\n" + "="*60)
    print("训练完成！使用了SIoU损失函数优化小目标检测")
    print("训练结果保存在: runs/train_env_siou/yolov8n_siou_dental/")
    print("所有随机种子已固定，确保结果可重现")
    print("="*60)