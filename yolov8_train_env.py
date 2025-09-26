import warnings
warnings.filterwarnings('ignore')
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any


def _auto_configure_cuda_visible_devices() -> Optional[str]:
    """在导入torch之前解析nvidia-smi输出，锁定显存最多的GPU。"""
    if os.environ.get('CUDA_VISIBLE_DEVICES') or os.environ.get('FORCE_CPU', '').lower() == 'true':
        return os.environ.get('CUDA_VISIBLE_DEVICES')

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            check=True,
            capture_output=True,
            text=True
        )
        candidates = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            idx_str, mem_str = [item.strip() for item in line.split(',')]
            try:
                free_gb = float(mem_str) / 1024.0
                candidates.append((free_gb, idx_str))
            except ValueError:
                continue

        if not candidates:
            return None

        candidates.sort(reverse=True)
        best_gpu = candidates[0][1]
        os.environ['CUDA_VISIBLE_DEVICES'] = best_gpu
        os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
        print(f"根据 nvidia-smi 自动选择物理 GPU {best_gpu} 作为训练设备")
        return best_gpu
    except Exception as exc:
        print(f"自动选择GPU失败，将在程序内重试. 原因: {exc}")
        return None


_VISIBLE_GPU = _auto_configure_cuda_visible_devices()

import torch
import math
import random
import numpy as np

# 使用标准ultralytics库，避免模块导入冲突
from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_iou


ROOT_DIR = Path(__file__).resolve().parent
MODEL_CONFIG_PATH = ROOT_DIR / 'ultralytics_local' / 'cfg' / 'models' / 'v8' / 'yolov8n-p2-dental.yaml'
PRETRAINED_WEIGHTS_PATH = ROOT_DIR / 'yolov8n.pt'
AREA_EPS = 1e-6
CACHE_MODE = 'disk'


def select_best_device() -> str:
    """选择当前可用显存最多的CUDA设备，若无CUDA则返回CPU。"""
    if not torch.cuda.is_available():
        print("未检测到CUDA设备，使用CPU训练。")
        return 'cpu'

    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    physical = visible.split(',')[0].strip() if visible else _VISIBLE_GPU
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / (1024 ** 3)
    print(f"自动选择GPU (逻辑索引0, 物理索引{physical})，当前可用显存约 {free_gb:.2f} GB")
    return '0'


def safe_train(trainer: 'SIoUTrainer', base_params: Dict[str, Any]):
    """在显存不足时自动降低配置并重试训练。"""
    attempts = 0
    params = base_params.copy()
    min_batch = 4
    min_imgsz = 640
    original_multi_scale = params.get('multi_scale', False)

    while True:
        try:
            return trainer.train(**params)
        except RuntimeError as err:
            message = str(err).lower()
            if 'out of memory' not in message and 'cuda error' not in message:
                raise

            attempts += 1
            torch.cuda.empty_cache()

            print(f"⚠️ 检测到显存不足，正在第 {attempts} 次自动调整参数后重试...")

            current_batch = params.get('batch', min_batch)
            current_imgsz = params.get('imgsz', min_imgsz)

            if current_batch > min_batch:
                new_batch = max(current_batch // 2, min_batch)
                params['batch'] = new_batch
                print(f"  -> 将 batch 大小从 {current_batch} 降至 {new_batch}")
                continue

            if current_imgsz > min_imgsz:
                new_imgsz = max(current_imgsz - 128, min_imgsz)
                params['imgsz'] = new_imgsz
                print(f"  -> 将训练分辨率从 {current_imgsz} 调整为 {new_imgsz}")
                continue

            if params.get('multi_scale', False):
                params['multi_scale'] = False
                print("  -> 关闭 multi_scale 减少峰值显存占用")
                continue

            raise RuntimeError("多次调参后显存仍不足，请手动检查其他占用或减小模型规模。") from err

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

    def __init__(self, model_config_path: Path = MODEL_CONFIG_PATH, pretrained_weights_path: Optional[Path] = PRETRAINED_WEIGHTS_PATH):
        model_config_path = Path(model_config_path)
        if not model_config_path.exists():
            raise FileNotFoundError(f"未找到模型配置文件: {model_config_path}")

        self.model = YOLO(str(model_config_path))

        if pretrained_weights_path is not None:
            pretrained_weights_path = Path(pretrained_weights_path)
            if not pretrained_weights_path.exists():
                raise FileNotFoundError(f"未找到预训练权重文件: {pretrained_weights_path}")
            # 使用官方预训练权重进行迁移学习
            self.model = self.model.load(str(pretrained_weights_path))

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
            
            # 计算SIoU损失并对小目标施加更高权重
            loss = siou_loss(pred_bboxes_pos, target_bboxes_pos)

            target_w = target_bboxes_pos[:, 2] - target_bboxes_pos[:, 0]
            target_h = target_bboxes_pos[:, 3] - target_bboxes_pos[:, 1]
            target_area = (target_w * target_h).clamp(min=AREA_EPS)

            # 目标越小，权重越大；保持均值为1避免整体梯度不稳定
            size_weights = (target_area.sqrt()).reciprocal()
            normalized_weights = size_weights / (size_weights.mean().clamp(min=AREA_EPS))

            weighted_loss = loss * normalized_weights
            return weighted_loss.sum() / (normalized_weights.sum().clamp(min=AREA_EPS))
        
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

    # 根据当前硬件自动选择设备
    device = select_best_device()
    training_device = device
    if device != 'cpu':
        torch.cuda.set_device(0)
        training_device = 0
        physical = (os.environ.get('CUDA_VISIBLE_DEVICES') or (_VISIBLE_GPU or '0')).split(',')[0].strip()
        print(f"已锁定物理 GPU {physical}，训练进程对应逻辑 GPU {training_device}")

    # 使用SIoU损失函数的训练器
    trainer = SIoUTrainer()

    # 针对小目标龋齿的重点改进：
    # 1. 使用包含P2检测头的自定义模型配置，保留更多高分辨率特征。
    # 2. 调整图像分辨率与多尺度增强，提升对微小病灶的感知能力。
    # 3. 在损失函数中根据目标面积自适应加权，减小小目标被忽略的风险。
    training_params = dict(
        data='data.yaml',  # 使用mini-dental数据集
        model=str(MODEL_CONFIG_PATH),
        epochs=80,  # 增加训练轮次以充分收敛
    batch=20,  # RTX 4090 单卡的默认起点，若显存不足自动缩减
    imgsz=832,  # 提升分辨率并适当降低显存峰值
        workers=2,  # 适度开启dataloader线程
    device=training_device,
        optimizer='AdamW',
        lr0=8e-4,
        lrf=0.05,
        warmup_epochs=5,
        patience=40,
        cos_lr=True,
        amp=True,
        cache=CACHE_MODE,
        project='runs/train_env_siou',
        name='yolov8n_siou_dental_p2',
        exist_ok=True,
        seed=42,
        multi_scale=False,
        close_mosaic=8,
        mosaic=0.5,
        mixup=0.15,
        scale=0.4,
        translate=0.05,
        hsv_h=0.005,
        hsv_s=0.5,
        hsv_v=0.2,
        max_det=1500,
        box=10.0,
        cls=0.7,
        dfl=1.6
    )

    # 开始训练 - 使用SIoU损失函数
    results = safe_train(trainer, training_params)

    print("\n" + "=" * 60)
    print("训练完成！使用了小目标友好的SIoU损失与P2输出结构")
    print("训练结果保存在: runs/train_env_siou/yolov8n_siou_dental_p2/")
    print("所有随机种子已固定，确保结果可重现")
    print("=" * 60)