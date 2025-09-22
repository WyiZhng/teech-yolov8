import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 使用预训练的YOLOv8n模型
    model = YOLO('yolov8n.pt')
    
    # 开始训练 - 使用WIoU v3损失函数（专门为小目标优化）
    results = model.train(
        data='ultralytics_local/cfg/datasets/mini-dental.yaml',  # 使用mini-dental数据集
        epochs=100,  # 训练轮次总数
        batch=32,  # 批量大小，即单次输入多少图片训练
        imgsz=640,  # 训练图像尺寸
        workers=8,  # 加载数据的工作线程数
        device=0,  # 指定训练的计算设备，无nvidia显卡则改为 'cpu'
        optimizer='Adam',  # 训练使用优化器，可选 auto,SGD,Adam,AdamW 等
        amp=True,  # True 或者 False, 解释为：自动混合精度(AMP) 训练
        cache=False,  # True 在内存中缓存数据集图像，服务器推荐开启
        save_period=30,
        box_loss='wiou3',  # 使用WIoU v3损失函数，专门优化小目标检测
        project='runs/train_wiou3',  # 保存训练结果的项目文件夹
        name='yolov8n_wiou3_smallbox',  # 训练任务名称
        exist_ok=True  # 允许覆盖已存在的训练结果
    )
    
    print("\n" + "="*60)
    print("训练完成！使用了WIoU v3损失函数优化小目标检测")
    print("="*60)