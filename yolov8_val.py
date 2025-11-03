from ultralytics import YOLO

# 高精度/部署用：1280_tail
m = YOLO('runs/detect/v8s_bin_1280_ab2_tail/weights/best.pt')  # 或 last.pt
m.train(data='datasets/mini-dental-binary+hardneg.yaml',
        epochs=10, imgsz=1280, batch=-1,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=3,
        cls=1.0, close_mosaic=0, amp=True, cache=True)