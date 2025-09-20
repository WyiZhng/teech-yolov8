from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pathlib import Path
import uuid
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List
import base64
from PIL import Image
from io import BytesIO

app = FastAPI(title="YOLOv8 Object Detection API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载模型
model = YOLO("best.pt")  # 或者使用其他模型如 "yolov8n.pt"

# 创建临时目录用于存储上传的图片
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 创建静态文件目录
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head>
            <title>YOLOv8 API</title>
            <meta http-equiv="refresh" content="0;url=/static/index.html">
        </head>
        <body>
            <p>重定向到演示页面...</p>
        </body>
    </html>
    """

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # 生成唯一文件名
    file_extension = file.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{file_extension}"
    file_path = UPLOAD_DIR / file_name
    
    try:
        # 保存上传的文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 运行YOLOv8检测
        results = model(file_path)
        result = results[0]  # 只处理第一张图片的结果
        
        # 处理检测结果
        detections = []
        
        if len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "box": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                })
        
        # 获取带有检测框的图像
        result_img = result.plot()
        
        # 将结果图像转换为base64
        _, buffer = cv2.imencode(f'.{file_extension}', result_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "detections": detections,
            "image_with_boxes": img_base64
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"处理图片时出错: {str(e)}"}
        )
    finally:
        # 删除临时文件
        if file_path.exists():
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300) 