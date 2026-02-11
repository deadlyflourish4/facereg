from ultralytics import YOLO
import io 
from PIL import Image
from fastapi import FastAPI, UploadFile
import time
import cv2
import numpy as np
import math 

app = FastAPI()
model = YOLO('C:\\Users\\anansupercuteeeee\\Music\\git\\facereg\\weights\\yolov11n-face.pt')

def detect_face(image):
    results = model.predict(source=image, save=False, conf=0.5, imgsz=640)[0]
    
    faces = []
    for i in range(len(results.boxes)):
        bbox = results.boxes.xyxy[i].tolist()
        conf = results.boxes.conf[i].item()
        
        if results.keypoints is not None:
            landmarks = result.keypoints.xy[i].tolist()  # [[x,y], [x,y], ...]
        else:
            landmarks = None

        faces.append({
            'bbox': bbox,
            'conf': conf,
            'landmarks': landmarks
        })
    return faces

def read_image(file_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(file_bytes))
    image = image.convert("RGB")
    return image

def check_blur(image, threshold=100):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return {
        "is_blurry": laplacian_var < threshold,
        "laplacian_var": round(laplacian_var, 2)
    } 

def check_brightness(image, low=40, high=220):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    mean_brightness = gray.mean()
    
    if mean_brightness < low:
        status = "too_dark"
    elif mean_brightness > high:
        status = "too_bright"
    else:
        status = "normal"
    
    return {
        "status": status,
        "mean_brightness": round(mean_brightness, 2)
    }

def check_face_size(bbox, min_width=80, min_height=80):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    return {
        "width": round(width, 2),
        "height": round(height, 2),
        "is_too_small": width < min_width or height < min_height
    }

def check_pose(landmarks, max_angle=30):
    eye_left = landmarks[0]
    eye_right = landmarks[1]

    dy = eye_right[1] - eye_left[1]
    dx = eye_right[0] - eye_left[0]

    if dx == 0:
        angle_deg = 90.0
    else:
        angle_deg = abs(math.degrees(math.atan(dy / dx)))
    return {
        "angle": round(angle_deg, 2),
        "is_tilted": angle_deg > max_angle
    }

def check_quality(face_crop, landmarks=None):
    issues = []

    # 1. Kiểm tra blur
    blur_result = check_blur(face_crop)
    if blur_result["is_blurry"]:
        issues.append("too_blurry")

    # 2. Kiểm tra brightness
    brightness_result = check_brightness(face_crop)
    if brightness_result["status"] != "normal":
        issues.append(brightness_result["status"])  # "too_dark" hoặc "too_bright"

    # 3. Kiểm tra face size
    w, h = face_crop.size  # PIL Image có .size trả về (width, height)
    size_result = check_face_size([0, 0, w, h])
    if size_result["is_too_small"]:
        issues.append("face_too_small")

    # 4. Kiểm tra pose (chỉ khi có landmarks)
    pose_result = None
    if landmarks is not None:
        pose_result = check_pose(landmarks)
        if pose_result["is_tilted"]:
            issues.append("extreme_pose")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "scores": {
            "blur": blur_result["laplacian_var"],
            "brightness": brightness_result["mean_brightness"],
            "face_size": {"width": w, "height": h},
            "angle": pose_result["angle"] if pose_result else None
        }
    }

@app.post("/detect")
async def detect(file: UploadFile):
    file_bytes = await file.read()

    image = read_image(file_bytes)
    start = time.time()
    faces = detect_face(image)
    processing_time = time.time() - start

    return {
        "total_faces": len(faces),
        "processing_time": processing_time,
        "faces": faces
    }

@app.post("/quality")
async def quality(file: UploadFile):
    file_bytes = await file.read()
    image = read_image(file_bytes)

    #1. Detect face
    faces = detect_face(image)

    if len(faces) == 0:
        return {
            "total_faces": 0,
            "message": "No detected face",
            "results": []
        }
    
    #2. Check quality
    results = []
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        face_crop = image.crop((x1, y1, x2, y2))
        landmarks = face["landmarks"]

        quality = check_quality(face_crop, landmarks)

        results.append({
            "bbox": face["bbox"],
            "confidence": face["conf"],
            "quality": quality
        })
    
    return {
        "total_faces": len(faces),
        "results": results
    }