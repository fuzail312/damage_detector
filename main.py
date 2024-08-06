from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
import shutil
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class Detection:
    # ... (rest of the Detection class code)

def save_uploaded_file(upload_file: UploadFile, output_folder: str) -> str:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_path = os.path.join(output_folder, upload_file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

@app.post("/process-image/")
async def process_image(image: UploadFile = File(...)):
    try:
        output_folder = "detected_images"
        image_path = save_uploaded_file(image, output_folder)

        detection = Detection(
            model_path='best.onnx',
            classes=['damaged door', 'damaged window', 'damaged headlight', 'damaged mirror', 'dent', 'damaged hood', 'damaged bumper', 'damaged wind shield']
        )

        image_np = cv2.imread(image_path)
        if image_np is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        results = detection(image_np)

        detected_boxes = results['boxes']

        if detected_boxes:
            image_filename = os.path.basename(image_path)
            output_image_path = os.path.join(output_folder, image_filename)

            for box, conf, cls in zip(detected_boxes, results['confidences'], results['classes']):
                left, top, width, height = box
                right, bottom = left + width, top + height
                label = f"{cls} ({conf:.2f}%)"

                cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(image_np, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imwrite(output_image_path, image_np)

        return JSONResponse(content={"message": "Image processed successfully", "output_image": output_image_path})
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-video/")
async def process_video(video: UploadFile = File(...), frames_per_second: int = -1):
    try:
        output_folder = "detected_images"
        video_path = save_uploaded_file(video, output_folder)

        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video file not found")

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        if frames_per_second == -1:
            frame_interval = 1
        else:
            frame_interval = int(frame_rate / frames_per_second)

        detection = Detection(
            model_path='best.onnx',
            classes=['damaged door', 'damaged window', 'damaged headlight', 'damaged mirror', 'dent', 'damaged hood', 'damaged bumper', 'damaged wind shield']
        )

        frame_number = 0
        processed_boxes = []
        saved_images = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frames_per_second == -1 or frame_number % frame_interval == 0:
                results = detection(frame)
                detected_boxes = results['boxes']

                new_boxes = []
                for box in detected_boxes:
                    overlap_found = False
                    for prev_box in processed_boxes:
                        iou = compute_iou(box, prev_box)
                        if iou > 0.5:
                            overlap_found = True
                            break

                    if not overlap_found:
                        new_boxes.append(box)
                        processed_boxes.append(box)

                if new_boxes:
                    image_path = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")

                    if image_path not in saved_images:
                        for box, conf, cls in zip(new_boxes, results['confidences'], results['classes']):
                            left, top, width, height = box
                            right, bottom = left + width, top + height
                            label = f"{cls} ({conf:.2f}%)"

                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        cv2.imwrite(image_path, frame)
                        saved_images.add(image_path)

            frame_number += 1

        cap.release()
        return JSONResponse(content={"message": "Video processed successfully", "output_folder": output_folder})
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yA - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
