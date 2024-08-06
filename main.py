from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
import shutil
from typing import List, Tuple

app = FastAPI()

class Detection:
    def __init__(self, model_path: str, classes: List[str]):
        self.model_path = model_path
        self.classes = classes
        self.model = self.__load_model()

    def __load_model(self) -> cv2.dnn_Net:
        net = cv2.dnn.readNet(self.model_path)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def __extract_output(self, preds: np.ndarray, image_shape: Tuple[int, int], input_shape: Tuple[int, int], score: float = 0.1, nms: float = 0.0, confidence: float = 0.0) -> dict:
        class_ids, confs, boxes = [], [], []

        image_height, image_width = image_shape
        input_height, input_width = input_shape
        x_factor = image_width / input_width
        y_factor = image_height / input_height

        rows = preds[0].shape[0]
        for i in range(rows):
            row = preds[0][i]
            conf = row[4]

            classes_score = row[4:]
            _, _, _, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]

            if classes_score[class_id] > score:
                confs.append(conf)
                label = self.classes[int(class_id)]
                class_ids.append(label)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

        r_class_ids, r_confs, r_boxes = [], [], []
        indexes = cv2.dnn.NMSBoxes(boxes, confs, confidence, nms)
        for i in indexes:
            r_class_ids.append(class_ids[i])
            r_confs.append(confs[i] * 100)
            r_boxes.append(boxes[i].tolist())

        return {
            'boxes': r_boxes,
            'confidences': r_confs,
            'classes': r_class_ids
        }

    def __call__(self, image: np.ndarray, width: int = 640, height: int = 640, score: float = 0.1, nms: float = 0.0, confidence: float = 0.0) -> dict:
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (width, height), swapRB=True, crop=False)
        self.model.setInput(blob)
        preds = self.model.forward()
        preds = preds.transpose((0, 2, 1))

        results = self.__extract_output(preds=preds, image_shape=image.shape[:2], input_shape=(height, width), score=score, nms=nms, confidence=confidence)
        return results

def save_uploaded_file(upload_file: UploadFile, output_folder: str) -> str:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_path = os.path.join(output_folder, upload_file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

@app.post("/process-image/")
async def process_image(image: UploadFile = File(...)):
    output_folder = "detected_images"
    image_path = save_uploaded_file(image, output_folder)

    detection = Detection(
        model_path='best.onnx',
        classes=['damaged door', 'damaged window', 'damaged headlight', 'damaged mirror', 'dent', 'damaged hood', 'damaged bumper', 'damaged wind shield']
    )

    image_np = cv2.imread(image_path)
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

@app.post("/process-video/")
async def process_video(video: UploadFile = File(...), frames_per_second: int = -1):
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

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
