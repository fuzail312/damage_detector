from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from zipfile import ZipFile
import cv2
import numpy as np
import os
import shutil

app = FastAPI()

class Detection:
    def __init__(self, model_path: str, classes: list):
        self.model_path = model_path
        self.classes = classes
        self.model = self.__load_model()

    def __load_model(self) -> cv2.dnn_Net:
        net = cv2.dnn.readNet(self.model_path)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def __extract_output(self, preds: np.ndarray, image_shape: tuple, input_shape: tuple, score: float = 0.1, nms: float = 0.0, confidence: float = 0.0) -> dict:
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

# Instantiate your detection model
detection = Detection(
    model_path='best.onnx',
    classes=['damaged door', 'damaged window', 'damaged headlight', 'damaged mirror', 'dent', 'damaged hood', 'damaged bumper', 'damaged wind shield']
)

def process_image(image: np.ndarray, width: int = 640, height: int = 640, score: float = 0.1, nms: float = 0.0, confidence: float = 0.0) -> np.ndarray:
    results = detection(image, width=width, height=height, score=score, nms=nms, confidence=confidence)
    for box, conf, cls in zip(results['boxes'], results['confidences'], results['classes']):
        left, top, width, height = box
        right, bottom = left + width, top + height
        label = f"{cls} ({conf:.2f}%)"

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def overlap_percentage(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    area1 = w1 * h1
    area2 = w2 * h2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    union = area1 + area2 - intersection

    return intersection / union

def is_significantly_different(new_boxes, prev_boxes, threshold=0.01):
    for new_box in new_boxes:
        for prev_box in prev_boxes:
            if overlap_percentage(new_box, prev_box) > threshold:
                return False
    return True

@app.post("/process-video/")
async def process_video_endpoint(file: UploadFile = File(...)):
    video_data = await file.read()
    temp_video_path = "temp_video.mp4"

    with open(temp_video_path, "wb") as f:
        f.write(video_data)
    
    cap = cv2.VideoCapture(temp_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    frames_per_second = 20
    frame_interval = int(frame_rate / frames_per_second)

    output_folder = "output_images"
    os.makedirs(output_folder, exist_ok=True)

    frame_number = 0
    prev_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            results = detection(frame)
            if results['boxes'] and is_significantly_different(results['boxes'], prev_detections):
                prev_detections.extend(results['boxes'])
                for box, conf, cls in zip(results['boxes'], results['confidences'], results['classes']):
                    left, top, width, height = box
                    right, bottom = left + width, top + height
                    label = f"{cls} ({conf:.2f}%)"

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imwrite(f"{output_folder}/frame_{frame_number}.jpg", frame)
        
        frame_number += 1

    cap.release()

    zip_path = "processed_images.zip"
    with ZipFile(zip_path, "w") as zip_file:
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            zip_file.write(file_path, filename)

    shutil.rmtree(output_folder)
    os.remove(temp_video_path)

    return StreamingResponse(BytesIO(open(zip_path, "rb").read()), media_type="application/zip")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
