from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
import cv2
import numpy as np
import os
from typing import List, Tuple

app = FastAPI()

# Define the Detection class as provided
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

def process_video(video_path: str, output_video_path: str, detection_delay: int = 30):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    frames_per_second = 20
    frame_interval = int(frame_rate / frames_per_second)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            results = detection(frame)
            if results['boxes']:
                for box, conf, cls in zip(results['boxes'], results['confidences'], results['classes']):
                    left, top, width, height = box
                    right, bottom = left + width, top + height
                    label = f"{cls} ({conf:.2f}%)"

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                for _ in range(detection_delay):
                    out.write(frame)
            else:
                out.write(frame)
        else:
            out.write(frame)

        frame_number += 1

    cap.release()
    out.release()

@app.post("/process-image/")
async def process_image_endpoint(file: UploadFile = File(...)):
    image_data = await file.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    processed_image = process_image(image)

    _, buffer = cv2.imencode('.jpg', processed_image)
    return StreamingResponse(BytesIO(buffer), media_type="image/jpeg")

@app.post("/process-video/")
async def process_video_endpoint(file: UploadFile = File(...), detection_delay: int = 30):
    video_data = await file.read()
    temp_video_path = "temp_video.mp4"

    with open(temp_video_path, "wb") as f:
        f.write(video_data)
    
    output_video_path = "output_video.mp4"
    process_video(temp_video_path, output_video_path, detection_delay)
    
    with open(output_video_path, "rb") as f:
        video_bytes = f.read()

    os.remove(temp_video_path)
    os.remove(output_video_path)
    
    return StreamingResponse(BytesIO(video_bytes), media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
