# Import necessary libraries
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import numpy as np
import cv2
import os
from typing import List, Tuple
from pyngrok import ngrok
import uvicorn
import nest_asyncio

# Apply the nest_asyncio patch to allow nested event loops in Colab
nest_asyncio.apply()

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

detection = Detection(
    model_path='best.onnx',
    classes=['damaged door', 'damaged window', 'damaged headlight', 'damaged mirror', 'dent', 'damaged hood', 'damaged bumper', 'damaged wind shield']
)

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type. Only JPEG and PNG are supported.")

    image_data = await file.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image data.")
    
    results = detection(image)
    output_path = "output_image.jpg"
    
    for box, conf, cls in zip(results['boxes'], results['confidences'], results['classes']):
        left, top, width, height = box
        right, bottom = left + width, top + height
        label = f"{cls} ({conf:.2f}%)"
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, image)
    return FileResponse(output_path, media_type="image/jpeg")

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    if file.content_type not in ["video/mp4"]:
        raise HTTPException(status_code=400, detail="Invalid video type. Only MP4 is supported.")
    
    video_data = await file.read()
    input_path = "input_video.mp4"
    output_path = "output_video.mp4"

    with open(input_path, "wb") as f:
        f.write(video_data)
    
    def process_video(video_path: str, output_video_path: str, frames_per_second: int = 20, detection_delay: int = 30):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        frame_interval = int(frame_rate / frames_per_second)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                results = detection(frame)

                for box, conf, cls in zip(results['boxes'], results['confidences'], results['classes']):
                    left, top, width, height = box
                    right, bottom = left + width, top + height
                    label = f"{cls} ({conf:.2f}%)"
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

    process_video(input_path, output_path)
    return FileResponse(output_path, media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
