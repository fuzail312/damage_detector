from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import cv2
import os
import shutil
import numpy as np
import tempfile
import zipfile
from typing import List

app = FastAPI()

def process_image_file(file_path: str) -> str:
    # Load the image
    image = cv2.imread(file_path)
    
    # Example detection: Draw a box around the center of the image
    height, width = image.shape[:2]
    top_left = (width // 4, height // 4)
    bottom_right = (3 * width // 4, 3 * height // 4)
    
    # Draw a rectangle on the image (replace this with your actual detection logic)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    
    # Save the processed image in the same directory
    processed_file_path = os.path.splitext(file_path)[0] + "_processed.jpg"
    cv2.imwrite(processed_file_path, image)
    
    return processed_file_path

def process_video_file(file_path: str) -> List[str]:
    cap = cv2.VideoCapture(file_path)
    
    # Create a folder in the same directory to save processed frames
    video_dir = os.path.splitext(file_path)[0] + "_processed_frames"
    os.makedirs(video_dir, exist_ok=True)
    
    frame_count = 0
    unique_frames = []
    processed_frame_files = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection (example: convert to grayscale)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Skip frames without significant detections
        if np.mean(processed_frame) < 50:  # Example condition to skip frame
            continue
        
        # Check if the frame is unique (example method)
        if not any(np.array_equal(processed_frame, uf) for uf in unique_frames):
            unique_frames.append(processed_frame)
            frame_file_path = os.path.join(video_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_file_path, processed_frame)
            processed_frame_files.append(frame_file_path)
            frame_count += 1
    
    cap.release()
    
    return processed_frame_files

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type. Only JPEG and PNG are supported.")
    
    # Save uploaded file to a local path
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the image
    processed_file_path = process_image_file(file_path)
    return FileResponse(processed_file_path, media_type="image/jpeg")

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    if file.content_type not in ["video/mp4"]:
        raise HTTPException(status_code=400, detail="Invalid video type. Only MP4 is supported.")
    
    # Save uploaded file to a local path
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the video
    processed_frame_files = process_video_file(file_path)
    
    # Create a temporary directory for the zip file
    with tempfile.TemporaryDirectory() as tempdir:
        # Create a zip file
        zip_path = os.path.join(tempdir, "processed_frames.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for frame_file in processed_frame_files:
                zipf.write(frame_file, os.path.basename(frame_file))
        
        return FileResponse(zip_path, media_type="application/zip", filename="processed_frames.zip")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
