from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import cv2
import os
import shutil
import numpy as np

app = FastAPI()

def process_image_file(file_path: str) -> str:
    # Load the image
    image = cv2.imread(file_path)
    
    # Perform image processing (example: convert to grayscale)
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Save the processed image in the same directory
    processed_file_path = os.path.splitext(file_path)[0] + "_processed.jpg"
    cv2.imwrite(processed_file_path, processed_image)
    
    return processed_file_path

def process_video_file(file_path: str) -> list:
    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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
    # Save uploaded file to a local path
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the video
    processed_frame_files = process_video_file(file_path)
    
    # Create a directory for processed frames
    processed_frames_dir = os.path.splitext(file_path)[0] + "_processed_frames"
    os.makedirs(processed_frames_dir, exist_ok=True)
    
    # Move processed frames to the directory
    for frame_file in processed_frame_files:
        shutil.move(frame_file, os.path.join(processed_frames_dir, os.path.basename(frame_file)))
    
    # Return the path to the directory containing processed frames
    return {"processed_frames_directory": processed_frames_dir}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
