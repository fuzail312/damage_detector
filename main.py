from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import cv2
import os
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

def process_video_file(file_path: str) -> str:
    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a folder in the same directory to save processed frames
    video_dir = os.path.splitext(file_path)[0] + "_processed_frames"
    os.makedirs(video_dir, exist_ok=True)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection (example: convert to grayscale)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # If a frame should be skipped, continue (dummy condition here)
        if np.mean(processed_frame) < 50:  # Example condition to skip frame
            continue
        
        # Save the processed frame
        frame_file_path = os.path.join(video_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_file_path, processed_frame)
        frame_count += 1
    
    cap.release()
    
    return video_dir

@app.post("/process_image/")
async def process_image(file_path: str = Form(...)):
    processed_file_path = process_image_file(file_path)
    return FileResponse(processed_file_path, media_type="image/jpeg")

@app.post("/process_video/")
async def process_video(file_path: str = Form(...)):
    processed_folder_path = process_video_file(file_path)
    return {"processed_frames_directory": processed_folder_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
