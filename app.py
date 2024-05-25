import gradio as gr
import cv2
import requests
import os
import sys
import torch
import io
from collections import OrderedDict
import numpy as np
from ultralytics import YOLO
import RRDBNet_arch as arch

# Google Drive file IDs
url_ESRGAN = 'https://drive.google.com/uc?id=1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene'
url_PSNR = 'https://drive.google.com/uc?id=1pJ_T-V1dpb1ewoEra1TGSWl5e6H7M4NN'

# Function to download files from Google Drive
def download_file_from_google_drive(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    return response.content

# Download the model files from Google Drive
net_PSNR_bytes = download_file_from_google_drive(url_PSNR)
net_ESRGAN_bytes = download_file_from_google_drive(url_ESRGAN)

# Load the models from bytes
net_PSNR = torch.load(io.BytesIO(net_PSNR_bytes), map_location=torch.device('cpu'))
net_ESRGAN = torch.load(io.BytesIO(net_ESRGAN_bytes), map_location=torch.device('cpu'))

# Set device to CUDA if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the YOLOv8 model
model_yolo = YOLO("yolov8n-face.pt")  # Replace with your model path

def detecte_enhance_faces(image, alpha):
    net_interp = OrderedDict()

    for k, v_PSNR in net_PSNR.items():
        v_ESRGAN = net_ESRGAN[k]
        net_interp[k] = (1 - alpha) * v_PSNR + alpha * v_ESRGAN

    # Initialize and load the model
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(net_interp, strict=True)
    model.eval()
    model_esrgan = model.to(device)

    # Convert the input image from RGB to BGR (OpenCV format)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Perform object detection
    results = model_yolo.predict(source=image, stream=True)

    # List to hold detected face images
    enhanced_face_images = []

    # Iterate through the detected objects
    for result in results:
        boxes = result.boxes  # Bounding boxes for detected objects
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]  # Box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extract the RoI (Region of Interest) from the image
            img = image[y1:y2, x1:x2]

            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)

            with torch.no_grad():
                output = model_esrgan(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round()

            # Convert the RoI back to RGB format
            roi = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            # Add the detected face to the list
            enhanced_face_images.append(roi)

    return enhanced_face_images

# Define the Gradio interface
iface = gr.Interface(
    fn=detecte_enhance_faces,
    inputs=[
        gr.Image(type="numpy", label="Upload an Image"),
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Alpha")
    ],
    outputs=gr.Gallery(label="Detected Faces"),
    title="YOLOv8 Face Detection",
    description="Upload an image and the app will detect and return the faces in it using YOLOv8."
)

# Launch the Gradio app
iface.launch()
