import gradio as gr
import cv2
import torch
from collections import OrderedDict
import numpy as np
from ultralytics import YOLO
import RRDBNet_arch as arch

net_PSNR_path = './models/RRDB_PSNR_x4.pth'
net_ESRGAN_path = './models/RRDB_ESRGAN_x4.pth'

net_PSNR = torch.load(net_PSNR_path)
net_ESRGAN = torch.load(net_ESRGAN_path)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_yolo = YOLO("./models/yolov8n-face.pt") 

def detecte_enhance_faces(image, alpha):
    net_interp = OrderedDict()

    for k, v_PSNR in net_PSNR.items():
        v_ESRGAN = net_ESRGAN[k]
        net_interp[k] = (1 - alpha) * v_PSNR + alpha * v_ESRGAN

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(net_interp, strict=True)
    model.eval()
    model_esrgan = model.to(device)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model_yolo.predict(source=image, stream=True)

    enhanced_face_images = []

    for result in results:
        boxes = result.boxes  
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            img = image[y1:y2, x1:x2]
            img = img * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)

            with torch.no_grad():
                output = model_esrgan(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)

            roi = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            enhanced_face_images.append(roi)

    return enhanced_face_images

iface = gr.Interface(
    fn=detecte_enhance_faces,
    inputs=[
        gr.Image(type="numpy", label="Upload an Image"),
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Alpha")
    ],
    outputs=gr.Gallery(label="Detected Faces"),
    title="Enhanced face detection (ESRGAN & YOLOv8)",
    description="To use the app, simply upload your image (jpeg, jpg or png). Please click submit only once",
    flagging_options=None 
)

iface.launch()
