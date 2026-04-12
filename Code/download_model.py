import urllib.request
import os

# Try multiple sources
urls = [
    "https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt",
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt",
]

filename = "yolov8n-face.pt"

if not os.path.exists(filename):
    print("Downloading YOLOv8 face model...")
    for url in urls:
        try:
            print(f"Trying: {url}")
            urllib.request.urlretrieve(url, filename)
            print("Download complete!")
            break
        except Exception as e:
            print(f"Failed: {e}")
            continue
else:
    print("Model already exists!")