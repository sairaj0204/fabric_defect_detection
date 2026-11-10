from ultralytics import YOLO
import torch

def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="data.yaml", 
        epochs=50,
        imgsz=640,
        batch=16,   
        device=0,     
        workers=4,    
        name="fabric_defect_run"
    )

if __name__ == '__main__':
    main()