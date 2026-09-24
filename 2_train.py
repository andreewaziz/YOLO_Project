from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolo11n.pt') 

    results = model.train(
        data='dataset.yaml',   
        epochs=50,             
        imgsz=640,             
        batch=8,               
        device='cpu',            
        workers=0              
    )
    print("Training with YOLOv11 finished!")