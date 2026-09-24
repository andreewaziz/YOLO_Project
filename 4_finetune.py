from ultralytics import YOLO

if __name__ == '__main__':
    # 1. بنحمل المخ اللي إحنا وصلناله (مش بنبدأ من الصفر)
    model = YOLO(r'runs\detect\train\weights\best.pt')

    # 2. بنبدأ التدريب التكميلي
    results = model.train(
        data='dataset.yaml',   
        epochs=30,             # 30 لفة كافيين جداً للداتا الجديدة
        imgsz=640,             
        batch=8,               # خليه 8 عشان ميتعبش الـ CPU
        device='cpu',          
        lr0=0.001,             # بنقلل معدل التعلم عشان ميضيعش اللي حفظه زمان
        augment=True,          # بنشغل ميزة الـ Augmentation عشان يتعود على الإضاءة الليلية
        workers=0
    )