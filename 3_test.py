from ultralytics import YOLO

if __name__ == '__main__':
    # 1. بنحمل "المخ" الجديد اللي إنت لسه مدربه حالا!
    # لاحظ إننا استخدمنا مسار الـ weights اللي لسه طالع من التدريب
    model = YOLO(r'runs\detect\train\weights\best.pt')

    print("🚀 Starting Video Tracking...")

    # 2. تشغيل الموديل على الفيديو مع ميزة التتبع (Tracking)
    results = model.track(
        source='test_video.mp4',  # ده اسم الفيديو اللي إنت حطيته في الفولدر
        show=True,                # هيفتح لك شاشة يعرضلك الفيديو وهو بيتحلل
        save=True,                # هيحفظ نسخة من الفيديو بالمربعات
        conf=0.10,    
        iou=0.5,            # هيرسم مربع لو متأكد بنسبة 25% فما فوق
        tracker="botsort.yaml"    # خوارزمية التتبع عشان المربع يلزق في الحيوان وهو بيجري
    )

    print("✅ Testing finished! Check the 'runs/detect/track' folder.")