import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=15):
    # إنشاء الفولدر لو مش موجود
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0

    print(f"🚀 Starting extraction from {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # بنسيف صورة كل (frame_rate) فريم عشان الصور متبقاش متكررة أوي
        if count % frame_rate == 0:
            frame_name = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
            saved_count += 1
            print(f"📸 Saved: {frame_name}")

        count += 1

    cap.release()
    print(f"✅ Finished! Extracted {saved_count} images to '{output_folder}'")

if __name__ == "__main__":
    # اتأكد إن ده اسم الفيديو بتاعك
    extract_frames('test_video.mp4', 'new_training_data', frame_rate=15)