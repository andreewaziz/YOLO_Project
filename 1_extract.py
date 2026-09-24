import cv2
import os
import glob

def process_video_folder(input_folder, output_folder, frame_rate=30):
    search_path = os.path.join(input_folder, "*.mp4")
    videos = glob.glob(search_path)
    
    if not videos:
        print(f"No videos found in {input_folder}")
        return

    for video_path in videos:
        video_name = os.path.basename(video_path).split('.')[0]
        cap = cv2.VideoCapture(video_path)
        count = 0
        saved_count = 0
        
        print(f"Processing: {video_name}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            if count % frame_rate == 0:
                height, width, _ = frame.shape
                
                # Split the stacked camera views
                top_half = frame[0:height//2, 0:width]
                bot_half = frame[height//2:height, 0:width]
                
                top_filename = os.path.join(output_folder, f"{video_name}_f{saved_count}_top.jpg")
                bot_filename = os.path.join(output_folder, f"{video_name}_f{saved_count}_bot.jpg")
                
                cv2.imwrite(top_filename, top_half)
                cv2.imwrite(bot_filename, bot_half)
                saved_count += 1
                
            count += 1
        cap.release()
        print(f"  -> Saved {saved_count * 2} separate images.")

if __name__ == '__main__':
    print("--- Extracting Training Videos ---")
    process_video_folder(r"raw_videos\train", r"dataset\images\train", frame_rate=30)
    
    print("\n--- Extracting Validation Videos ---")
    process_video_folder(r"raw_videos\val", r"dataset\images\val", frame_rate=30)
    
    print("\nExtraction complete! Now you are ready to annotate.")
