import os
import shutil
import pathlib
import concurrent.futures
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np

# 📌 경로 설정
video_path = "megu_video_2503192338.mp4"  # 🎥 비디오 파일
image_dir = pathlib.Path("images")  # 🎞️ 원본 이미지 저장 폴더
small_image_dir = pathlib.Path("images_small")  # 📏 크기 축소 이미지 저장 폴더

def setup_folders():
    """폴더 초기화"""
    for folder in [image_dir, small_image_dir]:
        if folder.exists():
            shutil.rmtree(folder)
            print(f"🗑️ Removed existing folder: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created folder: {folder}")

def save_video2images(video_path, output_folder="images", target_frames=99):
    """비디오를 프레임으로 변환하여 output_folder에 저장"""
    i = 0
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, total_frames // target_frames)

    print(f"Total frames in video: {total_frames}")
    print(f"Saving every {frame_skip} frames to get approximately {target_frames} images.")

    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            save_path = os.path.join(output_folder, f"image{i:04d}.png")
            cv2.imwrite(save_path, frame)
            print(f"🖼 Saved: {save_path}")
            i += 1

        frame_count += 1

    cap.release()
    print(f"✅ Images saved: {i}")
    print("✅ Video to image conversion completed.")

def video2array(video_path, target_frames=150):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return np.array([])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, total_frames // target_frames)

    print(f"Total frames in video: {total_frames}")
    print(f"Saving every {frame_skip} frames to get approximately {target_frames} images.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()

    frames_array = np.array(frames, dtype=np.uint8)
    print("Frames shape:", frames_array.shape)

    return frames_array

def resize_image(img_file):
    """이미지를 128x128 크기로 리사이즈하여 저장"""
    try:
        input_image = Image.open(str(img_file))
        resized_image = input_image.resize((128, 128), Image.LANCZOS)

        output_path = small_image_dir / img_file.name
        resized_image.save(output_path)
        print(f"📏 Resized and saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error resizing {img_file}: {e}")
        return None

if __name__ == "__main__":
    setup_folders()

    # 1️⃣ 비디오 → 이미지 추출
    save_video2images(video_path, output_folder=str(image_dir))

    # 2️⃣ 이미지 리사이즈 실행
    img_files = list(image_dir.glob("*.png"))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(resize_image, img_files)

    print("✅ All images processed successfully!")
