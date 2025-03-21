import os
import shutil
import pathlib
import utils.video
import concurrent.futures
from rembg import remove
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2

# 📌 파일 경로 설정
video_path = "megu_video_2503192338.mp4"  # 🎥 비디오 파일
image_dir = pathlib.Path("images_origin")  # 🎞️ 원본 이미지 저장 폴더
bg_removed_dir = pathlib.Path("images_no_bg")  # 🎨 배경 제거 후 저장 폴더
size_down_dir = pathlib.Path("images")  # 📏 최종 크기 조정된 이미지 저장 폴더
sd_nerf_data_dir = pathlib.Path("nerf_data/images")

# 📌 폴더 정리 및 생성
def setup_folders():
    for folder in [image_dir, bg_removed_dir, size_down_dir, sd_nerf_data_dir]:
        if folder.exists():
            shutil.rmtree(folder)
            print(f"🗑️ Removed existing folder: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created folder: {folder}")

# 📌 비디오 → 이미지 변환
def extract_frames():
    """ 비디오를 프레임별로 저장 """
    utils.video.save_video2images(video_path)
    
    # ✅ 저장된 이미지 확인
    extracted_files = list(image_dir.glob("*.png"))
    print(f"🔍 Extracted {len(extracted_files)} images to {image_dir}")

    if len(extracted_files) == 0:
        print("❌ No images extracted! Check the video file or `utils.video.save_video2images` function.")
        exit()

# 🔹 **경계선 강조 (특이점 강조)**
def enhance_edges(image):
    """ 경계를 강조하여 특이점을 뚜렷하게 만드는 함수 """
    edge_image = image.filter(ImageFilter.FIND_EDGES).convert("L")  # 경계 검출 후 흑백 변환
    edge_image = ImageEnhance.Contrast(edge_image).enhance(2.0)  # 대비 증가
    return Image.blend(image.convert("RGB"), edge_image.convert("RGB"), alpha=0.4)  # 블렌딩

# 📌 **배경 제거 함수 (크로마키 적용)**
def remove_background(img_file):
    """ 배경을 제거하고 초록색(크로마키) 배경을 적용한 이미지를 저장 """
    try:
        input_image = Image.open(str(img_file)).convert("RGBA")  # RGBA 모드 변환
        output_image = remove(
            input_image,
            alpha_matting=True,
            alpha_matting_foreground_threshold=150,  # ✅ 고정값
            alpha_matting_background_threshold=0,  # ✅ 고정값
            alpha_matting_erode_size=1,  # ✅ 고정값
            alpha_matting_mask_blur=0  # ✅ 고정값
        )  # 배경 제거

        # 🔹 투명한 부분을 초록색 배경(크로마키)으로 채우기
        chroma_key_bg = Image.new("RGBA", output_image.size, (0, 255, 0, 255))  # 크로마키 녹색 배경
        output_image = Image.alpha_composite(chroma_key_bg, output_image).convert("RGB")

        # 🔹 경계선 강조 적용
        output_image = enhance_edges(output_image)

        # 🔹 배경 제거된 이미지 저장
        output_path = bg_removed_dir / img_file.name
        output_image.save(output_path)
        print(f"🖼️ Background removed and saved: {output_path}")
        return output_path  # ✅ 저장된 이미지 경로 반환
    except Exception as e:
        print(f"❌ Error processing {img_file}: {e}")
        return None  # 오류 발생 시 None 반환

# 📌 **이미지 크기 조정 함수 (128×128)**
def resize_image(img_file):
    """ 이미지를 128x128 크기로 리사이즈하여 저장 """
    try:
        input_image = Image.open(str(img_file))
        resized_image = input_image.resize((128, 128), Image.LANCZOS)

        output_path = size_down_dir / img_file.name
        output_nerf_data = sd_nerf_data_dir / img_file.name
        resized_image.save(output_path)
        resized_image.save(output_nerf_data)
        print(f"📏 Resized and saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error resizing {img_file}: {e}")
        return None

# 📌 메인 실행 함수
if __name__ == "__main__":
    setup_folders()

    # 1️⃣ **비디오 → 이미지 변환**
    extract_frames()

    # ✅ 이미지 리스트 확인
    img_files = list(image_dir.glob("*.png"))

    # 2️⃣ **멀티스레드 배경 제거 및 크기 조정**
    print("🎨 Running background removal and resizing...")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        bg_removed_images = list(executor.map(remove_background, img_files))

    # ✅ None 값 제거 (에러 발생한 이미지 제외)
    bg_removed_images = [img for img in bg_removed_images if img is not None]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(resize_image, bg_removed_images)

    print("✅ All images processed successfully!")
