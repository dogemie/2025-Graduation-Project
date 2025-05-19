import os
import glob
import math
import numpy as np
from PIL import Image

def compute_focal_from_image(image_path, fov_deg=60):
    """
    이미지 파일의 가로 길이와 주어진 FOV를 기반으로 focal length를 계산합니다.
    focal = 0.5 * width / tan(0.5 * FOV)
    """
    im = Image.open(image_path)
    width, height = im.size
    focal = 0.5 * width / math.tan(0.5 * math.radians(fov_deg))
    return focal, width, height

def pose_spherical(theta, phi, radius):
    """
    주어진 spherical 좌표(θ, φ, 반경)를 바탕으로 카메라-투-월드(c2w) 4x4 pose 행렬을 생성합니다.
    
    theta: y축 기준 회전 (deg)
    phi: x축 기준 회전 (deg)
    radius: 원점으로부터의 거리
    """
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    
    # z축 방향으로 radius만큼 translation
    trans_t = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, radius],
        [0, 0, 0, 1]
    ])
    # x축 회전 (phi)
    rot_phi = np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi_rad), -np.sin(phi_rad), 0],
        [0, np.sin(phi_rad),  np.cos(phi_rad), 0],
        [0, 0, 0, 1]
    ])
    # y축 회전 (theta)
    rot_theta = np.array([
        [np.cos(theta_rad), 0, -np.sin(theta_rad), 0],
        [0, 1, 0, 0],
        [np.sin(theta_rad), 0,  np.cos(theta_rad), 0],
        [0, 0, 0, 1]
    ])
    # 카메라-투-월드 변환
    c2w = rot_theta @ rot_phi @ trans_t
    
    # LLFF/NeRF에서 사용하는 좌표계를 맞추기 위한 고정 변환 (y, z 반전)
    fix = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    c2w = fix @ c2w
    return c2w

def main(image_dir, output_filename="llff_data.npz", fov_deg=60, phi=-30, radius=4.0):
    """
    이미지 폴더 내의 이미지를 읽고,  
      - 첫 번째 이미지에서 focal을 계산 (모든 이미지에 동일)
      - 각 이미지에 대해 4x4 카메라-투-월드 pose를 생성한 후  
    이 정보를 npz 파일로 저장합니다.
    
    저장되는 npz 파일의 키는 아래와 같습니다:
      - images: (N, H, W, C) numpy 배열, N개의 이미지
      - poses: (N, 4, 4) numpy 배열, 각 이미지에 대응하는 pose 행렬
      - focal: float, 첫 번째 이미지에서 계산한 focal 값
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if len(image_paths) == 0:
        raise ValueError("지정된 폴더에서 이미지를 찾을 수 없습니다.")
    
    images_list = []
    poses_list = []
    num_images = len(image_paths)
    
    # 첫 번째 이미지로부터 focal을 계산 (모든 이미지에 동일하다고 가정)
    focal, width, height = compute_focal_from_image(image_paths[0], fov_deg)
    
    for i, img_path in enumerate(image_paths):
        # 이미지 로드 및 numpy array 변환
        im = Image.open(img_path)
        im_np = np.array(im)
        images_list.append(im_np)
        
        # 이미지별로 균등하게 분포하는 각도 계산 (원형 배치)
        theta = 360.0 * i / num_images
        c2w = pose_spherical(theta, phi, radius)
        poses_list.append(c2w)
    
    images_arr = np.stack(images_list, axis=0)   # (N, H, W, C)
    poses_arr = np.stack(poses_list, axis=0)       # (N, 4, 4)
    
    # npz 파일에 저장: images, poses, focal (focal은 단일 값)
    np.savez(output_filename, images=images_arr, poses=poses_arr, focal=focal)
    print(f"Saved npz file: {output_filename}")

if __name__ == "__main__":
    image_dir = "./images_small"  # 이미지 폴더 경로 (원하는 폴더로 수정)
    main(image_dir)
