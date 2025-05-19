import numpy as np
import os
from PIL import Image
import pycolmap

# 경로 설정
llff_pose_path = "Flank_Hyundong/images/poses_bounds.npy"
image_dir = "Flank_Hyundong/images"
sparse_model_path = "output/sparse/sparse_0/0"  # COLMAP sparse 경로
output_dir = "Flank_Hyundong"

# 1️⃣ Load poses_bounds.npy
poses_bounds = np.load(llff_pose_path)  # shape: [N, 3, 5]
poses = poses_bounds[:, :, :4]          # [N, 3, 4]
bounds = poses_bounds[:, :, 4]          # [N, 3] → near/far depth

# 2️⃣ Convert to [N, 4, 4]
def convert_to_4x4(poses_3x4):
    poses_4x4 = []
    for pose in poses_3x4:
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :4] = pose
        poses_4x4.append(mat)
    return np.stack(poses_4x4, axis=0)  # [N, 4, 4]

poses_4x4 = convert_to_4x4(poses)

# 3️⃣ Get image resolution
sample_img = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])[0]
img_path = os.path.join(image_dir, sample_img)
img = Image.open(img_path)
W, H = img.size
print(f"📐 Image resolution: {W} x {H}")

# 4️⃣ Get focal length from COLMAP
recon = pycolmap.Reconstruction(sparse_model_path)
camera = next(iter(recon.cameras.values()))
focal = camera.params[0]
print(f"📸 Focal length from COLMAP: {focal}")

# 5️⃣ Save results
np.save(os.path.join(output_dir, "poses.npy"), poses_4x4)
np.save(os.path.join(output_dir, "bounds.npy"), bounds)  # optional
np.save(os.path.join(output_dir, "focal.npy"), np.array([focal], dtype=np.float32))
np.save(os.path.join(output_dir, "hw.npy"), np.array([H, W], dtype=np.int32))

print("✅ Saved NeRF input files: poses.npy, focal.npy, hw.npy")
