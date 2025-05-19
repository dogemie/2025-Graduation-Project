import numpy as np

pose_path = "poses_bounds.npy"

poses = np.load(pose_path)
print(poses.shape)