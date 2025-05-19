import numpy as np
poses = np.load("llff_data.npz", allow_pickle=True)
poses_loaded = poses["focal_lengths"]
print(poses_loaded)