import numpy as np

data = np.load("./sfm_data.npz", allow_pickle=True)

R_all = np.array(data["camera_rotations"], dtype=np.float64)
t_all = np.array(data["camera_translations"], dtype=np.float64)

poses = []

for R, t in zip(R_all, t_all):
    c2w = np.eye(4)
    c2w[:3,:3] = R.T
    c2w[:3,3] = -R.T @ t
    poses.append(c2w)

poses = np.stack(poses)
np.save("poses.npy", poses)

print("Saved", len(poses), "camera poses")
