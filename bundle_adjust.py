import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from collections import defaultdict

data = np.load("sfm_data.npz", allow_pickle=True)

points_3d = np.array(data["points_3d"], dtype=np.float64)
camera_rotations = np.array(data["camera_rotations"], dtype=np.float64)
camera_translations = np.array(data["camera_translations"], dtype=np.float64)
obs_2d = np.array(data["observations_2d"], dtype=np.float64)
obs_cam_idx = data["obs_cam_idx"]
obs_point_idx = data["obs_point_idx"]

K = np.load("camera_matrix.npy")
num_cams = len(camera_rotations)

print("Loaded:")
print(" Cameras:", num_cams)
print(" Points:", len(points_3d))
print(" Observations:", len(obs_2d))

# ---------- MIN OBSERVATION FILTER ----------
# creates a default dict that automatically initialize missing keys with 0
obs_count = defaultdict(int)
# obs_point_idx contains point IDs for each 2D observation
# increment the count for corresponding 3D point for every observation 
for pid in obs_point_idx:
    obs_count[int(pid)] += 1

# Keeps only points that are observed in at least 2 cameras
# minimum requirement for triangulation consistency
# infinite depth points could dominate if no 'c >= 2'
valid_points = {pid for pid, c in obs_count.items() if c >= 2}

# convert it as sorted numpy arrays
valid_ids = np.array(sorted(valid_points))

# early exit if nothing survives 
if len(valid_ids) == 0:
    print("No multi-view points — skipping BA")
    exit()

# create a boolean mask. True if the observation references a valid 3D point
# prevent obs array misalignment 

# obs_point_idx = [3,3,3,7,7,10]
# valid_ids = [3,7]

# mask = [T,T,T,T,T,F]

mask = np.array([pid in valid_ids for pid in obs_point_idx])

# remove single view observations
obs_2d = obs_2d[mask]
obs_cam_idx = obs_cam_idx[mask]
obs_point_idx = obs_point_idx[mask]

# create mapping 
# prevent index error or wrong geometry 
id_map = {pid: i for i, pid in enumerate(valid_ids)}

# keeps only valid 3d points
# shape: (num_valid_points, 3)
points_3d = points_3d[valid_ids]

# rewrite observation indices to numpy arrays 
obs_point_idx = np.array([id_map[pid] for pid in obs_point_idx])

# ---------- PROJECTION ----------
# projects 3D -> 2D using pinhole camera 
def project(points, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    pts_cam = (R @ points.T + tvec.reshape(3,1)).T
    pts_img = (K @ pts_cam.T).T
    return pts_img[:, :2] / pts_img[:, 2:3]

# ---------- RESIDUAL ----------
# computes reprojection error: predicted image point - observed image point 
def residuals(params):
    cam_params = params[:6*num_cams]
    pts3d = params[6*num_cams:].reshape(-1,3)

    res = []
    for cam_i, pt_i, obs in zip(obs_cam_idx, obs_point_idx, obs_2d):
        rvec = cam_params[6*cam_i:6*cam_i+3]
        tvec = cam_params[6*cam_i+3:6*cam_i+6]
        proj = project(pts3d[pt_i:pt_i+1], rvec, tvec)
        res.extend(proj[0] - obs)
    return np.array(res)

# ---------- INITIAL GUESS ----------
# store all camera parameters
cam_init = []
# camera rotation: 3x3 matrices, camera translation: (3, ) translation vector 
for R, t in zip(camera_rotations, camera_translations):
    # convert rotation matrix to rotation vector 
    rvec, _ = cv2.Rodrigues(R)
    # flatten and store rotation. converts (3, 1) -> (3, ): 1-dimensional NumPy array with 3 elements
    cam_init.extend(rvec.flatten())
    # flatten and store translation. flattens to (3, )
    cam_init.extend(t.flatten())
# stack camera parameters and 3D points into one vector 
x0 = np.hstack([cam_init, points_3d.flatten()])

print("Running Bundle Adjustment...")

res = least_squares(
    residuals,
    x0,
    verbose=2,
    max_nfev=40,        # HARD STOP (MANDATORY)
    ftol=1e-4,
    xtol=1e-4,
    gtol=1e-4
)

optimized_points = res.x[6*num_cams:].reshape(-1,3)

# ---------- VISUALIZATION ----------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    optimized_points[:,0],
    optimized_points[:,1],
    optimized_points[:,2],
    s=5
)

cam_centers = -camera_rotations.transpose(0,2,1) @ camera_translations[:,:,None]
cam_centers = cam_centers.squeeze()

ax.plot(cam_centers[:,0], cam_centers[:,1], cam_centers[:,2], c="r")

ax.set_title("Day 5: Track-Based SfM + BA")
plt.show()

