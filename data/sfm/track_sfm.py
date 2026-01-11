import cv2
import numpy as np
import os

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
FRAME_DIR = "./data/frames"
OUT_PATH = "sfm_data.npz"
MIN_MATCHES = 50
MAX_POINTS = 1000   # IMPORTANT for BA stability

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
image_files = sorted([
    os.path.join(FRAME_DIR, f)
    for f in os.listdir(FRAME_DIR)
    if f.endswith(".png")
])

assert len(image_files) >= 2, "Need at least 2 frames"

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files]
K = np.load("./camera_matrix.npy")

print(f"Loaded {len(images)} frames")

# -------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------
orb = cv2.ORB_create(4000)

keypoints = []
descriptors = []

for img in images:
    kp, des = orb.detectAndCompute(img, None)
    keypoints.append(kp)
    descriptors.append(des)

# -------------------------------------------------
# INITIAL CAMERA POSE (first frame)
# -------------------------------------------------
camera_rotations = [np.eye(3)]
camera_translations = [np.zeros(3)]
used_image_indices = [0]   # CRITICAL: tracks which image each camera corresponds to

points_3d = []
observations_2d = []
obs_cam_idx = []
obs_point_idx = []

point_id = 0

# -------------------------------------------------
# MATCH + TRIANGULATE
# -------------------------------------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

for i in range(len(images) - 1):
    matches = bf.match(descriptors[i], descriptors[i + 1])

    if len(matches) < MIN_MATCHES:
        print(f"Skipping frame pair {i}-{i+1} (few matches)")
        continue

    pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        threshold=1.0
    )

    if E is None:
        print(f"Skipping frame pair {i}-{i+1} (no essential matrix)")
        continue

    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    # -------------------------------------------------
    # CAMERA CHAINING
    # -------------------------------------------------
    R_prev = camera_rotations[-1]
    t_prev = camera_translations[-1]

    R_curr = R @ R_prev
    t_curr = R @ t_prev + t.squeeze()

    # ADD CAMERA **ONCE AND ONLY ONCE**
    camera_rotations.append(R_curr)
    camera_translations.append(t_curr)
    used_image_indices.append(i + 1)

    cam_idx = len(camera_rotations) - 1

    # -------------------------------------------------
    # PROJECTION MATRICES
    # -------------------------------------------------
    P1 = K @ np.hstack((R_prev, t_prev.reshape(3, 1)))
    P2 = K @ np.hstack((R_curr, t_curr.reshape(3, 1)))

    # -------------------------------------------------
    # SAFE TRIANGULATION
    # -------------------------------------------------
    mask = mask.ravel().astype(bool)

    if mask.sum() < 8:
        print(f"Skipping frame pair {i}-{i+1} (too few inliers after RANSAC)")
        continue

    pts1_in = pts1[mask]
    pts2_in = pts2[mask]

    pts1_in = np.asarray(pts1_in, dtype=np.float64).T  # (2, N)
    pts2_in = np.asarray(pts2_in, dtype=np.float64).T  # (2, N)

    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)

    if pts1_in.shape[1] < 8:
        continue

    pts4d = cv2.triangulatePoints(P1, P2, pts1_in, pts2_in)
    pts3d = (pts4d[:3] / pts4d[3]).T

    # -------------------------------------------------
    # STORE POINTS + OBSERVATIONS
    # -------------------------------------------------
    for j, p3d in enumerate(pts3d):
        if not np.isfinite(p3d).all():
            continue

        pid = point_id
        points_3d.append(p3d)

        # observation in previous camera
        observations_2d.append(pts1_in.T[j])
        obs_cam_idx.append(cam_idx - 1)
        obs_point_idx.append(pid)

        # observation in current camera
        observations_2d.append(pts2_in.T[j])
        obs_cam_idx.append(cam_idx)
        obs_point_idx.append(pid)

        point_id += 1

# -------------------------------------------------
# CONVERT TO ARRAYS
# -------------------------------------------------
points_3d = np.asarray(points_3d)
observations_2d = np.asarray(observations_2d)
obs_cam_idx = np.asarray(obs_cam_idx)
obs_point_idx = np.asarray(obs_point_idx)

camera_rotations = np.asarray(camera_rotations)
camera_translations = np.asarray(camera_translations)
used_image_indices = np.asarray(used_image_indices)

# -------------------------------------------------
# LIMIT POINT COUNT (SAFE)
# -------------------------------------------------
if len(points_3d) > MAX_POINTS:
    keep = np.random.choice(len(points_3d), MAX_POINTS, replace=False)

    obs_mask = np.isin(obs_point_idx, keep)

    points_3d = points_3d[keep]
    observations_2d = observations_2d[obs_mask]
    obs_cam_idx = obs_cam_idx[obs_mask]
    obs_point_idx = obs_point_idx[obs_mask]

    # remap point ids
    remap = {old: new for new, old in enumerate(keep)}
    obs_point_idx = np.array([remap[i] for i in obs_point_idx])

# -------------------------------------------------
# CAMERA INDEX REMAP (CRITICAL)
# -------------------------------------------------
used_cams = np.unique(obs_cam_idx)

cam_remap = {old: new for new, old in enumerate(used_cams)}
obs_cam_idx = np.array([cam_remap[i] for i in obs_cam_idx])

camera_rotations = camera_rotations[used_cams]
camera_translations = camera_translations[used_cams]
used_image_indices = used_image_indices[used_cams]

# -------------------------------------------------
# SAVE SfM DATA
# -------------------------------------------------
np.savez(
    OUT_PATH,
    points_3d=points_3d,
    camera_rotations=camera_rotations,
    camera_translations=camera_translations,
    observations_2d=observations_2d,
    obs_cam_idx=obs_cam_idx,
    obs_point_idx=obs_point_idx,
    used_image_indices=used_image_indices
)

print("Saved sfm/sfm_data.npz")
print(f"Cameras: {len(camera_rotations)}")
print(f"Points: {len(points_3d)}")
print(f"Observations: {len(observations_2d)}")
