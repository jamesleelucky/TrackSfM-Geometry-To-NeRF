# import numpy as np
# import json
# import os
# import cv2

# # -------------------------------
# # LOAD DATA
# # -------------------------------
# poses = np.load("poses.npy")          # shape: (N, 4, 4)
# fx, fy, cx, cy = np.load("intrinsics.npy")

# image_dir = "./data/frames"  # folder containing captured frames. contains RGB frames used for NeRF
# # ensures images align exactly with pose order 
# image_files = sorted([
#     f for f in os.listdir(image_dir)
#     if f.endswith((".png", ".jpg", ".jpeg"))
# ])

# # print("Num images:", len(image_files))
# # print("Num poses:", len(poses))
# # print(image_files[:5])
# # print(len(image_files))

# # safety check. every image must have a corresponding pose. 
# assert len(image_files) == len(poses), "Image / pose count mismatch"

# # -------------------------------
# # EXTRACT INTRINSICS
# # -------------------------------
# # H: image height (pixel), W: image width (pixel). Loads first image
# # need for ray generation 
# H, W = cv2.imread(os.path.join(image_dir, image_files[0])).shape[:2]

# # K = | fx   0   cx |
# #     |  0  fy   cy |
# #     |  0   0    1 |

# # fx = K[0, 0]
# # fy = K[1, 1]
# # cx = K[0, 2]
# # cy = K[1, 2]

# # Computes horizontal field of view
# # derived from pinhole camera geometry 
# camera_angle_x = 2 * np.arctan(W / (2 * fx))

# # -------------------------------
# # CONVERT POSES TO NeRF FORMAT
# # OpenCV → NeRF coordinate fix
# # -------------------------------
# # Flipping Y and Z aligns your SfM poses with NeRF’s expected convention.
# def convert_pose(pose):
#     pose = pose.copy()
#     # mathematically: diag(1, -1, -1)
#     # Y is inverted because images grow downward
#     # Z is inverted because OpenCV looks forward, NeRF looks backward
#     pose[:3, 1] *= -1   # flip Y
#     pose[:3, 2] *= -1   # flip Z
#     return pose

# # applies the conversion to every camera pose
# poses = np.array([convert_pose(p) for p in poses])

# # -------------------------------
# # NORMALIZE SCALE & CENTER
# # -------------------------------
# # poses[a_slice, b_slice, c_slice]: a_slice = pose index (which camera poses will you take?), b_slice: which rows? c_slice: which columns? 
# cam_centers = poses[:, :3, 3]

# # computes the average camera location
# center = cam_centers.mean(axis=0)

# # recenters the entire scene around the origin
# poses[:, :3, 3] -= center

# # scale normalization of camera positions

# # computes the Euclidean length (camera distance) of each (x,y,z) vector and then find its maximum 
# scale = np.max(np.linalg.norm(poses[:, :3, 3], axis=1))
# # divides every camera center by the same value. normalize all camera positions 
# poses[:, :3, 3] /= scale

# print("Poses normalized")

# # -------------------------------
# # WRITE transforms.json
# # -------------------------------
# # container of NeRF frame data 
# frames = []

# # records image file path and camera-to-world matrix: core NeRF dataset structure 
# for i, fname in enumerate(image_files):
#     frames.append({
#         "file_path": f"./data/frames/{fname}",
#         "transform_matrix": poses[i].tolist()
#     })

# # Tells NeRF: Camera intrinsics, Image resolution, Camera poses
# transforms = {
#     "camera_angle_x": camera_angle_x,
#     "fl_x": fx,
#     "fl_y": fy,
#     "cx": cx,
#     "cy": cy,
#     "w": W,
#     "h": H,
#     "frames": frames
# }

# with open("transforms.json", "w") as f:
#     json.dump(transforms, f, indent=4)

# print("Saved transforms.json")

# # -------------------------------
# # RAY GENERATION (CORE NeRF STEP)
# # -------------------------------
# # converts a camera pose 
# def generate_rays(pose, H, W):
#     # creates pixel coordinate grids. i: x-coordinates, j: y-coordinates
#     # one (i, j) per pixel 
#     i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
#     # converts pixels to camera-space ray direction 
#     dirs = np.stack([
#         (i - cx) / fx,
#         -(j - cy) / fy,
#         -np.ones_like(i)
#     ], axis=-1) # rays pointing forward

#     # extracts rotation and translation
#     # converts rays from camera to world space 
#     R = pose[:3, :3]
#     t = pose[:3, 3]

#     # rotates all ray direction into world coordinates 
#     rays_d = (dirs @ R.T)
#     # normalize ray directions 
#     rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)
#     # ray origins = camera center. same origin for every pixel 
#     rays_o = np.broadcast_to(t, rays_d.shape)

#     return rays_o, rays_d # ray origins and ray directions have shape of (H, W, 3)

# # test on first frame
# rays_o, rays_d = generate_rays(poses[0], H, W)
# print("Generated rays:", rays_o.shape, rays_d.shape)

import numpy as np
import json
import os
import cv2

# -------------------------------
# LOAD DATA
# -------------------------------
poses = np.load("poses.npy")          # shape: (N, 4, 4)
fx, fy, cx, cy = np.load("intrinsics.npy")

image_dir = "./data/frames"  # folder containing captured frames. contains RGB frames used for NeRF

# ------------------------------------------------
# FILTER IMAGES TO MATCH VALID CAMERA POSES
# ------------------------------------------------
# used_images.npy is saved during SfM and contains indices of frames
# that successfully produced valid camera poses

# ------------------------------------------------
# LOAD SfM METADATA (image ↔ pose alignment)
# ------------------------------------------------
sfm = np.load("sfm_data.npz", allow_pickle=True)
used_image_indices = sfm["used_image_indices"]

# ensures images align exactly with pose order

# all_image_files = sorted([
#     f for f in os.listdir(image_dir)
#     if f.endswith((".png", ".jpg", ".jpeg"))
# ])

# image_files = [all_image_files[i] for i in used_images]

# # safety check. every image must have a corresponding pose.
# assert len(image_files) == len(poses), "Image / pose count mismatch"

image_dir = "./data/frames"

all_image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.endswith((".png", ".jpg", ".jpeg"))
])

# Select ONLY images that have valid camera poses
image_files = [all_image_files[i] for i in used_image_indices]

# Safety check
assert len(image_files) == len(poses), (
    f"Image / pose mismatch: {len(image_files)} images vs {len(poses)} poses"
)

# -------------------------------
# EXTRACT INTRINSICS
# -------------------------------
# H: image height (pixel), W: image width (pixel). Loads first image
# need for ray generation
H, W = cv2.imread(os.path.join(image_dir, image_files[0])).shape[:2]

# K = | fx   0   cx |
#     |  0  fy   cy |
#     |  0   0    1 |

# fx = K[0, 0]
# fy = K[1, 1]
# cx = K[0, 2]
# cy = K[1, 2]

# Computes horizontal field of view
# derived from pinhole camera geometry
camera_angle_x = 2 * np.arctan(W / (2 * fx))

# -------------------------------
# CONVERT POSES TO NeRF FORMAT
# OpenCV → NeRF coordinate fix
# -------------------------------
# Flipping Y and Z aligns your SfM poses with NeRF’s expected convention.
def convert_pose(pose):
    pose = pose.copy()
    # mathematically: diag(1, -1, -1)
    # Y is inverted because images grow downward
    # Z is inverted because OpenCV looks forward, NeRF looks backward
    pose[:3, 1] *= -1   # flip Y
    pose[:3, 2] *= -1   # flip Z
    return pose

# applies the conversion to every camera pose
poses = np.array([convert_pose(p) for p in poses])

# -------------------------------
# NORMALIZE SCALE & CENTER
# -------------------------------
# poses[a_slice, b_slice, c_slice]:
# a_slice = pose index (which camera poses will you take?)
# b_slice = which rows?
# c_slice = which columns?
cam_centers = poses[:, :3, 3]

# computes the average camera location
center = cam_centers.mean(axis=0)

# recenters the entire scene around the origin
poses[:, :3, 3] -= center

# scale normalization of camera positions

# computes the Euclidean length (camera distance) of each (x,y,z) vector and then find its maximum
scale = np.max(np.linalg.norm(poses[:, :3, 3], axis=1))

# divides every camera center by the same value. normalize all camera positions
poses[:, :3, 3] /= scale

print("Poses normalized")

# -------------------------------
# WRITE transforms.json
# -------------------------------
# container of NeRF frame data
frames = []

# records image file path and camera-to-world matrix: core NeRF dataset structure
for i, fname in enumerate(image_files):
    frames.append({
        "file_path": f"./data/frames/{fname}",
        "transform_matrix": poses[i].tolist()
    })

# Tells NeRF: Camera intrinsics, Image resolution, Camera poses
transforms = {
    "camera_angle_x": camera_angle_x,
    "fl_x": fx,
    "fl_y": fy,
    "cx": cx,
    "cy": cy,
    "w": W,
    "h": H,
    "frames": frames
}

with open("transforms.json", "w") as f:
    json.dump(transforms, f, indent=4)

print("Saved transforms.json")

# -------------------------------
# RAY GENERATION (CORE NeRF STEP)
# -------------------------------
# converts a camera pose
def generate_rays(pose, H, W):
    # creates pixel coordinate grids. i: x-coordinates, j: y-coordinates
    # one (i, j) per pixel
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")

    # converts pixels to camera-space ray direction
    dirs = np.stack([
        (i - cx) / fx,
        -(j - cy) / fy,
        -np.ones_like(i)
    ], axis=-1)  # rays pointing forward

    # extracts rotation and translation
    # converts rays from camera to world space
    R = pose[:3, :3]
    t = pose[:3, 3]

    # rotates all ray direction into world coordinates
    rays_d = (dirs @ R.T)

    # normalize ray directions
    rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)

    # ray origins = camera center. same origin for every pixel
    rays_o = np.broadcast_to(t, rays_d.shape)

    return rays_o, rays_d  # ray origins and ray directions have shape of (H, W, 3)

# test on first frame
rays_o, rays_d = generate_rays(poses[0], H, W)
print("Generated rays:", rays_o.shape, rays_d.shape)
