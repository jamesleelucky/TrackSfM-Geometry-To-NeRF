import cv2
import numpy as np
import time

# ---------- LOAD CALIBRATION ----------
K = np.load("camera_matrix.npy")
dist = np.load("dist_coeffs.npy")

# ---------- KEYFRAME PARAMETERS ----------
KEYFRAME_TRANSLATION = 0.15
KEYFRAME_ROTATION = np.deg2rad(5)
MIN_PARALLAX = 5.0  # pixels

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

for _ in range(10):
    cap.read()
    time.sleep(0.05)

# ---------- ORB ----------
orb = cv2.ORB_create(nfeatures=1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# ---------- STATE ----------
prev_gray = None
prev_kp = None
prev_desc = None
frame_idx = 0

# ---------- TRACK STRUCTURES ----------
next_track_id = 0
feature_to_track = {}   # (frame_idx, kp_idx) -> track_id
tracks = {}             # track_id -> list of (cam_idx, kp_idx)
triangulated_tracks = set()

# ---------- STORAGE ----------
points_3d = {}          # track_id -> 3D point
camera_rotations = []
camera_translations = []

obs_2d = []
obs_cam_idx = []
obs_point_idx = []

R_global = np.eye(3)
t_global = np.zeros((3, 1))

print("Running SfM... press 'q' to finish")

# ---------- MAIN LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, desc = orb.detectAndCompute(gray, None)

    if prev_desc is not None and desc is not None:
        matches = bf.match(prev_desc, desc)
        matches = sorted(matches, key=lambda x: x.distance)[:40]

        pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])

        pts_prev = cv2.undistortPoints(
            pts_prev.reshape(-1,1,2), K, dist, P=K
        ).reshape(-1,2)

        pts_curr = cv2.undistortPoints(
            pts_curr.reshape(-1,1,2), K, dist, P=K
        ).reshape(-1,2)

        E, mask = cv2.findEssentialMat(
            pts_prev, pts_curr,
            focal=1.0, pp=(0,0),
            method=cv2.RANSAC
        )

        if E is not None:
            _, R, t, pose_mask = cv2.recoverPose(
                E, pts_prev, pts_curr, mask=mask
            )

            if np.linalg.norm(t) < 1e-3:
                continue

            pts_prev_in = pts_prev[pose_mask.ravel() == 1]
            pts_curr_in = pts_curr[pose_mask.ravel() == 1]
            inlier_matches = [matches[i] for i in np.where(pose_mask.ravel()==1)[0]]

            if len(pts_prev_in) < 8:
                continue

            parallax = np.mean(
                np.linalg.norm(pts_prev_in - pts_curr_in, axis=1)
            )
            if parallax < MIN_PARALLAX:
                continue

            # ---------- UPDATE GLOBAL POSE ----------
            t_global = t_global + R_global @ t
            R_global = R @ R_global

            # ---------- KEYFRAME DECISION ----------
            # Initialize a flag that decides whether this frame becomes a new camera pose
            add_keyframe = False # most frames are not keyframes (due to stability + speed)
            # first valid pose must be a keyframe 
            if len(camera_rotations) == 0:
                add_keyframe = True
            else:
                # load the last accepted keyframe pose 
                R_prev = camera_rotations[-1]
                t_prev = camera_translations[-1]
                # R_prev.T @ R_global: relative rotation between last keyframe and current pose
                # cv2.Rodrigues(): converts rotation matrix to rotation vector
                # np.linalg.norm(): gives rotation magnitude in radians
                # this measures how much the camera rotated 
                rot_delta = np.linalg.norm(cv2.Rodrigues(R_prev.T @ R_global)[0])
                # measures translation distance since last keyframe 
                trans_delta = np.linalg.norm(t_global.flatten() - t_prev)
                # a new keyframe is added only if motion is significant
                if rot_delta > KEYFRAME_ROTATION or trans_delta > KEYFRAME_TRANSLATION:
                    add_keyframe = True
                    
            # if motion is too small, skip everything below: no triangulation, no track updates
            if not add_keyframe:
                continue
            # register this frame as a new camera. cam_idx becomes the camera ID used everywhere else 
            cam_idx = len(camera_rotations)
            camera_rotations.append(R_global.copy())
            camera_translations.append(t_global.flatten())

            # ---------- TRACK UPDATE ----------
            # Iterate through inlier feature matches between previous and current frame 
            for m in inlier_matches:
                # create unique identifiers for each keypoint: (frame_id, keypoint_index)
                prev_key = (frame_idx - 1, m.queryIdx)
                curr_key = (frame_idx, m.trainIdx)
                # if the feature was already seen before, reuse its existing track 
                # enables multi-view observations 
                if prev_key in feature_to_track:    
                    track_id = feature_to_track[prev_key]
                # create a new feature track, assign a unique id
                # this happens only once per real-world point
                else:
                    track_id = next_track_id
                    next_track_id += 1
                    tracks[track_id] = []
                # associate the current keypoint with this track
                # enables continuity into the next frame 
                feature_to_track[curr_key] = track_id
                # stores which camera saw the point, and where it appeared in the image 
                tracks[track_id].append((cam_idx, kp[m.trainIdx].pt))

            # ---------- TRIANGULATE NEW MULTI-VIEW TRACKS ----------
            # Iterate over all tracked features
            for track_id, obs in tracks.items():
                # skip tracks that already have a 3d point 
                # prevents duplicate geometry and BA inconsistency 
                if track_id in triangulated_tracks:
                    continue
                
                # a point must be seen by at least two cameras
                if len(obs) < 2:
                    continue

                # use the LAST TWO keyframe observations
                # select the two most recent keyframes views 
                (cam1, pt1) = obs[-2]
                (cam2, pt2) = obs[-1]

                # triangulation math 
                
                # shapes are fixed for projection matrix construction 
                R1 = camera_rotations[cam1]
                t1 = camera_translations[cam1].reshape(3,1)

                R2 = camera_rotations[cam2]
                t2 = camera_translations[cam2].reshape(3,1)

                # construct full camera projection matrices 
                # maps 3D world -> 2D images
                P1 = K @ np.hstack((R1, t1))
                P2 = K @ np.hstack((R2, t2))
                
                # ensure opencv-compatible shape 
                pt1 = np.array(pt1).reshape(2,1)
                pt2 = np.array(pt2).reshape(2,1)
                
                # performs linear triangulation
                # output: homogenuous 4D coordinate 
                X4 = cv2.triangulatePoints(P1, P2, pt1, pt2)
                
                # converts homogenuous -> Euclidean 3D points 
                X = (X4[:3] / X4[3]).reshape(3)

                # reject invalid points 
                if not np.isfinite(X).all():
                    continue
                
                # assign one 3D point per track 
                points_3d[track_id] = X

                # multiple observations of the same 3D points 
                for cam_i, pt in obs:
                    obs_2d.append(pt)
                    obs_cam_idx.append(cam_i)
                    obs_point_idx.append(track_id)

                triangulated_tracks.add(track_id)


    prev_gray = gray
    prev_kp = kp
    prev_desc = desc
    frame_idx += 1

    cv2.imshow("SfM Capture", gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------- SAVE ----------
track_ids = sorted(points_3d.keys())
points_3d_array = np.array([points_3d[tid] for tid in track_ids])
track_id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
obs_point_idx = np.array([track_id_to_idx[tid] for tid in obs_point_idx])

np.savez(
    "sfm_data.npz",
    points_3d=points_3d_array,
    camera_rotations=np.asarray(camera_rotations),
    camera_translations=np.asarray(camera_translations),
    observations_2d=np.asarray(obs_2d),
    obs_cam_idx=np.asarray(obs_cam_idx),
    obs_point_idx=obs_point_idx,
)


print("SfM data saved")
print("Avg obs per point:", len(obs_point_idx) / max(1, len(set(obs_point_idx))))