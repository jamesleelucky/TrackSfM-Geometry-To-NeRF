import numpy as np

K = np.load("./camera_matrix.npy")

fx = K[0,0]
fy = K[1,1]
cx = K[0,2]
cy = K[1,2]

intrinsics = np.array([fx, fy, cx, cy])
np.save("intrinsics.npy", intrinsics)



