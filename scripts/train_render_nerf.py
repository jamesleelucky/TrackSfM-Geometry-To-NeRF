import json
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # Adds a progress bar for training epochs

# NumPy handles array manipulation, ray geometry, and camera pose math 
# PyTorch defines NeRF neural network, run forward passes, backpropogate gradients, and optimize parameters 

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DEVICE = "cpu"
NUM_SAMPLES = 96           # samples per ray
BATCH_RAYS = 1024          # rays per iteration
EPOCHS = 6000 # total number of training iterations 
LR = 5e-4                   # learning rate 
PE_L = 6                   # positional encoding bands

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
# Loads the NeRF dataset metadata produced from SfM
with open("transforms.json", "r") as f:
    meta = json.load(f)
    
# list of images + poses 
frames = meta["frames"] 
# image resolution: Height and Width
H = meta["h"]
W = meta["w"]
# focal lengths
fx = meta["fl_x"]
fy = meta["fl_y"]
# principal points 
cx = meta["cx"]
cy = meta["cy"]

poses = []
images = []

for frame in frames:
    img = cv2.imread(frame["file_path"])
    # converts OpenCV’s BGR format to RGB (NeRF expects RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # normalize pixel values 
    img = img.astype(np.float32) / 255.0
    # store images and poses 
    images.append(img)
    poses.append(np.array(frame["transform_matrix"]))

# converts list to numpy arrays 
poses = np.array(poses)
images = np.array(images)

print(f"Loaded {len(images)} images")

# ---------------------------------------------------------
# RAY GENERATION
# ---------------------------------------------------------
# generates rays for every pixel of an image 
def generate_rays(pose, H_, W_): 
    # creates grids of pixel coordinates 
    i, j = np.meshgrid(np.arange(W_), np.arange(H_), indexing="xy")
    # transforms pixel coordinates into camera-space ray directions
    # Y → up (negated because image Y grows downward)
    # Z → forward (NeRF convention)
    dirs = np.stack([
        (i - cx) / fx,
        -(j - cy) / fy,
        # For every pixel, create a ray pointing forward with unit depth
        -np.ones_like(i) # create a new array with the same shape and data type as a given input array (i), but filled entirely with the value 1. creates a Z-direction component for every pixel ray
    ], axis=-1) # When NumPy (or PyTorch) operates on multi-dimensional arrays, axis tells it which dimension to operate along. axis=-1: Use the last dimension, no matter how many dimensions the array has

    # extract camera rotation and translation 
    R = pose[:3, :3]
    t = pose[:3, 3]

    # rotates rays from camera space -> world space 
    rays_d = (dirs @ R.T)
    # normalize ray directions 
    rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)
    # all rays originate from the camera center 
    rays_o = np.broadcast_to(t, rays_d.shape)
    # returns flattened ray origins and directions
    return rays_o.reshape(-1,3), rays_d.reshape(-1,3)

# ---------------------------------------------------------
# POSITIONAL ENCODING
# ---------------------------------------------------------
# Instead of giving the network just the position, we also give it patterns based on that position.
def positional_encoding(x, L):
    # original coordinates are reserved 
    out = [x]
    for i in range(L):
        # adds sinusoidal features 
        out.append(torch.sin((2**i) * x))
        out.append(torch.cos((2**i) * x))
    return torch.cat(out, dim=-1) # combining the waves -> sharp detail 

# ---------------------------------------------------------
# NeRF MODEL
# ---------------------------------------------------------
class NeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            # Defines NeRF MLP(multi layer perception)
            nn.Linear(3 + 2*3*PE_L, 128), # input size: 3 + 2*3*PE_L, output size: 120 
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.fc(x)

# creates the model and optimizer 
model = NeRF().to(DEVICE) # moves the model to CPU 

# Adam (Adaptive Moment Estimation) algorithm. It is the gold standard for NeRF because it automatically adjusts the learning rate for each individual weight. It handles "sparse gradients" well—essential in NeRF where only certain rays/points are updated at a time.
optimizer = optim.Adam(model.parameters(), lr=LR)  # lr: learning rate 

# ---------------------------------------------------------
# VOLUME RENDERING
# ---------------------------------------------------------
def volume_render(rgb, sigma, z_vals):
    delta = z_vals[...,1:] - z_vals[...,:-1]
    delta = torch.cat([delta, torch.ones_like(delta[..., :1]) * 1e10], -1)

    alpha = 1.0 - torch.exp(-sigma * delta)
    T = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], -1),
        -1
    )[..., :-1]

    weights = T * alpha
    rgb_map = (weights[..., None] * rgb).sum(dim=1)

    return rgb_map

# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
print("Training NeRF (RGB + Density)")

for epoch in tqdm(range(EPOCHS)):
    # randomly selects a view 
    img_idx = np.random.randint(len(images))
    
    # prepare rays and colors 
    pose = poses[img_idx]
    image = images[img_idx]

    rays_o, rays_d = generate_rays(pose, H, W)
    pixels = image.reshape(-1,3)

    # randomly samples rays 
    idx = np.random.choice(len(rays_o), BATCH_RAYS, replace=False)

    # selects specific data samples for a training batch and converts them into a format compatible with PyTorch
    rays_o = torch.tensor(rays_o[idx], dtype=torch.float32)
    rays_d = torch.tensor(rays_d[idx], dtype=torch.float32)
    target_rgb = torch.tensor(pixels[idx], dtype=torch.float32)

    # performs linear depth sampling along the rays to determine where in 3D space the neural network should be queried
    z_vals = torch.linspace(0.5, 3.0, NUM_SAMPLES)
    z_vals = z_vals.expand(BATCH_RAYS, NUM_SAMPLES)

    # compute 3D sample points 
    pts = rays_o[:,None,:] + rays_d[:,None,:] * z_vals[...,None]
    pts = pts.reshape(-1,3)

    # apply positional encoding for the sampled 3D points
    # Transforms raw 3D coordinates into a higher-dimensional space using sine and cosine functions. This allows the network to learn fine, high-frequency details like sharp edges that a standard MLP might miss.
    pts_enc = positional_encoding(pts, PE_L)

    # Feeds the encoded points into the neural network to get raw predictions.
    outputs = model(pts_enc)
    # Organizes the output so each ray in the batch has its own set of predictions for every sample point along that ray. each point has 4 values. 3 for color, 1 for density 
    outputs = outputs.reshape(BATCH_RAYS, NUM_SAMPLES, 4)

    # physical interpretation
    
    # Extracts the first three values (Red, Green, Blue) and uses a Sigmoid function to squash them between 0 and 1
    rgb = torch.sigmoid(outputs[...,:3])
    # Extracts the fourth value, representing volume density (opacity). It uses a ReLU activation to ensure density is never negative
    sigma = torch.relu(outputs[...,3])
    # Uses the colors and densities of all points along a ray to calculate a single final color for that pixel, simulating how light travels through a 3D scene.
    rgb_pred = volume_render(rgb, sigma, z_vals)

    # loss and optimization
    
    # Calculates the Mean Squared Error (MSE) between the model's predicted pixel color and the actual ground truth color from the original image
    loss = ((rgb_pred - target_rgb)**2).mean()
    
    # clears previous gradients so they don't accumulate from the last training step.
    optimizer.zero_grad()
    # Performs backpropagation, calculating how much each weight in the network contributed to the error.
    loss.backward()
    # Updates the neural network's weights based on the calculated gradients 
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.5f}")

# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------
torch.save(model.state_dict(), "nerf_rgb.pt")
print("Saved nerf_rgb.pt")

# ---------------------------------------------------------
# RENDER NOVEL VIEW
# ---------------------------------------------------------

RENDER_SCALE = 4   # try 4 or 8 on CPU

# calculates the new, smaller height and width for the rendered image
H_r = H // RENDER_SCALE
W_r = W // RENDER_SCALE

def render_view_chunked(pose, chunk=4096):
    # Generates the origin and direction for every pixel in the new low-res grid.
    rays_o, rays_d = generate_rays(pose, H_r, W_r)

    # Converts the ray data into PyTorch tensors for processing.
    rays_o = torch.tensor(rays_o, dtype=torch.float32)
    rays_d = torch.tensor(rays_d, dtype=torch.float32)

    # Defines the depth points to sample along each ray, from 0.3 (near) to 4.0 (far)
    z_vals = torch.linspace(0.3, 4.0, NUM_SAMPLES)

    rgb_out = []

    # Puts the neural network in evaluation mode (disables dropout, batch norm, etc.)
    model.eval()
    
    # Disables gradient calculation. This saves massive amounts of memory and speeds up computation because we are not training 
    with torch.no_grad():
        # Loops through the total list of rays (pixels) with step of 4096.
        # rays_o.shape[0]: total number of rays (shape[0]: size of first dimension, and the first dimension is the list of all individual ray origins)
        for i in range(0, rays_o.shape[0], chunk): 
            
            # point sampling 
            
            # select the current batch of ray origins and directions 
            ro = rays_o[i:i+chunk] # ray origin
            rd = rays_d[i:i+chunk] # ray direction \
            
            # To calculate 3D points for every ray using the formula points = origin + direction * depth, you need a 'depth value' for every sample on every ray.
            
            # The expand function takes your 1D z_vals and "stretches" it into a 2D matrix that matches your batch of rays.
            z = z_vals.expand(len(ro), NUM_SAMPLES)
            
            # points = origin + direction * depth
            # calculate the 3d coordinates for every sample points 
            pts = ro[:,None,:] + rd[:,None,:] * z[...,None]
            # flattens the points 
            # -1: calculate this missing dimension automatically based on the remaining elements
            # 3: the second dimension of the resulting tensor must be exactly 3
            # missing dimension = total num of elements in tensor / product of all other specified dimension
            # if pts has 12 elements in total, it reshapes to (12/3, 3) = (4, 3)
            pts = pts.reshape(-1,3)

            # Applies the high-frequency encoding to the 3D points.
            pts_enc = positional_encoding(pts, PE_L)
            # Queries the MLP (Neural Network) to get the predicted color and density for every point.
            out = model(pts_enc)
            
            out = out.reshape(len(ro), NUM_SAMPLES, 4)
            
            # Normalizes the network outputs into valid colors (0 to 1) and valid densities (0 to positive infinity)
            
            # In digital imaging, color values must exist in a fixed range, typically 0.0 to 1.0 (where 0 is black and 1 is full intensity).The sigmoid function squashes any input (from \(-\infty \) to \(+\infty \)) into exactly this (0, 1) range.
            rgb = torch.sigmoid(out[...,:3]) 
            sigma = torch.relu(out[...,3]) # volume density (opacity) must be non-negative 
            
            # Performs the "Volume Rendering" integration, collapsing all samples along a ray into a single pixel color
            rgb_map = volume_render(rgb, sigma, z)
            
            # saves the finished pixels for this chunk
            rgb_out.append(rgb_map)
            
    # torch.cat(rgb_out, dim=0): joins all the small chunks back into one continuous array of pixels.
    # .reshape(H_r, W_r, 3): changes the flat list of pixels back into an Image (Height x Width x RGB)
    # .cpu().numpy(): moves the image from the GPU to the CPU and converts it to a NumPy array.
    return torch.cat(rgb_out, dim=0).reshape(H_r, W_r, 3).cpu().numpy()

# calls the function using a pose from the dataset
novel = render_view_chunked(poses[len(poses)//2])
# converts the (0.0-1.0) floats into (0-255) integers so it can be saved as a standard image file.
novel = (novel * 255).astype(np.uint8)

cv2.imwrite("novel_view.png", cv2.cvtColor(novel, cv2.COLOR_RGB2BGR))
print("Saved novel_view.png")


