import os

import torch
import numpy as np
import cv2
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd, numpy_image_to_torch
import matplotlib.pyplot as plt
import time

# ALIKED+LightGlue (or change extractor to use SuperPoint+LightGlue)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

max_num_keypoints = 2048

# For superpoint+lightglue:
# feature_weights_path = "checkpoints/superpoint_v1.pth"
# matcher_weight_path = "checkpoints/superpoint_lightglue_v0-1_arxiv.pth"
# extractor = SuperPoint(weights_path=feature_weights_path, max_num_keypoints=max_num_keypoints).eval().to(device)
# matcher = LightGlue(path_or_url=matcher_weight_path).eval().to(device)


# For ALIKED+LightGlue:
# - ALIKED has 128-dim descriptors, so need aliked_lightglue matcher
local_weights_root = "checkpoints/pth"
local_aliked_path = os.path.join(local_weights_root, "aliked-n16rot.pth")
extractor = ALIKED(local_path=local_aliked_path, max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features='aliked', local_weights_root=local_weights_root).eval().to(device)

# Alternative: SuperPoint+LightGlue (uncomment to use)
# feature_weights_path = "checkpoints/superpoint_v1.pth"
# extractor = SuperPoint(weights_path=feature_weights_path, max_num_keypoints=max_num_keypoints).eval().to(device)
# matcher = LightGlue(features='superpoint').eval().to(device)

# or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
# extractor = DISK(max_num_keypoints=2048).eval().to(device)  # load the extractor
# matcher = LightGlue(features='disk').eval().to(device)  # load the matcher


# extractor = SIFT(max_num_keypoints=2048).eval().cuda()  # load the extractor
# matcher = LightGlue(features='sift').eval().cuda()  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0_path = "E:\\debug\\input\\ADSC00153_undist.jpg"
image1_path = "mirrored_left_right.jpg"

start_time = time.time()
image0 = load_image(image0_path).cuda() / 255.0
image1 = load_image(image1_path).cuda() / 255.0
end_time = time.time()
print(f"Image loading time: {end_time - start_time:.2f} seconds")

# extract local features
start_time = time.time()
feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
feats1 = extractor.extract(image1)
end_time = time.time()
print(f"Feature extraction time: {end_time - start_time:.2f} seconds")
print(f"feat0 num: {feats0['keypoints'].shape[1]}, feat1 num: {feats1['keypoints'].shape[1]}")

# match the features
start_time = time.time()
matches01 = matcher({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
matches = matches01['matches']  # indices with shape (K,2)
points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
end_time = time.time()
print(f"Matching time: {end_time - start_time:.2f} seconds")

# Randomly select visualize_num matching pairs for visualization
visualize_num = 50
total_matches = len(matches)
print(f"Total matches found: {total_matches}")

if total_matches > 0:
    # Select up to 100 random matches
    num_to_show = min(visualize_num, total_matches)
    random_indices = torch.randperm(total_matches)[:num_to_show]
    
    selected_points0 = points0[random_indices].cpu().numpy()
    selected_points1 = points1[random_indices].cpu().numpy()
    
    print(f"Selected {num_to_show} random matching pairs for visualization")
    
    # Load original images for visualization
    img0_orig = cv2.imread(image0_path)
    img1_orig = cv2.imread(image1_path)
    
    if img0_orig is not None and img1_orig is not None:
        # Convert BGR to RGB for matplotlib
        img0_rgb = cv2.cvtColor(img0_orig, cv2.COLOR_BGR2RGB)
        img1_rgb = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot image 0 with keypoints
        axes[0].imshow(img0_rgb)
        axes[0].scatter(selected_points0[:, 0], selected_points0[:, 1], 
                       c='red', s=20, alpha=0.7, edgecolors='white', linewidth=0.5)
        axes[0].set_title(f'Image 0 - {num_to_show} Matching Keypoints', fontsize=14)
        axes[0].axis('off')
        
        # Plot image 1 with keypoints
        axes[1].imshow(img1_rgb)
        axes[1].scatter(selected_points1[:, 0], selected_points1[:, 1], 
                       c='blue', s=20, alpha=0.7, edgecolors='white', linewidth=0.5)
        axes[1].set_title(f'Image 1 - {num_to_show} Matching Keypoints', fontsize=14)
        axes[1].axis('off')
        
        # Create a figure showing connected matches
        fig_combined, ax_combined = plt.subplots(figsize=(20, 10))
        
        # Concatenate images horizontally
        combined_img = np.hstack((img0_rgb, img1_rgb))
        ax_combined.imshow(combined_img)
        
        # Draw connecting lines between matching points
        for i in range(len(selected_points0)):
            # Points in the first image
            x1, y1 = selected_points0[i]
            # Points in the second image (adjusted for concatenated image)
            x2, y2 = selected_points1[i][0] + img0_rgb.shape[1], selected_points1[i][1]
            
            # Draw a line connecting the matching points with random color
            color = np.random.rand(3,)  # Random RGB color for each line
            ax_combined.plot([x1, x2], [y1, y2], color=color, linewidth=0.8, alpha=0.7)
        
        ax_combined.set_xlim(0, combined_img.shape[1])
        ax_combined.set_ylim(combined_img.shape[0], 0)
        ax_combined.set_title(f'Connected Matches Visualization - {num_to_show} Random Pairs', fontsize=16)
        ax_combined.axis('off')
        
        # Save the combined visualization
        plt.savefig('connected_matching_visualization.png', dpi=300, bbox_inches='tight')
        print("Connected visualization saved as 'connected_matching_visualization.png'")
        
        # Show the combined plot
        plt.show()
        
        # Also show the separated view as before
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(img0_rgb)
        axes[0].scatter(selected_points0[:, 0], selected_points0[:, 1], 
                       c='red', s=20, alpha=0.7, edgecolors='white', linewidth=0.5)
        axes[0].set_title(f'Image 0 - {num_to_show} Matching Keypoints', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(img1_rgb)
        axes[1].scatter(selected_points1[:, 0], selected_points1[:, 1], 
                       c='blue', s=20, alpha=0.7, edgecolors='white', linewidth=0.5)
        axes[1].set_title(f'Image 1 - {num_to_show} Matching Keypoints', fontsize=14)
        axes[1].axis('off')
        
        plt.suptitle(f'LightGlue Matching Results - {num_to_show} Random Pairs', fontsize=16)
        plt.tight_layout()
        plt.savefig('separate_matching_visualization.png', dpi=300, bbox_inches='tight')
        print("Separate visualization saved as 'separate_matching_visualization.png'")
        plt.show()
    else:
        print("Error: Could not load original images for visualization")
else:
    print("No matches found!")
