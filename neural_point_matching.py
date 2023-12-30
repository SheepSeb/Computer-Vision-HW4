import numpy as np
from scipy.optimize import least_squares
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
from superpoint_torch import SuperPoint

def plot_imgs(imgs, titles=None, cmap='brg', ylabel='', normalize=False, ax=None, dpi=100):
    n = len(imgs)
    if not isinstance(cmap, list):
        cmap = [cmap]*n
    if ax is None:
        _, ax = plt.subplots(1, n, figsize=(6*n, 6), dpi=dpi)
        if n == 1:
            ax = [ax]
    else:
        if not isinstance(ax, list):
            ax = [ax]
        assert len(ax) == len(imgs)
    for i in range(n):
        if imgs[i].shape[-1] == 3:
            imgs[i] = imgs[i][..., ::-1]  # BGR to RGB
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmap[i]),
                     vmin=None if normalize else 0,
                     vmax=None if normalize else 1)
        if titles:
            ax[i].set_title(titles[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    ax[0].set_ylabel(ylabel)
    plt.tight_layout()

img_file_name = 'calibration_pictures/image1.jpg'
img_2_file_name = 'calibration_pictures/image2.jpg'
detection_thresh = 0.09
nms_radius = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SuperPoint(detection_threshold=detection_thresh, nms_radius=nms_radius).eval()
weights = torch.load('superpoint_v6_from_tf.pth',map_location=device)
model.load_state_dict(weights)


img1 = cv2.imread(img_file_name)
img2 = cv2.imread(img_2_file_name)

images = [img_2_file_name, img_file_name]
img_processed = []
for img in images:
    img_processed.append(cv2.imread(img).mean(-1) / 255.)
    
h,w = np.array([i.shape for i in img_processed]).min(0)
images = np.stack([i[:h,:w] for i in img_processed])

with torch.no_grad():
    pred_th = model({'image': torch.from_numpy(images[:,None]).float()})
# plot_imgs(images, cmap='gray')
# for p, ax in zip(pred_th['keypoints'], plt.gcf().axes):
#     ax.scatter(*p.T, lw=0, s=4, c='lime')
    
# plt.show()

# Find the best matches using the descriptors
descriptors1 = pred_th['descriptors'][0].cpu().numpy()
descriptors2 = pred_th['descriptors'][1].cpu().numpy()

# Initialize the BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (the lower the better)
matches = sorted(matches, key=lambda x: x.distance)

# Convert keypoints to cv2.KeyPoint objects
keypoints1 = [cv2.KeyPoint(x=f[0], y=f[1], size=20) for f in pred_th['keypoints'][0].cpu().numpy()]
keypoints2 = [cv2.KeyPoint(x=f[0], y=f[1], size=20) for f in pred_th['keypoints'][1].cpu().numpy()]

# Draw the top matches
img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchesThickness=10)

img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img3 = cv2.resize(img3, (1280, 720))
cv2.imshow('Matches', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()