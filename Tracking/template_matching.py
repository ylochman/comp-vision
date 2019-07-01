import numpy as np
from matplotlib import pyplot as plt
import cv2
from utils import *

def match_template(image, template, matcher, pad=False, pad_value=0, DEBUG=False):
    """Computes
    
    Args:
        image: (H, W[, 3]) matrix corresponding to the image
        template: (h, w[, 3]) matrix corresponding to the template patch, where h <= H, w <= W
        matcher: matcher name: "SSD" (sum of squared differences),
                               "NCC" (normalized cross correlation) or
                               "SAD" (sum of absolute differences)
        pad: boolean indicating whether to pad image with pad_value constant
                     to get the same size output (default False)
        pad_value: int (default 0)

    Returns: matching map of size (H-h+1, W-w+1[, 3]) if pad is False,
                               or (H, W[, 3]) otherwise
    """
    assert image.ndim == template.ndim # 2 or 3
    if image.ndim == 3:
        assert image.shape[2] == template.shape[2] and image.shape[2] == 3
        C = 3
    else:
        C = 1
    H, W = image.shape[:2]
    h, w = template.shape[:2]
    assert h < H and w < W
    if pad:
        dH = h - 1
        dW = w - 1
        pad_width = ((dH//2, dH//2+dH%2), (dW//2, dW//2+dW%2), (0,0)) if C == 3 \
               else ((dH//2, dH//2+dH%2), (dW//2, dW//2+dW%2))
        image = np.pad(image, pad_width, mode='constant', constant_values=pad_value)
        H, W = image.shape[:2]
    H_new, W_new = H-h+1, W-w+1
    matching_map = np.zeros((H_new, W_new), dtype=np.float)
    matcher_dict = {"SSD": SSD, "NCC": NCC, "SAD": SAD}
    assert matcher in matcher_dict.keys()
    if matcher == "NCC":
        template = normalize_patch(template)
    matcher = matcher_dict[matcher]
    for y in range(H_new):
        for x in range(W_new):
            patch = image[y:y+h, x:x+w]
            match = matcher(patch, template)
            if DEBUG:
                imshow(patch, sub=(1,2,1))
                imshow(template, sub=(1,2,2), title=match)
                plt.show()
            matching_map[y,x] = match
    return matching_map