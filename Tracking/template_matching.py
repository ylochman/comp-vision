import numpy as np
from matplotlib import pyplot as plt
import cv2
from utils import *

def match_template(image, template, matchers, pad=False, pad_value=0, norm_patches=True, DEBUG=False):
    """Computes
    
    Args:
        image: (H, W[, 3]) matrix corresponding to the image
        template: (h, w[, 3]) matrix corresponding to the template patch, where h <= H, w <= W
        matchers: matcher name / list of matchers names. Available matchers:
                    "SSD" (sum of squared differences),
                    "CC" (cross correlation),
                    "SAD" (sum of absolute differences)
        pad: boolean indicating whether to pad image with pad_value constant
                     to get the same size output (default False)
        pad_value: int (default 0)
        norm_patches: boolean indicating whether to normalize patches (using intensity mean and std)

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
        image = np.pad(image, pad_width, mode="constant", constant_values=pad_value)
        H, W = image.shape[:2]
    H_new, W_new = H-h+1, W-w+1
    if norm_patches:
        image_cols = normalize(im2col(image, h, w, False), axis=(2,3)).reshape(H_new, W_new, h * w * C) / np.sqrt(C)
        template_flat = normalize(template, axis=(0,1)).flatten() / np.sqrt(C)
    else:
        image_cols = im2col(image, h, w, True)
        template_flat = template.flatten()
    matcher_dict = {"SSD": SSD, "CC": CC, "SAD": SAD}
    matching_maps = []
    if isinstance(matchers, str):
        matchers = [matchers]
    for matcher in matchers:
        assert matcher in matcher_dict.keys()
        matcher_fn = matcher_dict[matcher]
        # print(matcher, matcher_fn)
        matching_map = matcher_fn(image_cols, template_flat, -1)
        matching_maps.append(matching_map)
    return matching_maps