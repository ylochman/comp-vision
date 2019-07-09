import numpy as np
from matplotlib import pyplot as plt
import cv2
from utils import imread, imshow, imsave, draw_rect
from utils import normalize, im2col, CC, SSD, SAD

def match_template(image, template, matchers, pad=False, pad_value=0, norm_patches=True):
    """Computes matching map for an image given a template.
    
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
    is_list = isinstance(matchers, list)
    if not is_list:
        matchers = [matchers]
    for matcher in matchers:
        assert matcher in matcher_dict.keys()
        matcher_fn = matcher_dict[matcher]
        matching_map = matcher_fn(image_cols, template_flat, -1)
        matching_maps.append(matching_map)
    return matching_maps if is_list else matching_maps[0]

def locate(image, score_map, w, h, maximum=True, threshold=None):
    """Finds a match given a score map.
    
    Args:
        image: (H, W[, 3]) matrix corresponding to the image
        score_map: (H, W) score map
        w, h: template size
        maximum: detect maximum (if True) or minmum (if False)
        threshold: threshold value for score (to get more than one match), relative to maximum

    Returns: loc: (x, y) where x is an array of top-left x coordinates,
                               y is an array of top-left y coordinates
             matched: image with matched bboxes drawn
    """
    if threshold is None:
        _, _, min_loc, max_loc = cv2.minMaxLoc(score_map)
        top_left = max_loc if maximum else min_loc
        matched = draw_rect(image, (*top_left, w, h))
        loc = tuple([np.array([p]) for p in top_left])
    else:
        matched = image.copy()
        loc = np.where(score_map >= (threshold * score_map.max()) \
                       if maximum else \
                       score_map <= ((1-threshold) * score_map.max()))
        for pt in zip(*loc[::-1]):
            cv2.rectangle(matched, pt, (pt[0] + w, pt[1] + h), (255,0,0), 2)
    return loc, matched

def main(args):
    """Runs a template matching example"""
    image = imread('data/template-matching/{}/image.jpg'.format(args.dirname))
    template = imread('data/template-matching/{}/template.jpg'.format(args.dirname))

    if args.resize:
        image = cv2.resize(image, (0,0), fx=args.resize_factor, fy=args.resize_factor)
        template = cv2.resize(template, (0,0), fx=args.resize_factor, fy=args.resize_factor)
    if args.gray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

    H, W = image.shape[:2]
    h, w = template.shape[:2]
    C = 3 if image.ndim == 3 else 1
    H_new, W_new = H-h+1, W-w+1

    matchers = ["CC", "SSD", "SAD"]
    matchers_names = dict(zip(matchers, ["Cross Correlation",
                                        "Sum of Squared Differences",
                                        "Sum of Absolute Differences"]))
    score_maps = match_template(image, template, matchers, pad=False, norm_patches=True)
    for score_map, matcher in zip(score_maps, matchers):
        loc, matched = locate(image, score_map, w, h, maximum=(matcher=="CC"), threshold=None)
        imsave(score_map, path='res/template-matching/{}/N{}_score_map.jpg'.format(args.dirname, matcher), cmap='gray')
        imsave(matched, path='res/template-matching/{}/N{}_matched.jpg'.format(args.dirname, matcher))
        if args.show:
            imshow(score_map, title=matchers_names[matcher], sub=(1,2,1))
            imshow(matched, title='Matched', sub=(1,2,2))
            plt.show()

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", type=str, default="messi", help='path to image and template (default: messi)')
    parser.add_argument("--resize", action='store_true', default=False, help='resize (default: False)')
    parser.add_argument("--resize-factor", type=float, default=0.5, help='resize factor (default: 0.5)')
    parser.add_argument("--gray", action='store_true', default=False, help='convert to gray (default: False)')
    parser.add_argument("--show", action='store_true', default=False, help='show result (default: False)')
    args = parser.parse_args()
    main(args)