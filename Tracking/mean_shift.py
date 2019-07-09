import os
import numpy as np
import cv2
from utils import load_tracking_data
from utils import imread, imshow, imsave, draw_rect, draw_poly
from matplotlib import pyplot as plt

HSV_MIN_DEFAULT = np.array((0., 0., 0.))
HSV_MAX_DEFAULT = np.array((255., 255., 255.))

def mean_shift_tracking(images_files, roi,
                        hsv_min=HSV_MIN_DEFAULT, hsv_max=HSV_MAX_DEFAULT,
                        camshift=False,
                        show=False, save=None):
    RoIs = [roi]
    x, y, w, h = roi
    next_img = imread(images_files[0])
    if show:
        imshow(cv2.cvtColor(next_img, cv2.COLOR_RGB2HSV)[:,:,0], title="Initial: Hue channel", sub=(1,3,1))
        imshow(draw_rect(next_img, roi), title="Image + RoI", sub=(1,3,3))
        plt.show()
    if save is not None:
            if not camshift:
                imsave(draw_rect(next_img, roi), path='res/mean_shift/{}/{}'.format(save, os.path.split(images_files[0])[1]))
            else:
                imsave(draw_rect(next_img, roi), path='res/cam_shift/{}/{}'.format(save, os.path.split(images_files[0])[1]))
    roi_rgb = next_img[y:y+h, x:x+w]
    roi_hsv =  cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(roi_hsv, hsv_min, hsv_max)
    roi_hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0,180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    for i in range(1, len(images_files)):
        next_img = imread(images_files[i])
        hsv = cv2.cvtColor(next_img, cv2.COLOR_RGB2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        if not camshift:
            ret, roi = cv2.meanShift(dst, roi, term_crit)
        else:
            ret, roi = cv2.CamShift(dst, roi, term_crit)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
        RoIs.append(roi)
        if show:
            imshow(hsv[:,:,0], title="{}: Hue".format(i), sub=(1,3,1))
            imshow(dst, title="Back-Projected", sub=(1,3,2))
            if not camshift:
                imshow(draw_rect(next_img, roi), title="Image + RoI", sub=(1,3,3))
            else:
                imshow(draw_poly(next_img, pts), title="Image + RoI", sub=(1,3,3))
            plt.show()
        if save is not None:
            if not camshift:
                imsave(draw_rect(next_img, roi), path='res/mean_shift/{}/{}'.format(save, os.path.split(images_files[i])[1]))
            else:
                imsave(draw_poly(next_img, pts), path='res/cam_shift/{}/{}'.format(save, os.path.split(images_files[i])[1]))

def main(args):
    images_files, gt_rects = load_tracking_data("data/tracking/{}".format(args.dirname))
    roi = tuple(gt_rects[0])
    mean_shift_tracking(images_files, roi, camshift=args.adaptive, show=False, save=args.dirname)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", type=str, default="Jump", help="path to image and template (default: Jump)")
    parser.add_argument("--roi", type=str, default="", help="roi in (x, y, w, h) format")
    parser.add_argument("--adaptive", action='store_true', default=False, help="CAMShift")
    args = parser.parse_args()
    main(args)