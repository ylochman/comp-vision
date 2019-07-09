import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
from template_matching import match_template
from utils import imread, imshow, imsave, load_tracking_data, draw_rect

def track(images_files, RoI, search_radius=20, show=False, save=None):
    """
    Args:
        images_files: list of paths to images
        RoI: [x, y, w, h] array
            where (x,y) is the top-left coordinate of the RoI
                  w, h are width and height of the RoI 
    """
    print("Processing frame {}".format(1))
    frame0 = imread(images_files[0])
    postprocess(frame0, RoI, show, save, os.path.split(images_files[0])[1])
    RoIs = [RoI]
    for i in range(1, len(images_files)):
        print("Processing frame {}".format(i+1))
        frame1 = imread(images_files[i])
        matching_map, RoI = track_single(frame0, RoI, frame1, search_radius)
        RoIs.append(RoI)
        postprocess(frame1, RoI, show, save, os.path.split(images_files[i])[1])
    return RoIs

def track_single(prev_img, prev_RoI, next_img, search_radius=None):
    x1, y1, w, h = prev_RoI
    template = prev_img[y1:y1+h, x1:x1+w]
    if search_radius is None:
        search_radius = np.inf
    dy, dx = max(0, y1-search_radius), max(0, x1-search_radius)
    part = next_img[dy:y1+h+search_radius, dx:x1+w+search_radius]
    matching_map = match_template(part, template, "SSD", norm_patches=False)
    _, _, _, top_left = cv2.minMaxLoc(matching_map)
    x1, y1 = top_left
    x1, y1 = x1 + dx, y1 + dy
    next_RoI = np.array([x1, y1, w, h])
    return matching_map, next_RoI
    
def postprocess(frame, roi_coordinates, show, save, filename):
    if show:
        x1, y1, w, h = roi_coordinates
        imshow(frame[y1:y1+h, x1:x1+w], sub=(1,2,1))
        imshow(draw_rect(frame, roi_coordinates), sub=(1,2,2))
        plt.show()
    if save is not None:
        imsave(draw_rect(frame, roi_coordinates), path='res/tracking/{}/{}'.format(save, filename))

def showframes(frames):
    for i in range(len(frames)):
        cv2.imshow("Tracking", cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        cv2.waitKey(00) == ord('k')
        if cv2.waitKey(1) == ord('q'):
            break

def main(args):
    images_files, gt_rects = load_tracking_data("data/tracking/{}".format(args.dirname))
    roi = gt_rects[0] if args.roi == "" else args.roi
    RoIs = track(images_files, roi, search_radius=args.search_radius, save=args.dirname)
    # showframes(draw_rect(images, RoIs))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", type=str, default="Jump", help="path to image and template (default: Jump)")
    parser.add_argument("--roi", type=str, default="", help="roi in (x, y, w, h) format")
    parser.add_argument("--search-radius", type=int, default=20, help="(default: 20)")
    args = parser.parse_args()
    main(args)