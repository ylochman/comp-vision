import os
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import cv2
from copy import deepcopy
from template_matching import *
from utils import *

def track_single(prev_img, prev_RoI, next_img, search_radius):
    x1, y1, w, h = prev_RoI
    template = prev_img[y1:y1+h, x1:x1+w]
    dy, dx = max(0,y1-search_radius), max(0,x1-search_radius)
    part = next_img[dy:y1+h+search_radius, dx:x1+w+search_radius]
    matching_map = match_template(part, template, "CC", norm_patches=True)[0]
    _, _, _, top_left = cv2.minMaxLoc(matching_map)
    x1, y1 = top_left
    x1, y1 = x1 + dx, y1 + dy
    next_RoI = np.array([x1, y1, w, h])
    return matching_map, next_RoI

def track(frames, RoI, search_radius=20, show=False, show2=False):
    """
    Args:
        frames: list of images
        RoI: [x, y, w, h] array
            where (x,y) is the top-left coordinate of the RoI
                  w, h are width and height of the RoI 
    """
    if show:
        x1, y1, w, h = RoI
        imshow(frames[0][y1:y1+h, x1:x1+w], sub=(1,2,1))
        imshow(draw_rect(frames[0], RoI), sub=(1,2,2))
        plt.show()
    if show2:
        cv2.imshow("Tracking", cv2.cvtColor(draw_rect(frames[0], RoI), cv2.COLOR_RGB2BGR))
        cv2.waitKey(00) == ord('k')
    RoIs = [RoI]
    for i in range(1,len(frames)):
        print("Processing frame {}".format(i))
        frame = frames[i]
        matching_map, RoI = track_single(frames[i-1], RoI, frames[i], search_radius)
        RoIs.append(RoI)
        if show:
            imshow(matching_map, sub=(1,2,1))
            imshow(draw_rect(frame, RoI), sub=(1,2,2))
            plt.show()
    return RoIs

def draw_rect(frames, rects):
    frames_with_rects = deepcopy(frames)
    if not isinstance(rects, list):
        x1, y1, w, h = rects
        x2, y2 = x1 + w, y1 + h
        return cv2.rectangle(frames_with_rects, (x1, y1), (x2, y2), (255,0,0))
    else:
        for frame, rect in zip(frames_with_rects, rects):
            x1, y1, w, h = rect
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0))
        return frames_with_rects

def showframes(frames, title="video"):
    for i in range(len(frames)):
        cv2.imshow(title, cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        cv2.waitKey(00) == ord('k')
        if cv2.waitKey(1) == ord('q'):
            break

def load_data(datadirpath):
    images_files = sorted(glob(os.path.join(os.path.join(datadirpath, "img", "*.jpg"))))
    gt_rect_file = os.path.join(datadirpath, "groundtruth_rect.txt")
    images = np.stack([imread(imgfile) for imgfile in images_files], 0)
    with open(gt_rect_file, "r") as f:
        text = f.read()
        try:
            gt_rects = [np.array([int(num) for num in s.split(",")]) for s in text.split("\n") if s != '']
        except:
            gt_rects = [np.array([int(num) for num in s.split("\t")]) for s in text.split("\n") if s != '']
    # showframes(draw_rect(images, gt_rects))
    return images, gt_rects

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--dirPath", help="", type=str, default="data/Walking")
    parser.add_argument("-r", "--RoI", help="", type=str, default="")
    parser.add_argument("-s", "--searchRadius", help="", type=int, default=20)
    args = parser.parse_args()

    images, gt_rects = load_data(args.dirPath)    
    RoI = gt_rects[0] if args.RoI == "" else args.RoI
    RoIs = track(images, RoI, search_radius=args.searchRadius, show=False)
    showframes(draw_rect(images, RoIs))