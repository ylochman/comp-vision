import cv2
import numpy as np
import argparse
from utils import imread
from detectors import HoughTransform, FAST
from descriptors import BRIEF

class Test():
    def assignment1(self, image=None):
        if image is None:
            image_path = "ucu-cv-code/res/marker_cut_rgb_512.png"
            image = imread(image_path)
        ht = HoughTransform(image)
        ht.showSpace(save=True)
        # ht.showLines(save=False)
        # ht.showParameters(save=True)

    def assignment2():
        pass

    def assignment3():
        pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-A", "--assignment", help="# of the assigment", type=int, required=True)
    args = parser.parse_args()

    test = Test()
    if args.assignment == 1:
        image = np.zeros((100, 100))
        idx = np.arange(25, 75)
        image[idx[::-1], idx] = 255
        image[idx, idx] = 255
        test.assignment1(image)