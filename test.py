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
        ht.showLines(save=False)
        ht.showParameters(save=True)

    def assignment2(self):
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-A", "--assignment", help="# of the assigment", type=int, required=True)
    args = parser.parse_args()

    test = Test()
    if args.assignment == 1:
        test.assignment1()
    elif args.assignment == 2:
        test.assignment2()