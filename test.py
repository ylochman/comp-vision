import os
import numpy as np
import argparse
from utils import imread
from detectors import HoughTransform, FAST
from descriptors import BRIEF

class Test():
    def assignment1(self, image_path, N=5):
        image = imread(image_path)
        imagename = os.path.splitext(os.path.split(image_path)[1])[0]
        savedir = os.path.join("res", imagename)
        ht = HoughTransform(image)
        ht.getSpace(show=False, save=savedir);
        ht.getLines(N, show=False, save=savedir);
        np.savetxt(os.path.join(savedir, "lines_{}.txt".format(N)),
                   np.array(ht.lines).T)
        print("Lines (rho, theta):")
        print(np.array(ht.lines).T)
        
    def assignment2(self):
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-A", "--assignment",
                        help="# of the assigment",
                        type=int, required=True)
    parser.add_argument("-N", help="number of lines (for the assignment 1)",
                        type=int, required=True)
    args = parser.parse_args()

    test = Test()
    if args.assignment == 1:
        image_path = "./ucu-cv-code/res/marker_cut_rgb_512.png"
        test.assignment1(image_path, args.N)
    elif args.assignment == 2:
        test.assignment2()