import os
import numpy as np
import argparse
from utils import imread, gaussian_blur
from detectors import HoughTransform, FAST
from descriptors import BRIEF
import cv2

#################### Additional functions for Assignment 2 ############################
# See gaussian_blur in utils.py
# For find_keypoints_candidates see FAST in detectors
# For compute_descriptors see BRIEF in descriptors

# function for keypoints and descriptors calculation
def detect_keypoints_and_calculate_descriptors(img):
    # img - numpy 2d array (grayscale image)
    img_blur = gaussian_blur(img, 31, 3.0)

    # keypoints
    kp_arr = FAST(img_blur, denoise_sigma=None).getKeypoints()
    # kp_arr is array of 2d coordinates-tuples, example:
    # [(x0, y0), (x1, y1), ...]
    # xN, yN - integers

    # descriptors
    descr_arr = BRIEF(img_blur, denoise_sigma=None).describe(kp_arr).astype(np.float)
    print(descr_arr)
    # cv_descr_arr is array of descriptors (arrays), example:
    # [[v00, v01, v02, ...], [v10, v11, v12, ...], ...]
    # vNM - floats

    return kp_arr, descr_arr

def match_brute_force(descr_arr0, descr_arr1):
    bf = cv2.BFMatcher()
    # return 2 matches for keypoint
    matches = bf.knnMatch(descr_arr0, descr_arr1, k=2)
    matches_arr = []
    for match_a, match_b in matches:
        # mark match good if 2nd match has bigger distance
        # (filtering similar keypoints)
        if match_a.distance < 0.75 * match_b.distance:
            matches_arr.append((
                match_a.queryIdx,
                match_a.trainIdx
            ))
    return matches_arr

class Test():
    def assignment1(self, image_path, N=5):
        image = imread(image_path)
        imagename = os.path.splitext(os.path.split(image_path)[1])[0]
        savedir = os.path.join("res", imagename)
        ht = HoughTransform(image)
        ht.getSpace(show=True, save=savedir)
        ht.getLines(N, show=True, save=savedir)
        np.savetxt(os.path.join(savedir, "lines_{}.txt".format(N)),
                   np.array(ht.lines).T)
        print("Lines (rho, theta):")
        print(np.array(ht.lines).T)
        
    def assignment2(self, img_in_path):
        for test_name in ['translation_', 'translation_noise_', 'rotation_2_', 'rotation_5_']:
            for frame_idx in range(9):
                # read two frames
                img0_fpath = img_in_path + test_name + str(frame_idx) + '.png'
                img0 = cv2.imread(img0_fpath, cv2.IMREAD_GRAYSCALE)

                img1_fpath = img_in_path + test_name + str(frame_idx+1) + '.png'
                img1 = cv2.imread(img1_fpath, cv2.IMREAD_GRAYSCALE)

                rows = img0.shape[0]
                cols = img0.shape[1]

                # detect keypoints and calculate descriptors
                kp0, descr0 = detect_keypoints_and_calculate_descriptors(img0.copy())
                kp1, descr1 = detect_keypoints_and_calculate_descriptors(img1.copy())

                # match
                match_arr = match_brute_force(descr0, descr1)

                # draw on one image
                img_both = np.zeros((rows, cols*2), np.uint8)
                img_both[:, 0:cols] = img0
                img_both[:, cols:cols*2] = img1
                img_both_bgr = cv2.cvtColor(img_both, cv2.COLOR_GRAY2BGR)

                # keypoints as red circles
                for i in range(len(kp0)):
                    kp = kp0[i]
                    x = kp[0]
                    y = kp[1]
                    cv2.circle(img_both_bgr, (x, y), 10, (0, 0, 255))

                for i in range(len(kp1)):
                    kp = kp1[i]
                    x = kp[0] + cols
                    y = kp[1]
                    cv2.circle(img_both_bgr, (x, y), 10, (0, 0, 255))

                # matches as green lines
                for pair in match_arr:
                    x0 = kp0[pair[0]][0]
                    y0 = kp0[pair[0]][1]
                    x1 = kp1[pair[1]][0] + cols
                    y1 = kp1[pair[1]][1]
                    cv2.line(img_both_bgr, (x0, y0), (x1, y1), (0, 255, 0))

                # show image and wait for key press
                cv2.imshow('img_both', img_both_bgr)
                cv2.waitKey(-1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-A", "--assignment",
                        help="# of the assigment",
                        type=int, required=True)
    parser.add_argument("-N", help="number of lines (for the assignment 1)",
                        type=int)
    args = parser.parse_args()

    test = Test()
    if args.assignment == 1:
        image_path = "./materials-module1/res/marker_cut_rgb_512.png"
        test.assignment1(image_path, args.N)
    elif args.assignment == 2:
        img_in_path = "materials-module1/res/tracking/"
        test.assignment2(img_in_path)