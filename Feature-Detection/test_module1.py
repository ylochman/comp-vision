import os
import numpy as np
import argparse
from utils import imread, gaussian_blur
from detectors import HoughTransform, FAST
from descriptors import BRIEF
import cv2

#################### Additional functions for Assignment 2 ############################
# 1. See gaussian_blur in utils.py
# 2. For find_keypoints_candidates see FAST in detectors
# 3. For compute_descriptors see BRIEF in descriptors
# 4. detect_keypoints_and_calculate_descriptors is below:

def detect_keypoints_and_calculate_descriptors(img, brief):
    # img - numpy 2d array (grayscale image)
    img_blur = gaussian_blur(img, 11, 2.0)

    # keypoints
    kp_arr = FAST(img_blur, denoise_sigma=None).getKeypoints(threshold=1,
                                                            min_distance=10,
                                                            threshold_rel=0.15)

    # keypoints & descriptors
    kp_arr, descr_arr = brief.describe(img_blur, kp_arr, denoise_sigma=None)
    kp_arr = list(map(lambda pt: (pt[1], pt[0]), kp_arr))
    return kp_arr, descr_arr

# 5. utility function from ucu-cv-code
def match_brute_force(descr_arr0, descr_arr1):
    bf = cv2.BFMatcher()
    # return 2 matches for keypoint
    matches = bf.knnMatch(descr_arr0, descr_arr1, k=2)
    matches_arr = []
    if len(matches) > 0:
        for match_a, match_b in matches:
            # mark match good if 2nd match has bigger distance
            # (filtering similar keypoints)
            if match_a.distance < 0.75 * match_b.distance:
                matches_arr.append((
                    match_a.queryIdx,
                    match_a.trainIdx
                ))
    return matches_arr

##################################   Main testing module   ##################################
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
        brief = BRIEF(bin_descriptor_size=512, patch_size=49)
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
                kp0, descr0 = detect_keypoints_and_calculate_descriptors(img0.copy(), brief)
                kp1, descr1 = detect_keypoints_and_calculate_descriptors(img1.copy(), brief)

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