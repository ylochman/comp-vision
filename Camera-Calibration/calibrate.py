import os
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import cv2
from utils import imread, imshow


class Calibrator(object):
    """Camera calibrator from chessboard images
    """
    def __init__(self, datadir, ext='JPG', chessboard_shape=(6,9), criteria=None):
        # Find images
        self.imgpaths = sorted(glob('{}/*.{}'.format(datadir, ext)))
        assert len(self.imgpaths) > 0, "No images found"
        # Find chessboard square size
        if os.path.exists('{}/chessboard_square_size.txt'.format(datadir)):
            self.square_size = open('{}/chessboard_square_size.txt'.format(datadir)).read()
            self.square_size = float(self.square_size.split('mm')[0])
        else:
            self.square_size = 1
        self.chessboard_shape = chessboard_shape
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, \
                         30, 0.001) if criteria is None else criteria
        self.world_pts = np.zeros((chessboard_shape[1] * chessboard_shape[0], 3), np.float32)
        self.world_pts[:,:2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1,2)
        self.world_pts *= self.square_size

    def run(self, show=False, resize_factor=0.2):
        self.all_img_points = []
        self.files = []
        found = 0
        for imgpath in self.imgpaths:
            img = imread(imgpath)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.imgsize = img_gray.shape
            img_resized = cv2.resize(img, None,
                                     fx=resize_factor, fy=resize_factor)
            ret, corners = cv2.findChessboardCorners(img_gray, self.chessboard_shape, None)
            if ret:
                print('{}: corners found.'.format(imgpath))
                corners_subpix = cv2.cornerSubPix(img_gray, corners,
                                                  (11,11), (-1,-1),
                                                  self.criteria)
                if show:
                    cv2.drawChessboardCorners(img_resized, self.chessboard_shape,
                                corners_subpix*resize_factor, ret)
                    imshow(img_resized, title=imgpath)
                self.all_img_points.append(corners_subpix)
                self.files.append(os.path.split(imgpath)[1])
                found += 1
            else:
                print('{}: corners not found.'.format(imgpath))
                
        self.all_world_points = [self.world_pts] * len(self.all_img_points)
        self._calibrate()

    def remove_recalibrate(self, i):
        del self.files[i]
        del self.all_world_points[i]
        del self.all_img_points[i]
        self._calibrate()


    def _calibrate(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.all_world_points,
                                        self.all_img_points, self.imgsize[::-1], None, None,
                                        flags=cv2.CALIB_FIX_K3)
        self.camera_matrix = np.array(mtx)
        self.dist_coeff = np.array(dist)
        self.rvecs = np.array(rvecs)
        self.tvecs = np.array(tvecs)
        print('Camera matrix:\n{}'.format(self.camera_matrix))
        print('Distortion coefficients:\n{}'.format(self.dist_coeff.T))

    
    def undistort(self, img):
        h,  w = img.shape[:2]
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeff,
                                                            (w,h), 1, (w,h))
        img_ud = cv2.undistort(img, self.camera_matrix, self.dist_coeff, None, newcameramtx)
        return img_ud

    def calculate_reproj_error(self, show=True):
        mean_error = 0
        self.errors = np.zeros((len(self.all_img_points)))
        for i in range(len(self.all_world_points)):
            imgpoints_reprojected, _ = cv2.projectPoints(self.all_world_points[i], self.rvecs[i], self.tvecs[i],
                                self.camera_matrix, self.dist_coeff)
            error = cv2.norm(self.all_img_points[i], imgpoints_reprojected,
                            cv2.NORM_L2) / len(imgpoints_reprojected)
            self.errors[i] = error
            mean_error += error
        self.mean_error = mean_error / len(self.all_world_points)
        if show:
            plt.hlines(self.mean_error, 0, len(self.errors), linestyles='--')
            plt.bar(np.arange(len(self.files)), self.errors)
            # plt.bar(self.files, self.errors)
            plt.xticks(rotation=45)
            plt.show()
        return self.mean_error
