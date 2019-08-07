from glob import glob
import numpy as np
import cv2
from utils import imread, imshow, draw_pts


class Calibrator(object):
    """Camera calibrator
    """
    def __init__(self, datadir, criteria=None):
#         with open('{}/chessboard_square_size.txt'.format(datadir)) as f:
#             self.square_size_txt = f.read()
#         self.square_size = self.square_size_txt.split('mm')[0]
        self.imagpaths = sorted(glob('{}/*.JPG'.format(datadir)))
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, \
                         30, 0.001) if criteria is None else criteria
        world_pts = np.zeros((6,9,3), np.float32)
        world_pts[:,:,:2] = np.mgrid[0:9,0:6].transpose(2,1,0)
        self.world_pts = world_pts.reshape(-1,3)

    def run(self, show=False, resize_factor=0.2):
        all_img_points = []
        found = 0
        for imagpath in self.imagpaths:
            img = imread(imagpath)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_resized = cv2.resize(img, None,
                                     fx=resize_factor, fy=resize_factor)
            ret, corners = cv2.findChessboardCorners(img_gray, (6,9), None)
            if ret:
                print('{}: corners found.'.format(imagpath))
                corners_subpix = cv2.cornerSubPix(img_gray, corners,
                                                  (11,11), (-1,-1),
                                                  self.criteria)
                if show:
                    imshow(draw_pts(img_resized,
                                    (corners_subpix*resize_factor).squeeze(),
                                    radius=int(25*resize_factor), thickness=2),
                           title=imagpath)
                all_img_points.append(corners_subpix)
                found += 1
            else:
                print('{}: corners not found.'.format(imagpath))
                
        all_world_points = [self.world_pts] * len(all_img_points)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(all_world_points,
                                        all_img_points, img_gray.shape[::-1],
                                        None, None)
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
        img_ud = cv2.undistort(img, self.camera_matrix, self.dist_coeff[:2], None, newcameramtx)
        return img_ud
