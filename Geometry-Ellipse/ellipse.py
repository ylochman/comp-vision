import cv2
import numpy as np
from utils import url_to_image, imshow, assert_ellipse

def get_ellipse_from_5_points(pts, debug=False):
    """Gets ellipse from exactly 5 points by inversing matrix M.
    Ellipse equation: A*x**2 + 2*H*x*y + B*y**2 + 2*G*x + 2*F*y + C = 0
    in vector form:       (A H G)(x)
                   (x y 1)(H B F)(y) = 0
                          (G F C)(1)
    here A*B > H**2 and we set A = 1

    Args:
        pts ([N,2] np.ndarray): ellipse points

    Returns:
        [3,3] np.ndarray: normalized (A==1) ellipse matrix
    """
    num_pts = pts.shape[0]
    ones = np.ones(num_pts)
    M = np.stack((2*pts[:,0]*pts[:,1], pts[:,1]**2, 2*pts[:,0], 2*pts[:,1], ones), 1)
    R = pts[:,0]**2
    P = -np.matmul(np.linalg.inv(M), R)
    H, B, G, F, C = P
    E = np.array([[1, H, G],
                  [H, B, F],
                  [G, F, C]], np.double)

    assert_ellipse(E)
    pts_hom = np.concatenate((pts, ones[:,None]), 1).T
    errors = np.sum(pts_hom * (E.dot(pts_hom)), 0)    
    if debug:
        _ellipse_debug(E, pts, errors)
    return E



def get_ellipse_from_points_svd(pts, debug=False):
    """Gets ellipse from N points with SVD.
    Ellipse equation: A*x**2 + 2*H*x*y + B*y**2 + 2*G*x + 2*F*y + C = 0
      in vector form:       (A H G)(x)
                     (x y 1)(H B F)(y) = 0
                            (G F C)(1)
    here A*B > H**2 and we set A = 1

    Args:
        pts ([N,2] np.ndarray): ellipse points

    Returns:
        [3,3] np.ndarray: normalized (A==1) ellipse matrix
    """
    num_pts = pts.shape[0]
    ones = np.ones(num_pts)
    M = np.stack((pts[:,0]**2, pts[:,1]**2, ones, 2*pts[:,0]*pts[:,1], 2*pts[:,0], 2*pts[:,1]), 1)

    # SVD
    U, S, VT = np.linalg.svd(M)
    R = VT[-1, :]
    R /= R[0]
    A, B, C, H, G, F = R
    E = np.array([[A, H, G],
                  [H, B, F],
                  [G, F, C]], np.double)

    assert_ellipse(E)
    pts_hom = np.concatenate((pts, ones[:,None]), 1).T
    errors = np.sum(pts_hom * (E.dot(pts_hom)), 0)
    if debug:
        print("U, S, V.T:", U.shape, S.shape, VT.shape)
        _ellipse_debug(E, pts, errors)
    return E, errors.sum()

def _ellipse_debug(E, pts, errors):
    print("E:\n", E)
    print('Result (error) on given pts: errors')
    print('Total error:', errors.sum())