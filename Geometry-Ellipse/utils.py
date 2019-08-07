import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import cv2


def url_to_image(url):
    print("downloading %s" % (url))
    return io.imread(url)

def imshow(img, cmap='gray', sub=None, title=None,
           ax='off', xticks=[], yticks=[]):
    if sub is not None:
        plt.subplot(*sub)
    if title is None:
        title = img.shape
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.grid(False)
    plt.axis(ax)
    if sub is None:
        plt.show()

def assert_ellipse(E):
    """Checks that A * B > H**2
    
    Args:
        E ([3,3] np.ndarray): ellipse matrix
    
    Returns:
        [3,3] np.ndarray: normalized (A==1) ellipse matrix
    """
    A = E[0,0]
    B = E[1,1]
    H = E[0,1]
    assert A * B > H**2, "Not Ellipse!!!"

def draw_ellipse(imgc, E, debug=False):
    U, S, VT = np.linalg.svd(E)
    HinvT = np.matmul(U, np.diag(np.sqrt(S)))
    H = np.linalg.inv(HinvT.T)

    if debug:
        print( "U:\n", U)
        print( "V @ M1:\n", np.matmul(np.diag([1,1,-1]),  VT).T)
        print( "V.T:\n", VT)
        print( "U @ S @ V.T:\n", np.matmul(np.matmul(U, np.diag(S)), VT) )

    img = imgc.copy()
    for alpha in range(0, 360):
        a1 = np.matmul(H, [[np.sin(np.pi * alpha/180)], [np.cos(np.pi * alpha/180)], [1]] )
        a1 = (a1/a1[2])
        a2 = np.matmul(H, [[np.sin(np.pi * (alpha + 1)/180)], [np.cos(np.pi * (alpha + 1)/180)], [1]] )
        a2 = (a2/a2[2])
        cv2.line(img, (a1[0],a1[1]), (a2[0],a2[1]), (0,0,255), 2)
    return img

def draw_pts(imgc, pts):
    img = imgc.copy()
    for pt in pts:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0,255,0), -1)
    return img