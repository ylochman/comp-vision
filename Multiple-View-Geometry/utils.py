from matplotlib import pyplot as plt
import cv2
import numpy as np

def imread(imgpath):
    return cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)

def imshow(img, cmap='gray', sub=None, title=None,
           ax='off', xticks=[], yticks=[], resize_factor=None):
    if resize_factor is not None:
        img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor)
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

def drawlines(img1, img2, lines, pts1, pts2):
    """
    Args:
        img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines"""
    r,c = img1.shape[:2]
    for r,pt1,pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1, img2

def drawpts(imgc, pts):
    """[summary]
    
    Args:
        imgc ([type]): image
        pts ([type]): [(x,y)..] points
    
    Returns:
        [type]: image
    """
    img = imgc.copy()
    for pt in pts:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0,255,0), -1)
    return img

def normalize2dpts(X):
    t = np.mean(X, 0)
    X0 = X - t
    s = np.sqrt(2) / np.linalg.norm(X0, axis=1).mean()
    T = np.array([[s, 0, -s * t[0]],
                  [0, s, -s * t[1]],
                  [0, 0,  1]])
    X0 *= s
    return X0, T

def hom(X):
    """[N,2] -> [N,3]"""
    N = X.shape[0]
    return np.concatenate((X, np.ones((N,1))), 1)


def apply(T, X):
    return T.dot(hom(X).T).T[:,:2]


