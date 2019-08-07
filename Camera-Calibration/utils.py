from matplotlib import pyplot as plt
import cv2


def imread(imgpath):
    return cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)

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

def draw_pts(imgc, pts, radius=4, color=(0,255,0), thickness=-1):
    """Draws points in the image
    
    Args:
        imgc (np.ndarray): image
        pts ([N,2] np.ndarray): 2D points
    
    Returns:
        (np.ndarray): image with points
    """
    img = imgc.copy()
    for pt in pts:
        cv2.circle(img, (int(pt[0]), int(pt[1])), radius, color, thickness)
    return img