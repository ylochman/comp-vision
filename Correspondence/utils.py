import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage import io

def url_to_image(url):
    return io.imread(url)

def imread(imgpath):
    return cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)

def imshow(img, cmap='gray', sub=None, title=None):
    if sub is not None:
        plt.subplot(*sub)
    if title is None:
        title = img.shape
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    if sub is None:
        plt.show()

def normalize_minmax(image, minvalue=0, maxvalue=255, uint=False, dim=None):
    imagemin = image.min(dim, keepdims=True)
    imagemax = image.max(dim, keepdims=True)
    eps = 1e-5
    image_norm = (image - imagemin) / (imagemax - imagemin + eps)
    res = image_norm * (maxvalue - minvalue) + minvalue
    res = res if not uint else np.round(res).astype(np.uint8)
    return res

def imshow_with_patch(y1, y2, x1, x2, img, sub=None):
    img_patch = img.copy()
    cv2.rectangle(img_patch, (x1, y1), (x2, y2), (255,0,0))
    imshow(img_patch, sub=sub)

def distance(pts1, pts2):
    return np.linalg.norm(pts1-pts2, axis=-1)