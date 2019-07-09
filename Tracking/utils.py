import os
from copy import deepcopy
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import cv2

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

def imsave(image, path, cmap=None):
    dirpath = os.path.split(path)[0]
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    plt.imsave(path, image, cmap=cmap)

def normalize_minmax(image, minvalue=0, maxvalue=255, uint=False, dim=None, eps=1e-30):
    imagemin = image.min(dim, keepdims=True)
    imagemax = image.max(dim, keepdims=True)
    image_norm = (image - imagemin) / (imagemax - imagemin + eps)
    res = image_norm * (maxvalue - minvalue) + minvalue
    res = res if not uint else np.round(res).astype(np.uint8)
    return res

def matrix_info(matrix):
    print('Shape: {}'.format(matrix.shape))
    print('Dtype: {}'.format(matrix.dtype))
    print('Min: {}'.format(matrix.min()))
    print('Max: {}'.format(matrix.max()))

def im2col(img, h, w, flatten=True):
    """
    Args:
        img: image of size [H, W[, 3]]
        h: int, kernel width
        w: int, kernel height
        flatten: boolean indicating output matrix shape
    Returns: [H_new, W_new, h, w[, 3]] tensor
          or [H_new, W_new, h * w [* 3]] matrix (if flatten is True),
              where H_new = H-h+1, W_new = W-w+1
    """
    H, W = img.shape[:2]
    C = 3 if img.ndim == 3 else 1
    H_new, W_new = H - h + 1, W - w + 1
    
    I_y = np.stack([np.arange(H_new)] * W_new, 1)[:,:,np.newaxis,np.newaxis,np.newaxis]
    I_x = np.stack([np.arange(W_new)] * H_new, 0)[:,:,np.newaxis,np.newaxis,np.newaxis]

    i = np.stack([np.stack([np.arange(h)] * w, 1)] * C, 2)[np.newaxis,np.newaxis,:,:,:]
    j = np.stack([np.stack([np.arange(w)] * h, 0)] * C, 2)[np.newaxis,np.newaxis,:,:,:]
    k = np.stack([np.stack([np.arange(C)] * w, 0)] * h, 0)[np.newaxis,np.newaxis,:,:,:]
    res = img[I_y + i, I_x + j, k] if C == 3 else img[(I_y + i)[:,:,:,:,0], (I_x + j)[:,:,:,:,0]]
    if flatten:
        return res.reshape(H_new, W_new, h * w * C)
    return res

def normalize(x, axis=(0,1)):
    """Intensity normalization of a patch.
    
    Args:
        patch: (H, W[, 3]) patch matrix
        axis: int, tuple or None
    """
    mean = x.mean(axis, keepdims=True)
    std = np.sqrt(((x - mean)**2).sum(axis, keepdims=True))
    return (x - mean) / std

def CC(x, y, axis=-1):
    """Computes the cross correlation of two elements.
    
    Args:
        x: first element of a vector space (or batch of elements)
            x should have size [[N1...Nk,] M1...Mp]
                where N1...Nk indicate batch, M1...Mp indicate vector space dimensions
        y: second element of the vector space
            y should have size [M1...Mp]
        axis: int, tuple or None 
            if int, tuple -- result is summed over it
            if None -- result is returned without summation
    """
    if axis == -1:
        assert y.ndim == 1
        res = (x.astype(float)).dot(y)
    res = x.astype(float) * y
    if axis is None:
        return res
    return res.sum(axis)

def SSD(x, y, axis=-1):
    """Computes the sum of squared differences of two elements.
    
    Args:
        x: first element of a vector space (or batch of elements)
            x should have size [[N1...Nk,] M1...Mp]
                where N1...Nk indicate batch, M1...Mp indicate vector space dimensions
        y: second element of the vector space
            y should have size [M1...Mp]
        axis: int, tuple or None 
            if int, tuple -- result is summed over it
            if None -- result is returned without summation
    """
    res = (x.astype(float) - y)**2
    if axis is None:
        return res
    return res.sum(axis)

def SAD(x, y, axis=-1):
    """Computes the sum of absolute differences of two elements.
    
    Args:
        x: first element of a vector space (or batch of elements)
            x should have size [[N1...Nk,] M1...Mp]
                where N1...Nk indicate batch, M1...Mp indicate vector space dimensions
        y: second element of the vector space
            y should have size [M1...Mp]
        axis: int, tuple or None 
            if int, tuple -- result is summed over it
            if None -- result is returned without summation
    """
    res = np.abs(x.astype(float) - y)
    if axis is None:
        return res
    return res.sum(axis)

def Sobel_kernel(direction):
    """
    Args:
        direction = 'x' or 'y'
    """
    kernel = np.outer(np.array([1, 2, 1]), np.array([-1, 0, 1]))
    return kernel if direction == 'x' else kernel.T


def load_tracking_data(datadirpath, files=True):
    images_files = sorted(glob(os.path.join(os.path.join(datadirpath, "img", "*.jpg"))))
    gt_rect_file = os.path.join(datadirpath, "groundtruth_rect.txt")
    with open(gt_rect_file, "r") as f:
        text = f.read()
        try:
            gt_rects = [np.array([int(num) for num in s.split(",")]) for s in text.split("\n") if s != '']
        except:
            gt_rects = [np.array([int(num) for num in s.split("\t")]) for s in text.split("\n") if s != '']
    if files:
        return images_files, gt_rects    
    images = np.stack([imread(imgfile) for imgfile in images_files], 0)
    return images, gt_rects


def draw_rect(frames, rects):
    """
    Args:
        frames -- (H, W[, 3]) image or list of images: [img_1 ... img_n]
        rects -- bounding box (bbox) or list of bboxes
                where bbox is [x, y, w, h]
                      x, y -- top-left coordinates
                      w, h -- size of bbox
    """
    is_list = isinstance(rects, list)
    if not is_list:
        frames = [frames]
        rects = [rects]
    frames_with_rects = deepcopy(frames)
    for frame, rect in zip(frames_with_rects, rects):
        x1, y1, w, h = rect
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0))
    return frames_with_rects if is_list else frames_with_rects[0]

def draw_poly(frames, pts):
    """
    Args:
        frames -- (H, W[, 3]) image or list of images: [img_1 ... img_n]
        pts -- points of polygon (bbox) or list of polygons points
                where bbox is [x, y, w, h]
                      x, y -- top-left coordinates
                      w, h -- size of bbox
    """
    is_list = isinstance(pts, list)
    if not is_list:
        frames = [frames]
        pts = [pts]
    frames_with_poly = deepcopy(frames)
    for frame, pts_ in zip(frames_with_poly, pts):
        cv2.polylines(frame, [pts_], True, 255, 2)
    return frames_with_poly if is_list else frames_with_poly[0]