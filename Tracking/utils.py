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
    Returns: [h, w[, 3], H_new, W_new] tensor
          or [h * w [* 3], H_new * W_new] matrix (if flatten is True),
              where H_new = H-h+1, W_new = W-w+1
    """
    H, W = img.shape[:2]
    C = 3 if img.ndim == 3 else 1
    H_new, W_new = H - h + 1, W - w + 1
    
    I_y = np.stack([np.arange(H_new)] * W_new, 1)[:,:,np.newaxis,np.newaxis,np.newaxis]
    I_x = np.stack([np.arange(W_new)] * H_new, 0)[:,:,np.newaxis,np.newaxis,np.newaxis]

    i0 = np.stack([np.stack([np.arange(h)] * w, 1)] * C, 2)[np.newaxis,np.newaxis,:,:,:]
    j0 = np.stack([np.stack([np.arange(w)] * h, 0)] * C, 2)[np.newaxis,np.newaxis,:,:,:]
    k0 = np.stack([np.stack([np.arange(C)] * w, 0)] * h, 0)[np.newaxis,np.newaxis,:,:,:]
    if C == 3:
        return img[i0+I_y,j0+I_x,k0]
    return img[(i0+I_y)[:,:,:,:,0], (j0+I_x)[:,:,:,:,0]]

def normalize_patch(patch):
    """Intensity normalization of a patch
    
    Args:
        patch: (H, W[, 3]) matrix corresponding to the first patch
    """
    mean = patch.mean((0,1))
    std = (np.sum((patch - mean)**2)) ** .5
    return (patch - mean) / std

def NCC(patch1, patch2, is_normalized1=False, is_normalized2=True):
    """Computes the normalized cross correlation of two patches.
    
    Args:
        patch1: (H, W[, 3]) matrix corresponding to the first patch
        patch2: (H, W[, 3]) matrix corresponding to the second patch
        is_normalized1: boolean indicating whether patch1 is already normalized
        is_normalized2: boolean indicating whether patch2 is already normalized
    """
    assert patch1.shape == patch2.shape, "patch1 and patch2 should have the same shape"
    if not is_normalized1:
        patch1 = normalize_patch(patch1)
    if not is_normalized2:
        patch2 = normalize_patch(patch2)
    return np.sum(patch1 * patch2)

def SSD(patch1, patch2):
    """Computes the sum of squared differences of two patches.
    
    Args:
        patch1: (H, W[, 3]) matrix corresponding to the first patch
        patch2: (H, W[, 3]) matrix corresponding to the second patch
    """
    assert patch1.shape == patch2.shape, "patch1 and patch2 should have the same shape"
    return np.sum((patch1 - patch2)**2)

def SAD(patch1, patch2):
    """Computes the sum of absolute differences of two patches.
    
    Args:
        patch1: (H, W[, 3]) matrix corresponding to the first patch
        patch2: (H, W[, 3]) matrix corresponding to the second patch
    """
    assert patch1.shape == patch2.shape, "patch1 and patch2 should have the same shape"
    return np.sum(np.abs(patch1 - patch2))