import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy.stats import norm

def imread(imgpath):
    return cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)

def imshow(img, cmap=None, sub=None):
    if sub is not None:
        plt.subplot(*sub)
    plt.title(img.shape)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')

def imsave(image, path, filename, cmap=None):
    if not os.path.exists(path):
        os.makedirs(path)
    plt.imsave(os.path.join(path, filename), image, cmap=cmap)

def gaussian_kernel(kernel_size=5, sigma=3, pdf=True, channels=None):
    """Returns a 2D Gaussian kernel.
    Args:
        kernel_size -- 
        sigma -- 
        pdf -- whether to calculate PDF (if True) or difference of CDF (if False)
        channels -- number of channels to repeat kernel,
                    can be used for convolving with RGB images (then channels=3)
    """
    if pdf:
        lim = kernel_size//2 - ((kernel_size - 1) % 2) / 2
        x = np.linspace(-lim, lim, kernel_size)
        kern1d = norm.pdf(x, scale=sigma)
    else:
        lim = kernel_size//2 + (kernel_size % 2) / 2
        x = np.linspace(-lim, lim, kernel_size+1)
        kern1d = np.diff(norm.cdf(x, scale=sigma))
    kern2d = np.outer(kern1d, kern1d)
    kern2d = kern2d / kern2d.sum()
    if channels is not None:
        kern2d = np.stack([kern2d] * channels, 0)
        kern2d = np.stack([kern2d] * channels, 3)
    return kern2d

def Sobel_kernel(direction):
    """
    Args:
        direction = 'x' or 'y'
    """
    kernel = np.outer(np.array([1, 2, 1]), np.array([-1, 0, 1]))
    return kernel if direction == 'x' else kernel.T

def conv2D(img, kernel, normalize=True):
    """ Returns a 2D convolution (image * kernel) result
    Args:
        img -- image of shape (H, W) or (H, W, C)
        kernel -- kernel of shape (H_k, W_k) or (new_C, H_k, W_k, C)
        normalize -- whether to normalize kernel (if True) or not
    Returns:
        convolved -- convolved image of shape (H+1-H_k, W+1-W_k)
                     or (H+1-H_k, W+1-W_k, new_C)
    """
    assert img.ndim in [2, 3]
    channels = (img.ndim == 3)
    img = img.copy()
    kernel = kernel.copy()
    if not channels:
        assert kernel.ndim == 2
        img = img[:,:,np.newaxis]
        kernel = kernel[np.newaxis,:,:,np.newaxis]
    assert img.ndim == 3 and kernel.ndim == 4
    if normalize:
        kernel = kernel / (kernel.sum(axis=(1,2,3)) + 1e-5)
    H, W, C = img.shape
    new_C, H_k, W_k, C_ = kernel.shape
    assert C == C_
    new_H, new_W = H + 1 - H_k, W + 1 - W_k
    convolved = [np.zeros((new_H, new_W))] * new_C
    for convolved_, kernel_ in zip(convolved, kernel):
        for i in range(new_H):
            for j in range(new_W):
                convolved_[i, j] = np.sum(img[i:i+H_k, j:j+W_k, :] * kernel_)
    convolved = np.stack(convolved, axis=2)
    return convolved if channels else convolved.squeeze()

def normalize_minmax(image, minvalue=0, maxvalue=255, uint=False):
    image_norm = (image - image.min()) / (image.max() - image.min())
    res = image_norm * (maxvalue - minvalue) + minvalue
    return res if not uint else np.round(res).astype(np.uint8)

def threshold(image, t):
    return (image > t).astype(np.uint8)
