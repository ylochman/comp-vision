import numpy as np
import cv2
from scipy.stats import multivariate_normal as mnorm
from matplotlib import pyplot as plt

task_dirtree = {
    1: ("text", "text"),
    2: ("count", "count"),
    3: ("object", "obj")
}

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

def save(text_result, task, example_id, resdir='res'):
    plt.imsave('{}/{}/{}{}.jpg'.format(resdir, *task_dirtree[task], example_id),
               text_result, cmap='gray')

def matrix_info(matrix):
    print('Shape: {}'.format(matrix.shape))
    print('Dtype: {}'.format(matrix.dtype))
    print('Min: {}'.format(matrix.min()))
    print('Max: {}'.format(matrix.max()))


def get_example(i, task, channel=None, brightness_value=20, clahe=False, clahe2=False, show=True,
                text_bboxes=[], root="materials-module2/tasks"):
    (directory, file) = task_dirtree[task]
    image_path = "{}/{}/{}{}.jpg".format(root, directory, file, i)
    image = imread(image_path)
    
    image_gray = np.round(image.mean(2)).astype(np.uint8) if channel is None else image[:,:,channel]
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        image_gray = clahe.apply(image_gray)
    if clahe2:
        l, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(18,18))
        l = clahe.apply(l)
        image_enhanced = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB).astype(np.uint8)
        image_gray = image_enhanced.mean(2) if channel is None else image_enhanced[:,:,channel]
    image_gray = increase_brightness(image_gray, value=brightness_value)
    if show:
        imshow(draw_bboxes(image, text_bboxes), sub=(1,3,1))
        imshow(image_gray, sub=(1,3,2))
    save(image, task, '{}_orig'.format(i))
    return image, image_gray


def increase_brightness(img, value=20):
    img = img.astype(int)
    img += value
    img[img > 255] = 255
    img[img < 0] = 0
    return img.astype(np.uint8)


def draw_bboxes(image, bboxes):
    image_rect = image.copy()
    for bbox in bboxes:
        pt1 = bbox[0]
        pt1 = (pt1[1], pt1[0])
        pt2 = bbox[1]
        pt2 = (pt2[1], pt2[0])
        image_rect = cv2.rectangle(image_rect, pt1, pt2, (0,0,255), 2)
    return image_rect

def get_mask_from_bboxes(imagesize, bboxes):
    """
    Args:
        imagesize: (H, W)
        bboxes: [[[y1,x1],[y2,x2]]_1,...,[[y1,x1],[y2,x2]]_n]
            where [y1, x1] is upper left, [y2,x2] is down right
    """
    mask = np.zeros(imagesize)
    for bbox in bboxes:
        pt1 = bbox[0]
        pt2 = bbox[1]
        mask[pt1[0]:pt2[0], pt1[1]:pt2[1]] = 1
    return mask


def detect_text(image, image_gray, kernel=(11,11), dilate_kernel=None, dilate_iter=3, show=False):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    img_sobel = cv2.Sobel(image_gray, cv2.CV_8U, 1, 0)
    img_threshold = cv2.threshold(img_sobel, 0, 255,
                                  cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    img_threshold = cv2.morphologyEx(img_threshold[1],
                                     cv2.MORPH_CLOSE, element)
    contours = cv2.findContours(img_threshold,0,1)[0]
    text_mask = np.zeros_like(image)
    text_mask = cv2.drawContours(text_mask, contours, -1, (1,1,1), -1)
    if dilate_kernel:
        if isinstance(dilate_kernel, int):
            dilate_kernel = (dilate_kernel, dilate_kernel)
        text_mask = cv2.dilate(text_mask, np.ones(dilate_kernel), iterations=dilate_iter)
    text_mask = text_mask[:,:,0]
    if show:
        image_contours = image.copy()
        imshow(cv2.drawContours(image_contours, contours, -1,
                                (0,0,255)), sub=(1,2,1))
        imshow(text_mask, sub=(1,2,2))
        plt.show()
    return contours, text_mask


def gaussian_kernel(kernel_size=5, cov=3):
    """Returns a 2D Gaussian kernel (can have correlation in axes).
    Args:
        kernel_size -- kernel size
        cov -- covariance matrix
    """
    if isinstance(cov, int) or isinstance(cov, float):
        cov = np.array([[cov,0],[0,cov]])
    rv = mnorm([0, 0], cov)
    lim = kernel_size//2 - ((kernel_size - 1) % 2) / 2
    x, y = np.mgrid[-lim:lim+1:1, -lim:lim+1:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    kern2d = rv.pdf(pos)
    return kern2d

def dilate(image, kernel=2, it=1):
    if image.max() <= 1:
        image = np.round(image * 255).astype(np.uint8)
    return cv2.dilate(image, np.ones((kernel,kernel)), iterations=it).astype(bool).astype(np.uint8)

def fillHoles(image, kernel=(3,3)):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if image.max() <= 1:
        image = np.round(image * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)