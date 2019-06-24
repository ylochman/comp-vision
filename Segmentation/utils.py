import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (20, 20)

import cv2
from scipy.signal import convolve2d
from mycv import conv2D, imread, imshow, gaussian_kernel, normalize_minmax

task_dirtree = {
    1: ("text", "text"),
    2: ("count", "count"),
    3: ("object", "obj")
}


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

def get_example(i, task, channel=None, brightness_value=20, clahe=False, clahe2=False, show=True,
                text_bboxes=[]):
    (directory, file) = task_dirtree[task]
    image_path = "materials-module2/tasks/{}/{}{}.jpg".format(directory, file, i)
    image = imread(image_path)
    
    image_gray = image.mean(2) if channel is None else image[:,:,channel]
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
#         imshow(image_gray, sub=(1,3,2))
        imshow(image_gray, sub=(1,3,2))
    save(image, task, '{}_orig'.format(i))
    return image, image_gray

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
    if show:
        image_contours = image.copy()
        imshow(cv2.drawContours(image_contours, contours, -1,
                                (0,0,255)), sub=(1,2,1))
        imshow(text_mask, sub=(1,2,2))
        plt.show()
    return contours, text_mask[:,:,0]

def matrix_info(matrix):
    print('Shape: {}'.format(matrix.shape))
    print('Dtype: {}'.format(matrix.dtype))
    print('Min: {}'.format(matrix.min()))
    print('Max: {}'.format(matrix.max()))
    
def save(text_result, task, example_id, resdir='res'):
    plt.imsave('{}/{}/{}{}.jpg'.format(resdir, *task_dirtree[task], example_id),
               text_result, cmap='gray')