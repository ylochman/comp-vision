import cv2
import numpy as np
from matplotlib import pyplot as plt

COLOR = ('r','g','b')

def hist(img, minvalue=0, maxvalue=255, out_dict=True):
    if not out_dict:
        hist = np.zeros((maxvalue - minvalue + 1) * 3, dtype=int)
        for ch, col in enumerate(COLOR):
            img_ = img[:,:,ch]
            for j, i in enumerate(range(minvalue, maxvalue+1)):
                hist[j + ch * (maxvalue - minvalue + 1)] = len(img_[img_==i])
        return hist
    hist = {}
    for ch, col in enumerate(COLOR):
        img_ = img[:,:,ch]
        histr = np.zeros(maxvalue - minvalue + 1, dtype=int)
        for j, i in enumerate(range(minvalue, maxvalue+1)):
            histr[j] = len(img_[img_==i])
        hist[col] = histr
    return hist

def hist_cv2(img):
    hist = {}
    for ch, col in enumerate(COLOR):
        histr = cv2.calcHist([img], [ch], None, [256], [0,256])
        hist[col] = histr
    return hist

def hist_np(img):
    hist = {}
    for ch, col in enumerate(COLOR):
        histr = np.histogram(img[:,:,ch].flatten(), bins=255)[0]
        hist[col] = histr
    return hist

def histshow(hist, sub=None, title='', isdict=True):
    if sub is not None:
        plt.subplot(*sub)
    plt.title(title)
    if isdict:
        for col in COLOR:
            plt.plot(hist[col], color=col)
            plt.xlim([0,256])
    else:
        plt.plot(hist, color='k')
    if sub is None:
        plt.show()