import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import requests
from io import BytesIO

def imread(imgpath):
    return cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)

def imreadurl(id):
    url = "http://www.ee.columbia.edu/~rj2349/index_files/Homework1/ukbench{:0>5}.jpg".format(id)
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return np.array(image)

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

def imlistshow(imglist, rows=2, cmap=None, titles=None):
    if titles is None:
        titles = np.arange(1,len(imglist)+1)
    k = len(imglist) // rows
    for i, img in enumerate(imglist):
        imshow(img, cmap, sub=(rows,k,i+1), title=titles[i])
    plt.show()

def similarity(v1, v2):
    return v1.dot(v2)

def distance(v1, v2):
    return np.linalg.norm(v1 - v2)