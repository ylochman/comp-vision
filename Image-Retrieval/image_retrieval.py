import cv2
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from mycv import imshow

def imread(id):
    url = "http://www.ee.columbia.edu/~rj2349/index_files/Homework1/ukbench{:0>5}.jpg".format(id)
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return np.array(image)