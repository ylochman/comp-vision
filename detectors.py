import numpy as np
from matplotlib import pyplot as plt
from utils import gaussian_kernel, conv2D

class CannyEdge():
    def __init__(self, image):
        self.image_orig = image
        self.image = image.mean(2) if image.ndim == 3 else image
        
    def getEdges(self):

class HoughTransform():
    """
    Hough space: I(dist, theta) \in Z
       dist: from -max_distance // 2 to max_distance // 2
       theta: from -90 to 89
    """
    def __init__(self, image, theta_density=180):
        self.image_orig = image
        self.image = Canny(image).getEdges()
        self.imgsize = np.array(self.image.shape)
        self.max_distance = 2 * np.ceil(np.sqrt(np.sum(self.imgsize**2))).astype(np.int)
        self.offset = self.max_distance // 2
        self.theta_density = theta_density
        self.image_hough_space = self._getSpace()
        
    def _normalize(self, img):
        return np.round((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    
    def _getSpace(self):
        thetas = np.linspace(-np.pi/2, np.pi / 180 * 89, self.theta_density)#.reshape(1,-1)
        cthetas = np.cos(thetas)
        sthetas = np.sin(thetas)
        
        image_hough_space = np.zeros((self.max_distance, 180), dtype=np.int8)

        y_idxs, x_idxs = np.nonzero(self.image)
        y_idxs, x_idxs = y_idxs.reshape(-1,1), x_idxs.reshape(-1,1)

        indices = np.stack([np.round(x_idxs * np.cos(thetas) \
                                   + y_idxs * np.sin(thetas)).astype(np.int) + self.offset,
                            np.repeat(np.arange(self.theta_density).reshape(1,-1),
                                      100, axis=0)], axis=2).reshape(-1,2)

        # np.add.at(image_hough_space, indices, 1)
        for j in np.arange(180):
            ind = np.round(x_idxs * cthetas[j] \
                         + y_idxs * sthetas[j]).astype(np.int) \
                  + self.offset
            np.add.at(image_hough_space[:, j], ind, 1)
        #     image_hough_space[ind, j] += 1
        return image_hough_space
        
    def showSpace(self, save=False): 
        plt.imshow(self._normalize(self.image_hough_space), cmap='gray')
        plt.axis()
        plt.xticks(ticks=np.arange(10, 180, 10, dtype=np.int),
                   labels=np.linspace(-80, 90, 18, dtype=np.int))
        plt.xlabel('theta, degrees')
        plt.yticks(ticks=np.arange(50, self.max_distance, 50, dtype=np.int),
                   labels=np.linspace(-(self.offset-50), self.offset,
                                      self.max_distance//50, dtype=np.int))
        plt.ylabel('distance, pixels')
        plt.show()


class FAST():
    def __init__(self, image):
        self.image = image

