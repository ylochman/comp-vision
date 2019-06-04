import numpy as np
from matplotlib import pyplot as plt
from utils import gaussian_blur

class BRIEF():
    """Binary Robust Independent Elementary Features for keypoint description
    """
    def __init__(self, image, denoise_kernel=None, denoise_sigma=1):
        self.image_orig = image
        self.image = image.mean(2) if image.ndim == 3 else image
        self.image = gaussian_blur(self.image, denoise_kernel=denoise_kernel,
                                               denoise_sigma=denoise_sigma)

    def describe(self, keypoints, descriptor_size=128, patch_size=49):
        H, W = self.image.shape
        s = patch_size // 2

        filtered_keypoints = filter(lambda pt: pt[0] > s
                                           and pt[1] > s
                                           and pt[0] < H - s
                                           and pt[1] < W - s, keypoints)

        self.keypoints = np.array(list(filtered_keypoints))
        self.descriptors = np.zeros((self.keypoints.shape[0], descriptor_size),
                                    dtype=bool)
        pts1, pts2 = np.split(np.random.randint(-s, s+1,
                                                (descriptor_size*2,2)
                                            ).astype(np.int32), 2)
        for k, keypoint in enumerate(self.keypoints):
            self.descriptors[k, :] = self.image[tuple((keypoint+pts1).T)] < \
                                self.image[tuple((keypoint+pts2).T)]
        return self.descriptors