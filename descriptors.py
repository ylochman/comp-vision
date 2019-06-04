import numpy as np
from matplotlib import pyplot as plt
from utils import gaussian_blur

class BRIEF():
    """Binary Robust Independent Elementary Features for keypoint description
    """
    def __init__(self, bin_descriptor_size=256, patch_size=49):
        assert bin_descriptor_size % 8 == 0
        self.bin_descriptor_size = bin_descriptor_size
        self.descriptor_size = self.bin_descriptor_size // 8
        self.patch_size = patch_size
        self.s = self.patch_size // 2
        self.pts1, self.pts2 = np.split(np.random.randint(-self.s, self.s+1,
                                        (self.bin_descriptor_size*2, 2)
                                    ).astype(np.int32), 2)



    def describe(self, image, keypoints, denoise_kernel=None, denoise_sigma=1):
        self.image_orig = image
        self.image = image.mean(2) if image.ndim == 3 else image
        self.image = gaussian_blur(self.image, denoise_kernel=denoise_kernel,
                                               denoise_sigma=denoise_sigma)
        H, W = self.image.shape

        filtered_keypoints = filter(lambda pt: pt[0] > self.s
                                           and pt[1] > self.s
                                           and pt[0] < H - self.s
                                           and pt[1] < W - self.s, keypoints)

        self.keypoints = np.array(list(filtered_keypoints))
        self.bin_descriptors = np.zeros((self.keypoints.shape[0], self.bin_descriptor_size),
                                    dtype=bool)
        for k, keypoint in enumerate(self.keypoints):
            self.bin_descriptors[k, :] = self.image[tuple((keypoint+self.pts1).T)] < \
                                self.image[tuple((keypoint+self.pts2).T)]
        self.bin_descriptors = self.bin_descriptors.astype(np.uint8)
        self.num_keypoints = self.bin_descriptors.shape[0]
        self.descriptors = np.array([int(''.join(map(str, d)), 2)
                            for d in self.bin_descriptors.reshape(-1, 8)])
        self.descriptors = self.descriptors.reshape(self.num_keypoints, self.descriptor_size).astype(np.uint8)
        return self.keypoints, self.descriptors