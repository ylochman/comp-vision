import cv2
import numpy as np
from glob import glob
import os
import tqdm
import matplotlib as mpl
from utils_hist import hist
from utils import imread, similarity, distance, imlistshow

class Dataset():
    """Dataset for Image Retrieval.

    Args:
        path -- str, path to a dataset directory
        ext -- str, extension of images in dataset (default: 'jpg')
        notebook -- bool, runing in jupyter (default: False)

    Main functions:
        get_hists() -- extracts color histogram for each image from dataset
        get_similarity(i, j) -- returns cosine similarity between images `i` and `j`
        get_distance(i, j) -- returns l2 distance between images `i` and `j`
        get_N_closest(i, N) -- returns N closest images to an image `i`
                               using similarity or distance
    """
    def __init__(self, path, ext="jpg", notebook=False):
        self.path = path
        self.ext = ext
        self.paths = sorted(glob(os.path.join(self.path, "*.{}".format(self.ext))))
        self.size = len(self.paths)
        self.hists = None
        self.progress = tqdm.tqdm_notebook if notebook else tqdm.tqdm
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        imgpath = os.path.join(self.path, "ukbench{:0>5}.{}".format(i, self.ext))
        return imread(imgpath)
    
    def get_hists(self):
        self.hists = np.zeros((2000, 256*3), dtype=int)
        for i in self.progress(range(self.size)):
            img = self.__getitem__(i)
            self.hists[i] = hist(img, out_dict=False)
        self._l2normalize_hists()
    
    def _assert_hists(self):
        assert self.hists is not None, "Call Dataset.get_hists() first"
    
    def _l2normalize_hists(self):
        self._assert_hists()
        self.hists = self.hists / (np.linalg.norm(self.hists, axis=1, keepdims=True) + 1e-15)

    def get_similarity(self, i, j):
        self._assert_hists()
        return similarity(self.hists[i], self.hists[j])
        
    def get_distance(self, i, j):
        self._assert_hists()
        return distance(self.hists[i], self.hists[j])
    
    def get_N_closest(self, i, N=10):
        self._assert_hists()
        N_closest = np.zeros(N, dtype=np.uint16)
        N_maximums = np.ones(N, dtype=np.float) * (-2)
        for j in range(self.size):
            if j != i:
                current_similarity = self.get_similarity(i, j)
                argmin = np.argmin(N_maximums)
                if current_similarity > N_maximums[argmin]:
                    N_maximums[argmin] = current_similarity
                    N_closest[argmin] = j
        return N_closest[np.argsort(-N_maximums)]

if __name__=="__main__":
    dataset = Dataset("./data", notebook=False)
    dataset.get_hists()

    mpl.rcParams['figure.figsize'] = (20,5)
    queries = [4, 40, 60, 588, 1562]
    for i in queries:
        result = dataset.get_N_closest(i)
        imlistshow([dataset[r] for r in [i]+list(result)], rows=1,
                titles=['Query (id {})'.format(i)]+['#{} (id {})'.format(ind+1, r) for ind, r in enumerate(result)])