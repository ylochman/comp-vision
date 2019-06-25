import numpy as np
from matplotlib import pyplot as plt
import cv2
from utils import *

class PatchMatch():
    def __init__(self, img_s, img_t):
        self.img_s = img_s
        self.img_t = img_t
        self.shape_s = self.img_s.shape
        self.shape_t = self.img_t.shape
        I_y = np.stack([np.arange(self.shape_t[0])]*self.shape_t[1], 1)
        I_x = np.stack([np.arange(self.shape_t[1])]*self.shape_t[0], 0)
        self.I = np.stack([I_y, I_x], 2)
        self._random_init()

    def show_NNF(self, value=225, sub=None):
        NNF_y = self.NNF[:,:,0]
        NNF_x = self.NNF[:,:,1]
        s = normalize_minmax(np.sqrt(NNF_y**2 + NNF_x**2), uint=True)
        h = normalize_minmax(np.arctan2(NNF_x, NNF_y), uint=True)
        v = np.ones(NNF_y.shape, dtype=np.uint8) * value

        NNF_HSV = cv2.cvtColor(np.stack([h,s,v], 2), cv2.COLOR_HSV2RGB)
        imshow(NNF_HSV, sub=sub)
        return NNF_HSV

    def reconstruct(self, sub=None):
        T = self.I + self.NNF
        T_y = T[:,:,0]
        T_x = T[:,:,1]
        reconstructed = self.img_s[(T_y, T_x)]
        imshow(reconstructed, sub=sub)
        return reconstructed

    def _random_init(self):
        T_y = np.random.randint(0, self.shape_s[0], self.shape_t[:2])
        T_x = np.random.randint(0, self.shape_s[1], self.shape_t[:2])
        T = np.stack([T_y, T_x], 2)
        self.NNF = T - self.I

    def _get_neighbors(self, y, x):
        ys = [y]
        xs = [x]
        if self.condition_y(y):
            ys += [self.prev_y(y)]
            xs += [x]
        if self.condition_x(x):
            ys += [y]
            xs += [self.prev_x(x)]
        ys = np.array(ys)
        xs = np.array(xs)
        return ys, xs

    def _specify_dp(self, y_t, x_t, y_s, x_s):
        dpy1 = np.minimum(np.minimum(self.dp, y_t), y_s)
        dpy2 = np.minimum(np.minimum(self.dp, self.shape_t[0] - y_t - 1), self.shape_s[0] - y_s - 1)
        dpx1 = np.minimum(np.minimum(self.dp, x_t), x_s)
        dpx2 = np.minimum(np.minimum(self.dp, self.shape_t[1] - x_t - 1), self.shape_s[1] - x_s - 1)
        return dpy1, dpy2, dpx1, dpx2

    def _compare_patches(self, y_t, x_t, y_s, x_s):
        dpy1, dpy2, dpx1, dpx2 = self._specify_dp(y_t, x_t, y_s, x_s)
        patch_target = self.img_t[y_t-dpy1:y_t+dpy2+1, x_t-dpx1:x_t+dpx2+1]
        patch_source = self.img_s[y_s-dpy1:y_s+dpy2+1, x_s-dpx1:x_s+dpx2+1]
        assert patch_target.ndim == 3
        return distance(patch_target, patch_source).mean()

    def _propagate(self, ys, xs, debug=False):
        D = []
        y_t, x_t = ys[0], xs[0]
        vs = []
        for (y,x) in zip(ys, xs):
            v = self.NNF[y, x]
            y_s = np.minimum(self.shape_s[0]-1, np.maximum(0, y_t + v[0]))
            x_s = np.minimum(self.shape_s[1]-1, np.maximum(0, x_t + v[1]))
            vs.append(np.array([y_s - y_t, x_s - x_t]))
            d = self._compare_patches(y_t, x_t, y_s, x_s)
            D.append(d)
        opt = np.argmin(D)
        return vs[opt]

    def _random_search(self, y_t, x_t, w=None, alpha=0.5, debug=False):
        if w is None:
            w = max(*self.shape_s[:2])
        v0 = self.NNF[y_t, x_t]
        D = [self._compare_patches(y_t, x_t, y_t + v0[0], x_t + v0[1])]
        vs = [v0]
        k = w
        while k >= 1:
            R = np.random.rand(2) * 2 - 1
            v = np.round(v0 + k * R).astype(int)
            y_s = np.minimum(self.shape_s[0]-1, np.maximum(0, y_t + v[0]))
            x_s = np.minimum(self.shape_s[1]-1, np.maximum(0, x_t + v[1]))
            vs.append(np.array([y_s - y_t, x_s - x_t]))
            d = self._compare_patches(y_t, x_t, y_s, x_s)
            D.append(d)
            k *= alpha
        opt = np.argmin(D)
        return vs[opt]

    def _define_order(self, it):
        if it % 2 == 0:
            self.Y = np.arange(self.shape_t[0])
            self.X = np.arange(self.shape_t[1])
            self.prev_y = lambda y: y - 1
            self.prev_x = lambda x: x - 1
            self.condition_y = lambda y: y - 1 > -1
            self.condition_x = lambda x: x - 1 > -1
        else:
            self.Y = np.arange(self.shape_t[0]-1, -1, -1)
            self.X = np.arange(self.shape_t[1]-1, -1, -1)
            self.prev_y = lambda y: y + 1
            self.prev_x = lambda x: x + 1
            self.condition_y = lambda y: y + 1 < self.shape_t[0]
            self.condition_x = lambda x: x + 1 < self.shape_t[1]

    def _run_iteration(self, it, debug=False):
        T = self.I + self.NNF
        self._define_order(it)
        if debug:
            (x,y) = (10,10)
            ys, xs = self._get_neighbors(y, x)
            print('ys: ', ys)
            print('xs: ', xs)
            self.NNF[y,x] = self._propagate(ys, xs, debug)
            self.NNF[y, x] = self._random_search(y, x, debug)
        else:
            for y in self.Y:
                for x in self.X:
                    ys, xs = self._get_neighbors(y, x)
                    self.NNF[y,x] = self._propagate(ys, xs)
                    self.NNF[y,x] = self._random_search(y, x)
                    # if (y,x) == (self.shape_t[0]//2, self.shape_t[1]//2):
                        # self.show_NNF(sub=(1,2,1))
                        # self.reconstruct(sub=(1,2,2))
                        # plt.show()
        return T

    def run(self, iterations, patch_size, show_iterations=False):
        assert patch_size % 2 == 1
        self.dp = patch_size // 2
        for it in range(iterations):
            self._run_iteration(it)
            if show_iterations:
                self.show_NNF(sub=(1,2,1))
                self.reconstruct(sub=(1,2,2))
                plt.show()


def test():
    img_s = imread('img/source.png')
    img_t = imread('img/target.png')
    RESIZE_FACTOR = 0.2
    img_s = cv2.resize(img_s, (0,0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    img_t = cv2.resize(img_t, (0,0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

    PM = PatchMatch(img_s, img_t)
    PM.run(iterations=1, patch_size=3)


if __name__ == "__main__":
    test()