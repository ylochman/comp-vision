import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from utils import gaussian_kernel, Sobel_kernel, conv2D
from utils import normalize_minmax, threshold, imshow, imsave
from cv2 import line as cv2_line

class Canny():
    def __init__(self, image):
        self.image_orig = image
        self.image = image.mean(2) if image.ndim == 3 else image
        self.image = normalize_minmax(self.image)

    def getEdges(self, denoise_kernel=5, denoise_sigma=1, t=150,
                 show=False, save=None):
        """
        Args:
            denoise_kernel -- gaussian kernel size for denoising
            denoise_sigma -- gaussian kernel sigma for denoising
            t -- threshold
        """
        # smooth
        g_kernel = gaussian_kernel(kernel_size=denoise_kernel,
                                   sigma=denoise_sigma,
                                   pdf=True, channels=None)
        padding = denoise_kernel // 2
        image_padded = np.pad(self.image, [(padding, padding)]*2, 'reflect')
        self.denoised = normalize_minmax(conv2D(image_padded, g_kernel))

        # get gradient
        denoised_padded = np.pad(self.denoised, [(1, 1)]*2, 'reflect')
        self.dx = conv2D(denoised_padded, Sobel_kernel('x'))
        self.dy = conv2D(denoised_padded, Sobel_kernel('y'))
        self.gradient_magnitude = np.sqrt(self.dx**2 + self.dy**2)
        self.gradient_angle = np.arctan2(self.dy, self.dx)

        # threshold
        self.edges = threshold(normalize_minmax(self.gradient_magnitude), t)

        if show:
            imshow(self.edges)
            plt.show()
        if save is not None:
            imsave(self.edges, save, filename='edges.jpg', cmap='gray')
        return self.edges

    def showSteps(self):
        imshow(self.image, 'gray', (1,6,1))
        imshow(self.denoised, 'gray', (1,6,2))
        imshow(normalize_minmax(self.dx), 'gray', (1,6,3))
        imshow(normalize_minmax(self.dy), 'gray', (1,6,4))
        imshow(normalize_minmax(self.gradient_magnitude), 'gray', (1,6,5))
        imshow(normalize_minmax(self.edges), 'gray', (1,6,6))
        plt.show()

class HoughTransform():
    """
    Args:
        image -- image
        theta_density -- number of angles (density)
    """
    def __init__(self, image, theta_density=180):
        self.image_orig = image
        self.imgsize = np.array(self.image_orig.shape[:2])
        self.d = 2 * np.ceil(np.sqrt(np.sum(self.imgsize**2))).astype(np.int)
        self.offset = self.d // 2
        self.t = theta_density
        self.edges = None
        self.image_hough_space = None
        self.lines = None
    
    def getSpace(self, show=False, save=None):
        """Returns image representation in Hough space:
            $I(\rho, \theta): D \times T \to Z^{+}$
            $D = [- d // 2; d// 2] \cap Z$
            $T = [-90; 89] \cap Z$
            where $d$ is the maximum distance
        """
        self.edges = Canny(self.image_orig).getEdges(save=save)
        if self.image_hough_space is None:
            self._getSpace();
        if show:
            self._showSpace(normalize_minmax(self.image_hough_space))
        if save is not None:
            imsave(normalize_minmax(self.image_hough_space), save, filename='hough_space.jpg', cmap='gray')
        return self.image_hough_space

    def _getSpace(self):
        self.image_hough_space = np.zeros((self.d, self.t), dtype=np.uint64)

        self.thetas = np.linspace(-np.pi/2, np.pi * 89 / 180, self.t)
        cthetas = np.cos(self.thetas).reshape(1,-1)
        sthetas = np.sin(self.thetas).reshape(1,-1)

        y_idxs, x_idxs = np.nonzero(self.edges)
        y_idxs, x_idxs = y_idxs.reshape(-1,1), x_idxs.reshape(-1,1)
        rho_indices = x_idxs * cthetas + y_idxs * sthetas
        rho_indices = np.round(rho_indices).astype(np.int) + self.offset

        self.theta_indices = np.repeat(np.arange(self.t).reshape(1,-1),
                                  rho_indices.shape[0], axis=0).flatten()

        rho_indices = rho_indices.flatten()
        
        np.add.at(self.image_hough_space, (rho_indices, self.theta_indices), 1)
        self.image_hough_space = normalize_minmax(self.image_hough_space, uint=True)
        return self.image_hough_space

    def _showSpace(self, image):
        plt.imshow(image, cmap='gray')
        plt.axis()
        plt.xticks(ticks=np.arange(10, 180, 10, dtype=np.int),
                   labels=np.linspace(-80, 90, 18, dtype=np.int))
        plt.xlabel('theta, degrees')
        plt.yticks(ticks=np.arange(50, self.d, 50, dtype=np.int),
                   labels=np.linspace(-(self.offset-50), self.offset,
                                      self.d//50, dtype=np.int))
        plt.ylabel('distance, pixels')
        plt.title('Hough transform')
        plt.show()
        

    def getLines(self, N=10, min_distance=20,
                 threshold_abs=0, threshold_rel=0.25,
                 show=False, save=None):
        """Returns lines in format:
            $[(\rho_1, \theta_1), \dots, (\rho_N, \theta_N)]$
            where $\theta_i$ is in radians, 
        Args:
            N -- number of lines
            min_distance -- peaks are separated by at least this value
            threshold_abs -- minimum intensity of peaks
            threshold_rel -- minimum relative to maximum intensity of peaks
        """
        self._getLines(N, min_distance, threshold_abs, threshold_rel);
        if show or save is not None:
            rho, theta = self.lines
            self.image_lines = self._drawLines(self.image_orig, rho, theta)
            if show:
                imshow(self.image_lines)
                plt.show()
            if save is not None:
                imsave(self.image_lines, save, filename='lines_{}.jpg'.format(N))
        return self.lines

    def _getLines(self, N=100, min_distance=20,
                 threshold_abs=0, threshold_rel=0.25):
        if self.image_hough_space is None:
            self.getSpace()
        coordinates = peak_local_max(self.image_hough_space,
                                     threshold_abs=threshold_abs,
                                     threshold_rel=threshold_rel,
                                     min_distance=min_distance,
                                     num_peaks=N)

        thetas = self.thetas[coordinates[:,1]]
        rhos = coordinates[:,0] - self.offset
        self.lines = (rhos, thetas)
        return self.lines

    def _drawLines(self, image, rho, theta):
        image_lines = image.copy()
        for dist, angle in zip(rho, theta):
            x = [0, self.image_orig.shape[1]]
            y = [0, 0]
            for i in range(2):
                y[i] = int((dist - x[i] * np.cos(angle)) / np.sin(angle))
                if y[i] < 0 or y[i] >= self.image_orig.shape[0]:
                    y[i] = int(min(max(y[i], 0), self.image_orig.shape[0]))
                    x[i] = int((dist - y[i] * np.sin(angle)) / np.cos(angle))
            cv2_line(image_lines, (x[0],y[0]), (x[1],y[1]),
                 color=(255,0,0), thickness=2)
        return image_lines

class FAST():
    def __init__(self, image):
        self.image = image

