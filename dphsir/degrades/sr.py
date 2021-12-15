
import os

import hdf5storage
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

from .utils import fspecial_gaussian, imresize_np, classical_degradation

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class AbstractDownsample:
    def is_classical_downsample(self):
        return False

    def scale_factor(self):
        return 1

    def kernel(self):
        pass


class ClassicalDownsample(AbstractDownsample):
    def __init__(self, sf, type, kernel_path=None):
        """ sf: scale factor
        """
        if type < 0 or type >= 8:
            raise ValueError('Invalid type of kernel, choice [0,1,2,3,4,5,6,7,8]')
        if kernel_path is None:
            kernel_path = os.path.join(CURRENT_DIR, 'kernels', 'kernels_12.mat')

        self.sf = sf
        self.type = type
        self.kernels = hdf5storage.loadmat(kernel_path)['kernels']
        self.k = self.kernels[0, type].astype('float')

    def __call__(self, img):
        """ input: [w,h,c]
            data range: both (0,255), (0,1) are ok
        """
        img_L = classical_degradation(img, self.k, self.sf)
        return img_L

    def kernel(self):
        return self.k

    def is_classical_downsample(self):
        return True

    def scale_factor(self):
        return self.sf

    def __str__(self):
        return 'classical' + str(self.type) + '_sf' + str(self.sf)


class GaussianDownsample(AbstractDownsample):
    def __init__(self, sf, use_zoom=False, ksize=8, sigma=3):
        """ sf: scale factor
        """
        self.sf = sf
        self.k = fspecial_gaussian(ksize, sigma)
        self.use_zoom = use_zoom

    def __call__(self, img):
        """ input: [w,h,c]
            data range: both (0,255), (0,1) are ok
        """
        if self.use_zoom:
            x = ndimage.filters.convolve(img, np.expand_dims(self.k, axis=2), mode='wrap')
            img_L = zoom(x, zoom=(1/self.sf, 1/self.sf, 1), order=2)
        else:
            img_L = classical_degradation(img, self.k, self.sf)
        return img_L

    def kernel(self):
        return self.k

    def is_classical_downsample(self):
        return True

    def scale_factor(self):
        return self.sf

    def __str__(self):
        return 'zoom_sf' + str(self.sf)


class BiCubicDownsample(AbstractDownsample):
    def __init__(self, sf, kernel_path=None):
        if sf not in [2, 3, 4]:
            raise ValueError('Invalid scale factor, choose from [2,3,4]')
        if kernel_path is None:
            kernel_path = os.path.join(CURRENT_DIR, 'kernels', 'kernels_bicubicx234.mat')

        self.sf = sf
        self.kernels = hdf5storage.loadmat(kernel_path)['kernels']
        self.k = self.kernels[0, sf-2].astype(np.float64)

    def __call__(self, img):
        """ input: [w,h,c]
            data range: both (0,255), (0,1) are ok
        """
        img_L = imresize_np(img, 1/self.sf)
        return img_L

    def kernel(self):
        return self.k

    def scale_factor(self):
        return self.sf

    def __str__(self):
        return 'bicubic' + '_sf' + str(self.sf)


class UniformDownsample(AbstractDownsample):
    def __init__(self, sf):
        self.sf = sf
        self.k = np.ones((sf, sf)) / (sf*sf)

    def __call__(self, img):
        """ input: [w,h,c]
            data range: both (0,255), (0,1) are ok
        """
        img_L = classical_degradation(img, self.k, self.sf)
        return img_L

    def kernel(self):
        return self.k

    def scale_factor(self):
        return self.sf

    def __str__(self):
        return 'uniform' + '_sf' + str(self.sf)
