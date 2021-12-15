
import numpy as np
from scipy import ndimage


from .utils import fspecial_gaussian


class GaussianBlur:
    def __init__(self, ksize=8, sigma=3):
        self.k = fspecial_gaussian(ksize, sigma)

    def __call__(self, img):
        # img_L = np.fft.ifftn(np.fft.fftn(img) * np.fft.fftn(np.expand_dims(self.k, axis=2), img.shape)).real
        img_L = ndimage.filters.convolve(img, np.expand_dims(self.k, axis=2), mode='wrap')
        return img_L

    def kernel(self):
        return self.k
