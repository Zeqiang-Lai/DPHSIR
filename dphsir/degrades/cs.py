import hdf5storage
import os
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class CASSI(object):
    """ Only work when img size = [512, 512, 31]
    """

    def __init__(self):
        self.mask = hdf5storage.loadmat(os.path.join(CURRENT_DIR, 'kernels', 'cs_mask_cassi.mat'))['mask']

    def __call__(self, img):
        return np.sum(img * self.mask, axis=2)
