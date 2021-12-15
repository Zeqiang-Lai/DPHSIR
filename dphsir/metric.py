import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import functools

# Data format: H W C

__all__ = [
    'psnr',
    'ssim',
    'sam',
    'ergas',
    'mpsnr',
    'mssim',
    'mpsnr_max'
]


def psnr(output, target, data_range=1):
    return peak_signal_noise_ratio(target, output, data_range=data_range)


def ssim(img1, img2, **kwargs):
    return structural_similarity(img1, img2, multichannel=True, **kwargs)


def sam(img1, img2, eps=1e-8):
    """
    Spectral Angle Mapper which defines the spectral similarity between two spectra
    """
    tmp1 = np.sum(img1*img2, axis=2) + eps
    tmp2 = np.sqrt(np.sum(img1**2, axis=2)) + eps
    tmp3 = np.sqrt(np.sum(img2**2, axis=2)) + eps
    tmp4 = tmp1 / tmp2 / tmp3
    angle = np.arccos(tmp4.clip(-1, 1))
    return np.mean(np.real(angle))


def ergas(output, target, r=1):
    b = target.shape[-1]
    ergas = 0
    for i in range(b):
        ergas += np.mean((target[:, :, i]-output[:, :, i])**2) / (np.mean(target[:, :, i])**2)
    ergas = 100*r*np.sqrt(ergas/b)
    return ergas


# ---------------------------------------------------------------------------- #
#                               BandWise Metrics                               #
# ---------------------------------------------------------------------------- #

def bandwise(func):
    @functools.wraps(func)
    def warpped(output, target, *args, **kwargs):
        C = output.shape[-1]
        total = 0
        for ch in range(C):
            x = output[:, :, ch]
            y = target[:, :, ch]
            total += func(x, y, *args, **kwargs)
        return total / C
    return warpped


@bandwise
def mpsnr(output, target, data_range=1):
    return psnr(target, output, data_range=data_range)


@bandwise
def mssim(img1, img2, **kwargs):
    return ssim(img1, img2, **kwargs)


def mpsnr_max(output, target):
    """ Different from mpsnr, this function use max value of 
        each channel (instead of 1 or 255) as the peak signal.
    """
    total = 0.
    for k in range(target.shape[-1]):
        peak = np.amax(target[:, :, k])**2
        mse = np.mean((output[:, :, k]-target[:, :, k])**2)
        total += 10*np.log10(peak / mse)
    return total / target.shape[-1]
