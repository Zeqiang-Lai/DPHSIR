import torch
import numpy as np

from dphsir.solvers.utils import single2tensor4


# ---------------------------------------------------------------------------- #
#                          Proximal operators for SISR                         #
# ---------------------------------------------------------------------------- #


class CloseFormed_ADMM_Prox:
    def __init__(self, kernel, sf):
        self.k = kernel
        self.sf = sf
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device
        return self

    def prepare(self, img_L, x0):
        """ img_L (ndarray): [H, W, C]
        """
        sf = self.sf

        x = single2tensor4(img_L).to(self.device)
        k = single2tensor4(np.expand_dims(self.k, 2)).to(self.device)

        h, w = x.shape[-2:]
        FB = p2o(k, (h*sf, w*sf))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        STy = upsample(x, sf=sf)
        FBFy = FBC*torch.fft.fftn(STy, dim=(-2, -1))

        self.FB, self.FBC, self.F2B, self.FBFy = FB, FBC, F2B, FBFy

    def solve(self, x, rho):
        """ x (torch.Tensor): [B, C, H, W]
            rho (float): 

            return: x (torch.Tensor): [B, C, H, W]
        """
        FB, FBC, F2B, FBFy = self.FB, self.FBC, self.F2B, self.FBFy
        sf = self.sf
        rho = torch.tensor(rho).float().repeat(1, 1, 1, 1).to(self.device)

        rho = rho / 2
        FR = FBFy + torch.fft.fftn(rho*x, dim=(-2, -1))
        x1 = FB.mul(FR)
        FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = FBR.div(invW+rho)
        FCBinvWBR = FBC * invWBR.repeat(1, 1, sf, sf)
        FX = (FR-FCBinvWBR)/rho
        Xest = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))
        return Xest


def splits(a, sf):
    '''split a into sfxsf distinct blocks
    Args:
        a: NxCxWxH
        sf: split factor
    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


# ---------------------------------------------------------------------------- #
#                         Get Initial Solution for SISR                        #
# ---------------------------------------------------------------------------- #

def interpolate(img_L, sf, mode='cubic', enable_shift_pixel=True):
    import cv2
    mode_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA
    }
    if mode not in mode_map:
        raise ValueError('invalid mode: {}, choose from {}'.format(mode, mode_map.keys()))
    mode = mode_map[mode]
    x = cv2.resize(img_L, (img_L.shape[1]*sf, img_L.shape[0]*sf), interpolation=mode)
    if enable_shift_pixel:
        x = shift_pixel(x, sf)
    return x


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    from scipy.interpolate import interp2d

    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)
    return x


# ---------------------------------------------------------------------------- #
#         Namespaces that groups each prox and init functions togerther        #
# ---------------------------------------------------------------------------- #

class proxs:
    CloseFormedADMM = CloseFormed_ADMM_Prox


class inits:
    interpolate = interpolate
