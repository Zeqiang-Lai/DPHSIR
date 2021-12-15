import torch
import numpy as np

from dphsir.solvers.utils import single2tensor4
from .ops import p2o, cconj, r2c, cabs2, csum, cdiv, cmul, splits, upsample


class Prox:
    """ Solve the x subproblem for sisr. Shared by HQS and ADMM.
    """

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

        w, h = x.shape[-2:]
        FB = p2o(k, (w*sf, h*sf))
        FBC = cconj(FB, inplace=False)
        F2B = r2c(cabs2(FB))
        STy = upsample(x, sf=sf)
        FBFy = cmul(FBC, torch.rfft(STy, 3, onesided=False))

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
        FR = FBFy + torch.rfft(rho*x, 3, onesided=False)
        x1 = cmul(FB, FR)
        FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = cdiv(FBR, csum(invW, rho))
        FCBinvWBR = cmul(FBC, invWBR.repeat(1, 1, sf, sf, 1))
        FX = (FR-FCBinvWBR)/rho.unsqueeze(-1)
        Xest = torch.irfft(FX, 3, onesided=False)
        return Xest


def init(img_L, sf, enable_shift_pixel=False):
    import cv2
    x = cv2.resize(img_L,
                   (img_L.shape[1]*sf, img_L.shape[0]*sf),
                   interpolation=cv2.INTER_CUBIC)
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
