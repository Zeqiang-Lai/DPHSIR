import numpy as np
import scipy.interpolate
import torch

from dphsir.solvers.utils import single2tensor4


class Prox:
    def __init__(self, mask):
        self.mask = mask
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device
        return self

    def prepare(self, img_L, x0):
        """ img_L (ndarray): [H, W, C]
        """
        x0 = single2tensor4(x0).to(self.device)
        mask = single2tensor4(self.mask).to(self.device)
        Stx = x0 * mask

        self.Stx, self.mask = Stx, mask

    def solve(self, x, rho):
        """ x (torch.Tensor): [B, H, W, C]
            rho (float): 

            return: x (torch.Tensor): [B, H, W, C]
        """
        Stx, mask = self.Stx, self.mask
        rho = torch.tensor(rho).float().repeat(1, 1, 1, 1).to(self.device)

        rhs = Stx + rho * x
        x = rhs / (mask + rho)

        return x


def Interpolation_OLRT(img, mask):
    """
    simulate random projection
    mask=0 denotes preserved pixels
    Delaunay triangulation based interpolation

    ```matlab
    [x,y,z] = ind2sub(size(c),find(c==0));
    [M,N,B]=size(im_n);
    [x1,y1,z1]=meshgrid(1:M,1:N,1:B);
    im_r=griddata(x,y,z,im_n(c==0),x1,y1,z1);
    for i =1:B
        im_r(:,:,i) = im_r(:,:,i)';
    end
    I=find(isnan(im_r)==0);
    I=find(isnan(im_r)==1);
    %J1=max(1,I-1);J2=min(M*N,I+1);
    im_r(I)=128;
    ```
    """
    w, h = img.shape
    idxs = np.argwhere(mask == 1)
    x1, y1 = np.mgrid[0:w:1, 0:h:1]
    img_r = scipy.interpolate.griddata(idxs, img[mask == 1], (x1.flatten(), y1.flatten()))
    img_r = img_r.reshape(w, h)
    img_r[np.isnan(img_r)] = 0.5
    return img_r


def Interpolation_OLRT_3D(img, mask):
    b = img.shape[-1]
    r = np.zeros_like(img)
    for i in range(b):
        r[:, :, i] = Interpolation_OLRT(img[:, :, i], mask[:, :, i])
    return r


class inits:
    interpolate = Interpolation_OLRT_3D
    none = lambda x, mask: x