import torch
import numpy as np


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
        y = torch.tensor(img_L).float().to(self.device)
        mask = torch.tensor(self.mask.astype('float')).float().to(self.device)
        def A(x): return torch.sum(x*mask, dim=2)
        def At(x): return x.unsqueeze(dim=-1)*mask
        phi = torch.sum(mask**2, dim=2)
        self.y, self.A, self.At, self.phi = y, A, At, phi

    def solve(self, x, rho):
        """ x (torch.Tensor): [B, C, H, W]
            rho (float): 

            return: x (torch.Tensor): [B, H, W, C]
        """
        y, A, At, phi = self.y, self.A, self.At, self.phi
        rho = torch.tensor(rho).float().to(self.device)
        x = x.squeeze().permute(1, 2, 0)  # convert back to [H,W,C]

        rhs = At((y-A(x))/(phi+rho))
        x = x + rhs

        x = x.permute(2, 0, 1).unsqueeze(0)
        return x


def init(img_L, mask):
    return np.expand_dims(img_L, axis=-1) * mask.astype('float')
