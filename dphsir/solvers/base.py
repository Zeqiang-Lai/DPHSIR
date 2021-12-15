from typing import Callable
import torch
import numpy as np

from .utils import single2tensor4, tensor2single


class Prox:
    def __init__(self):
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device
        return self

    def prepare(self, img_L, x0):
        """ img_L (ndarray): [H, W, C]
        """
        pass

    def solve(self, x, rho):
        """ x (torch.Tensor): [1, C, H, W]
            rho (float): 

            return: x (torch.Tensor): [1, C, H, W]
        """
        pass


class PnPSolver:
    def __init__(self,
                 init: Callable[[np.ndarray], np.ndarray],
                 prox: Prox,
                 denoise: Callable[[torch.Tensor, float, int], torch.Tensor]):
        """ Base class for any plug-and-play solver with only one input and prior. 

        Args:
            init: an function that gives the initial restored result. 
                  The function should follow the signature: (img_L) -> img_R
            prox: the proximal operator that solves the data subproblem problem. 
                  Must be a subclass of `solvers.base.Prox`.
            denoise: a denoising function that solves the prior subproblem problem. 
                     The function should follow the signature: (img_L, rho, iter_num) -> img_H
        """
        self.init = init
        self.prox = prox
        self.denoise = denoise

        self.device = torch.device('cpu')

    def restore(self, *inputs, iter_num, **params):
        """ Restore the image from the low-quality input. 

            - img_L (np.ndarray): low-quality image, 
                                  shape: (H, W, C), 
                                  dtype: np.float32, 
                                  range: [0, 1].
            - iter_num: number of iterations
        """
        raise NotImplementedError

    def to(self, device):
        self.device = device
        return self


class HQSSolver(PnPSolver):
    def restore(self, img_L, iter_num, rhos, sigmas):
        x = self.init(img_L)
        self.prox.prepare(img_L, x)

        x = single2tensor4(x).to(self.device)

        for i in range(iter_num):
            context = {'x': x}
            rho = rhos(i, **context)
            sigma = sigmas(i, **context)

            x = self.prox.solve(x, rho)
            x = self.denoise(x, sigma, i)

        x = tensor2single(x)
        return x


class ADMMSolver(PnPSolver):
    def restore(self, img_L, iter_num, rhos, sigmas):
        x = self.init(img_L)
        self.prox.prepare(img_L, x)

        x = single2tensor4(x).to(self.device)
        v = x.clone()
        u = torch.zeros_like(x)

        for i in range(iter_num):
            context = {'x': x, 'v': v, 'u': u}
            rho = rhos(i, **context)
            sigma = sigmas(i, **context)

            # x update
            xtilde = v - u
            x = self.prox.solve(xtilde, rho)

            # v update
            vtilde = x + u
            v = self.denoise(vtilde, sigma, iter=i)

            # u update
            u = u + x - v

        x = tensor2single(x)
        return x
