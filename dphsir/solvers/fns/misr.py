import torch

from dphsir.solvers.base import PnPSolver, call
from dphsir.solvers.utils import single2tensor4, tensor2single


class SpeProx:
    def __init__(self, srf):
        """ srf (ndarray): [3, C] """
        self.srf = srf
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device
        return self

    def prepare(self, rgb, x0):
        """ rgb (ndarray): [H, W, 3]
        """
        srf = self.srf

        z = rgb.reshape(rgb.shape[0]*rgb.shape[1], rgb.shape[2]).transpose()  # 3,H*W
        T2 = srf.T @ srf    # C,3 @ 3,C = C,C
        Ttz = srf.T @ z   # C,3 @ 3,H*W = C,H*W

        self.I = torch.eye(T2.shape[0]).to(self.device)
        self.T2 = torch.tensor(T2).float().to(self.device)
        self.Ttz = torch.tensor(Ttz).float().to(self.device)

    def solve(self, input, rho):
        """ x (torch.Tensor): [1, C, H, W]
            rho (float): 

            return: x (torch.Tensor): [1, C, H, W]
        """
        xtilde1, xtilde2 = input
        _, C, H, W = xtilde1.shape
        I, T2, Ttz = self.I, self.T2, self.Ttz
        rho = torch.tensor(rho).float().repeat(1, 1, 1, 1).to(self.device)

        xtilde1 = xtilde1.squeeze().reshape((C, H*W))
        xtilde2 = xtilde2.squeeze().reshape((C, H*W))
        x = torch.inverse(T2 + 2*rho*I).matmul(Ttz + rho*(xtilde1 + xtilde2))
        x = x.reshape((1, C, H, W))
        return x


class ADMMSolver(PnPSolver):
    def __init__(self, init, prox, denoise):
        """ prox: (prox_spe, prox_spa)
        """
        prox_spe, prox_spa = prox
        super().__init__(init, prox_spa, denoise)
        self.prox_spe = prox_spe
        self.prox_spa = prox_spa

    def restore(self, input, iter_num, rhos, sigmas, callbacks=None):
        hsi, rgb = input

        x = self.init(hsi)
        self.prox_spe.prepare(rgb, x)
        self.prox_spa.prepare(hsi, x)

        x = single2tensor4(x).to(self.device)
        w = x.clone()
        v = x.clone()
        u = torch.zeros_like(x)
        m = torch.zeros_like(x)

        for i in range(iter_num):
            context = {'x': x, 'v': v, 'u': u}
            rho = rhos(i, **context)
            sigma = sigmas(i, **context)

            # x update
            xtilde1 = v - u
            xtilde2 = w - m
            x = self.prox_spe.solve((xtilde1, xtilde2), rho)

            wtilde = x + m
            w = self.prox_spa.solve(wtilde, rho)

            # v update
            vtilde = x + u
            v = self.denoise(vtilde, sigma, iter=i)

            # u update
            u = u + x - v
            m = m + x - w

            context.update({'iter': i, 'total': iter_num})
            call(callbacks, **context)

        x = tensor2single(x.clamp_(0, 1))
        return x
