from abc import abstractmethod
import torch


class Denoiser:
    def denoise(self, x, sigma):
        raise NotImplementedError

    def __call__(self, x, sigma, iter=None):
        return self.denoise(x, sigma)


class Denoiser2D(Denoiser):
    def denoise(self, x, sigma):
        outs = []
        for band in x.split(1, 1):
            band = self.denoise_2d(band, sigma)
            outs.append(band)
        return torch.cat(outs, dim=1)

    @abstractmethod
    def denoise_2d(self, x, sigma):
        pass
