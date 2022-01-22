import torch
import torch.nn as nn
import numpy as np

from .base import Denoiser, Denoiser2D


class TVDenoiser(Denoiser):
    def __init__(self, iter_num=5, use_3dtv=False):
        self.iter_num = iter_num
        self.use_3dtv = use_3dtv

    def denoise(self, x, sigma):
        from .models.TV_denoising import TV_denoising, TV_denoising3d
        x = x.squeeze()
        if self.use_3dtv:
            x = TV_denoising3d(x, sigma, self.iter_num)
        else:
            x = TV_denoising(x, sigma, self.iter_num)
        x = x.unsqueeze(0)
        return x

    def to(self, device):
        return self


class FFDNetDenoiser(Denoiser2D):
    def __init__(self, n_channels, model_path):
        from .models.network_ffdnet import FFDNet
        model = FFDNet(in_nc=n_channels, out_nc=n_channels, nc=64, nb=15, act_mode='R')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False
        self.model = model

    def denoise_2d(self, x, sigma):
        sigma = sigma.float().repeat(1, 1, 1, 1)
        x = self.model(x, sigma)
        return x

    def to(self, device):
        self.model.to(device)
        return self


class FFDNet3DDenoiser(Denoiser):
    def __init__(self, model_path):
        from .models.network_ffdnet import FFDNet3D
        model = FFDNet3D(in_nc=32, out_nc=31, nc=64, nb=15, act_mode='R')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False
        self.model = model

    def denoise(self, x, sigma):
        x = torch.cat((x, sigma.float().repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
        x = self.model(x)
        return x

    def to(self, device):
        self.model.to(device)
        return self


class IRCNNDenoiser(Denoiser2D):
    def __init__(self, n_channels, model_path):
        from .models.network_dncnn import IRCNN as net
        self.model = net(in_nc=n_channels, out_nc=n_channels, nc=64)
        self.model25 = torch.load(model_path)
        self.former_idx = 0

    def denoise_2d(self, x, sigma):
        current_idx = np.int(np.ceil(sigma.cpu().numpy()*255./2.)-1)

        if current_idx != self.former_idx:
            self.model.load_state_dict(
                self.model25[str(current_idx)], strict=True)
            self.model.eval()
            for _, v in self.model.named_parameters():
                v.requires_grad = False
            self.model = self.model.to(self.device)
        self.former_idx = current_idx

        x = self.model(x)
        return x

    def to(self, device):
        self.device = device
        return self


class DRUNetDenoiser(Denoiser2D):
    def __init__(self, n_channels, model_path):
        from .models.network_unet import UNetRes as net
        model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512],
                    nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False
        self.model = model

    def denoise_2d(self, x, sigma):
        x = torch.cat((x, sigma.float().repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
        x = self._denoise(x, refield=32, min_size=256, modulo=16)
        return x

    def to(self, device):
        self.model.to(device)
        return self

    def _denoise(self, L, refield=32, min_size=256, sf=1, modulo=1):
        '''
        model:
        L: input Low-quality image
        refield: effective receptive filed of the network, 32 is enough
        min_size: min_sizeXmin_size image, e.g., 256X256 image
        sf: scale factor for super-resolution, otherwise 1
        modulo: 1 if split
        '''
        h, w = L.size()[-2:]
        if h*w <= min_size**2:
            L = nn.ReplicationPad2d((0, int(np.ceil(w/modulo)*modulo-w), 0, int(np.ceil(h/modulo)*modulo-h)))(L)
            E = self.model(L)
            E = E[..., :h*sf, :w*sf]
        else:
            top = slice(0, (h//2//refield+1)*refield)
            bottom = slice(h - (h//2//refield+1)*refield, h)
            left = slice(0, (w//2//refield+1)*refield)
            right = slice(w - (w//2//refield+1)*refield, w)
            Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]

            if h * w <= 4*(min_size**2):
                Es = [self.model(Ls[i]) for i in range(4)]
            else:
                Es = [self._denoise(Ls[i], refield=refield,
                                    min_size=min_size, sf=sf, modulo=modulo) for i in range(4)]

            b, c = Es[0].size()[:2]
            E = torch.zeros(b, c, sf * h, sf * w).type_as(L)

            E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
            E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
            E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
            E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
        return E


class QRNN3DDenoiser(Denoiser):
    def __init__(self, model_path, use_noise_map=True):
        from .models.qrnn import qrnn3d, qrnn3d_masked
        model = qrnn3d_masked() if use_noise_map else qrnn3d()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False

        self.model = model
        self.use_noise_map = use_noise_map

    def denoise(self, x, sigma):
        if self.use_noise_map:
            x = torch.cat((x, sigma.float().repeat(1, x.shape[1], x.shape[2], x.shape[3])), dim=0)
        x = torch.unsqueeze(x, 0)
        x = self.model(x)
        x = torch.squeeze(x)
        x = torch.unsqueeze(x, 0)
        return x

    def to(self, device):
        self.model.to(device)
        return self


class GRUNetDenoiser(Denoiser):
    def __init__(self, model_path):
        from .models.qrnn import grunet_masked_nobn
        model = grunet_masked_nobn()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False

        self.model = model
        self.use_noise_map = True

    def denoise(self, x, sigma):
        if self.use_noise_map:
            sigma = torch.tensor(sigma).float().to(x.device)
            sigma = sigma.repeat(1, x.shape[1], x.shape[2], x.shape[3])
            x = torch.cat((x, sigma), dim=0)
        x = torch.unsqueeze(x, 0)
        x = self.model(x)
        x = torch.squeeze(x)
        x = torch.unsqueeze(x, 0)
        return x

    def to(self, device):
        self.model.to(device)
        return self


class GRUNetTVDenoiser(GRUNetDenoiser):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.tv_denoiser = TVDenoiser()

    def denoise(self, x, sigma):
        x1 = super().denoise(x, sigma)
        x2 = self.tv_denoiser.denoise(x, sigma*255)
        return (x1+x2)/2
