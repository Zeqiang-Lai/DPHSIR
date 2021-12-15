
import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp

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


class DeepTVDenoiser:
    def __init__(self, deep_denoise, tv_denoising=None,
                 deep_hypara_list=[40., 20., 10., 5.], tv_hypara_list=[10, 0.01]):
        self.deep_hypara_list = deep_hypara_list
        self.tv_hypara_list = tv_hypara_list
        self.tv_denoising = tv_denoising
        self.deep_denoise = deep_denoise

    def denoise(self, x):
        # x: 1,31,512,512
        deep_num = len(self.deep_hypara_list)
        tv_num = len(self.tv_hypara_list)
        deep_list = [self.deep_denoise(x, torch.tensor(level/255.).to(x.device)) for level in self.deep_hypara_list]
        deep_list = [tmp.squeeze().permute(1, 2, 0) for tmp in deep_list]

        tv_list = [self.tv_denoising(x.squeeze().permute(1, 2, 0), level, 5).clamp(0, 1) for level in self.tv_hypara_list]

        ffdnet_mat = np.stack(
            [x_ele[:, :, :].cpu().numpy().reshape(-1).astype(np.float64) for x_ele in deep_list],
            axis=0)
        tv_mat = np.stack(
            [x_ele[:, :, :].cpu().numpy().reshape(-1).astype(np.float64) for x_ele in tv_list],
            axis=0)
        w = cp.Variable(deep_num + tv_num)
        P = np.zeros((deep_num + tv_num, deep_num + tv_num))
        P[:deep_num, :deep_num] = ffdnet_mat @ ffdnet_mat.T
        P[:deep_num, deep_num:] = -ffdnet_mat @ tv_mat.T
        P[deep_num:, :deep_num] = -tv_mat @ ffdnet_mat.T
        P[deep_num:, deep_num:] = tv_mat @ tv_mat.T
        one_vector_ffdnet = np.ones((1, deep_num))
        one_vector_tv = np.ones((1, tv_num))
        objective = cp.quad_form(w, P)
        problem = cp.Problem(
            cp.Minimize(objective),
            [one_vector_ffdnet @ w[:deep_num] == 1,
                one_vector_tv @ w[deep_num:] == 1,
                w >= 0])
        problem.solve()
        w_value = w.value
        x_ffdnet, x_tv = 0, 0

        for idx in range(deep_num):
            x_ffdnet += w_value[idx] * deep_list[idx]
        for idx in range(tv_num):
            x_tv += w_value[idx + deep_num] * tv_list[idx]
        v = 0.5 * (x_ffdnet + x_tv)
        v = v.permute(2, 0, 1).unsqueeze(0)
        return v


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
        from .models.qrnn3d import qrnn3d, qrnn3d_masked
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


class UNetDenoiser(Denoiser):
    def __init__(self, model_path):
        from .models.qrnn3d import unet_masked_nobn
        model = unet_masked_nobn()
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


class UNetTVDenoiser(UNetDenoiser):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.tv_denoiser = TVDenoiser()

    def denoise(self, x, sigma):
        x1 = super().denoise(x, sigma)
        x2 = self.tv_denoiser.denoise(x, sigma*255)
        return (x1+x2)/2
