from functools import partial

import torch
import dphsir.solvers.fns.sisr as sisr
from dphsir.degrades import (BiCubicDownsample, ClassicalDownsample,
                             GaussianDownsample, UniformDownsample)
from dphsir.denoisers import UNetDenoiser, Augment
from dphsir.metric import mpsnr
from dphsir.solvers.base import ADMMSolver
import dphsir.solvers.params as param_settings
from dphsir.utils.io import load_mat

device = torch.device('cuda:0')
sf = 2
downsample = ClassicalDownsample(sf=sf, type=0)

path = 'Lehavim_0910-1717.mat'
data = load_mat(path)
gt = data['gt']
low = downsample(gt)


model_path = '/media/exthdd/laizeqiang/lzq/projects/hyper-pnp/DPIR/model_zoo/unet_qrnn3d.pth'
denoiser = UNetDenoiser(model_path).to(device)
denoise = Augment(denoiser)


init = partial(sisr.inits.interpolate, sf=sf, enable_shift_pixel=True)
prox = sisr.proxs.CloseFormedADMM(downsample.kernel(), sf=sf).to(device)


iter_num = 2
rhos, sigmas = param_settings.admm_log_descent(sigma=max(0.255/255., 0),
                                               iter_num=iter_num,
                                               modelSigma1=35, modelSigma2=10,
                                               w=1)
print(rhos)
solver = ADMMSolver(init, prox, denoise).to(device)
pred = solver.restore(low, iter_num=iter_num, rhos=rhos, sigmas=sigmas)


print(mpsnr(init(low), gt))
print(mpsnr(pred, gt))
# 47.5494
