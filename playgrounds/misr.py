from torchlight.metrics import mpsnr
from functools import partial

import dphsir.solvers.fns.sisr as sisr
import dphsir.solvers.fns.misr as misr
import torch
from dphsir.degrades import (BiCubicDownsample, ClassicalDownsample,
                             GaussianDownsample, UniformDownsample, HSI2RGB)
from dphsir.denoisers.wrapper import UNetDenoiser
from dphsir.solvers.base import ADMMSolver
from dphsir.solvers.params import admm_log_descent
from dphsir.utils.io import loadmat, show_hsi


sf = 2

path = '/media/exthdd/laizeqiang/lzq/projects/hyper-pnp/DPIR/log/test_misr/mat/Lehavim_0910-1717.mat'
data = loadmat(path)
gt = data['gt']
downsample = ClassicalDownsample(sf=sf, type=0)
low = downsample(gt)
img_L_spe = HSI2RGB()(gt)


model_path = '/media/exthdd/laizeqiang/lzq/projects/hyper-pnp/DPIR/model_zoo/unet_qrnn3d.pth'
device = torch.device('cuda:0')
denoiser = UNetDenoiser(model_path).to(device)

init = partial(sisr.inits.interpolate, sf=sf, enable_shift_pixel=True)
prox_spa = sisr.proxs.CloseFormedADMM(downsample.kernel(), sf=sf).to(device)
prox_spe = misr.SpeProx(HSI2RGB().SPE).to(device)
denoise = denoiser

rhos, sigmas = admm_log_descent(sigma=max(0.255/255., 0),
                             iter_num=24,
                             modelSigma1=35, modelSigma2=10,
                             w=1)

solver = misr.ADMMSolver(init, prox_spe, prox_spa, denoise).to(device)
pred = solver.restore((low, img_L_spe), iter_num=24, rhos=rhos, sigmas=sigmas)
print(pred.shape)


print(mpsnr(init(low), gt))
print(mpsnr(pred, gt))
# 59.7401
