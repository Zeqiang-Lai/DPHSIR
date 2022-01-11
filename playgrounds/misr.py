from functools import partial

import dphsir.solvers.fns.misr as misr
import dphsir.solvers.fns.sisr as sisr
import torch
from dphsir.degrades import (HSI2RGB, BiCubicDownsample, ClassicalDownsample,
                             GaussianDownsample, UniformDownsample)
from dphsir.denoisers.wrapper import UNetDenoiser
from dphsir.solvers import callbacks
from dphsir.solvers.params import admm_log_descent
from dphsir.utils.io import loadmat
from torchlight.metrics import mpsnr

path = 'Lehavim_0910-1717.mat'
data = loadmat(path)
gt = data['gt']

sf = 2
spa_down = ClassicalDownsample(sf=sf, type=0)
spe_down = HSI2RGB()
low = spa_down(gt)
rgb = spe_down(gt)

device = torch.device('cuda:0')
model_path = 'unet_qrnn3d.pth'
denoiser = UNetDenoiser(model_path).to(device)

init = partial(sisr.inits.interpolate, sf=sf, enable_shift_pixel=True)
prox_spa = sisr.proxs.CloseFormedADMM(spa_down.kernel(), sf=sf).to(device)
prox_spe = misr.SpeProx(spe_down.srf).to(device)
denoise = denoiser
solver = misr.ADMMSolver(init, prox_spe, prox_spa, denoise).to(device)

rhos, sigmas = admm_log_descent(sigma=max(0.255/255., 0),
                                iter_num=24,
                                modelSigma1=35, modelSigma2=10,
                                w=1)

pred = solver.restore((low, rgb), iter_num=24, rhos=rhos, sigmas=sigmas,
                      callbacks=[callbacks.ProgressBar(24)])

print(pred.shape)
print(mpsnr(init(low), gt))
print(mpsnr(pred, gt))
# Expect: 59.7401
