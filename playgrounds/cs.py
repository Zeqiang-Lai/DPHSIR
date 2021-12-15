from torchlight.metrics import mpsnr
from dphsir.utils.io import loadmat, show_hsi
import torch
from dphsir.denoisers.wrapper import UNetDenoiser
from functools import partial

import dphsir.solvers.fns.cs as cs
from dphsir.solvers.base import ADMMSolver, HQSSolver
from dphsir.solvers.params import admm_log_descent

path = '/home/laizeqiang/Desktop/lzq/projects/hyper-pnp/DPIR/testsets/icvl_small_ref/cs_casii/Lehavim_0910-1717.mat'
data = loadmat(path)
gt = data['gt']
low = data['low']
mask = data['mask']

model_path = '/media/exthdd/laizeqiang/lzq/projects/hyper-pnp/DPIR/model_zoo/unet_qrnn3d.pth'
device = torch.device('cuda:0')
denoiser = UNetDenoiser(model_path).to(device)

init = partial(cs.init, mask=mask)
prox = cs.Prox(mask).to(device)
denoise = denoiser

rhos, sigmas = admm_log_descent(sigma=max(0.255/255., 0),
                                iter_num=24,
                                modelSigma1=50, modelSigma2=45,
                                w=1)

solver = ADMMSolver(init, prox, denoise).to(device)
pred = solver.restore(low, iter_num=24, rhos=rhos, sigmas=sigmas)
print(pred.shape)

# %%
print(mpsnr(init(low), gt))
print(mpsnr(pred, gt))
