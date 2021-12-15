from torchlight.metrics import mpsnr
from dphsir.utils.io import loadmat, show_hsi
import torch
from dphsir.denoisers.wrapper import UNetDenoiser
from functools import partial

import dphsir.solvers.fns.inpaint as task
from dphsir.solvers.base import ADMMSolver, HQSSolver
from dphsir.solvers.params import admm_log_descent

path = '/media/exthdd/laizeqiang/lzq/projects/hyper-pnp/DPIR/log/test_inpainting_file/mat/Lehavim_0910-1717.mat'
data = loadmat(path)
gt = data['gt']
low = data['low']
mask = data['mask'].astype('float')

model_path = '/media/exthdd/laizeqiang/lzq/projects/hyper-pnp/DPIR/model_zoo/unet_qrnn3d.pth'
device = torch.device('cuda:0')
denoiser = UNetDenoiser(model_path).to(device)

init = partial(task.init, mask=mask)
prox = task.Prox(mask).to(device)
denoise = denoiser.denoise

rhos, sigmas = admm_log_descent(sigma=max(0.255/255., 0),
                                iter_num=24,
                                modelSigma1=5, modelSigma2=4,
                                w=1,
                                lam=0.6)

solver = ADMMSolver(init, prox, denoise).to(device)
pred = solver.restore(low, iter_num=24)
print(pred.shape)

# %%
print(mpsnr(init(low), gt))
print(mpsnr(pred, gt))
