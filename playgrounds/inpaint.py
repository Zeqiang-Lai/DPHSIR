from functools import partial

import torch

import dphsir.solvers.fns.inpaint as task
from dphsir.degrades.inpaint import FastHyStripe
from dphsir.denoisers.wrapper import GRUNetDenoiser
from dphsir.solvers import callbacks
from dphsir.solvers.base import ADMMSolver
from dphsir.solvers.params import admm_log_descent
from dphsir.utils.io import loadmat
from dphsir.metrics import mpsnr

path = 'Lehavim_0910-1717.mat'
data = loadmat(path)
gt = data['gt']

degrade = FastHyStripe()
low, mask = degrade(gt)
mask = mask.astype('float')

device = torch.device('cuda:0')

model_path = 'unet_qrnn3d.pth'
denoiser = GRUNetDenoiser(model_path).to(device)

init = partial(task.inits.none, mask=mask)
prox = task.Prox(mask).to(device)
denoise = denoiser
solver = ADMMSolver(init, prox, denoise).to(device)

iter_num = 24
rhos, sigmas = admm_log_descent(sigma=max(0.255/255., 0),
                                iter_num=iter_num,
                                modelSigma1=5, modelSigma2=4,
                                w=1,
                                lam=0.6)

pred = solver.restore(low, iter_num=iter_num, rhos=rhos, sigmas=sigmas,
                      callbacks=[callbacks.ProgressBar(iter_num)])

print(pred.shape)
print(mpsnr(init(low), gt))
print(mpsnr(pred, gt))

# Expected: 74.88
