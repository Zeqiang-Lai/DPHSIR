from functools import partial

import torch

import dphsir.solvers.fns.sisr as sisr
from dphsir.degrades import (BiCubicDownsample, GaussianDownsample, UniformDownsample)
from dphsir.denoisers import Augment, GRUNetDenoiser
from dphsir.metrics import mpsnr
from dphsir.solvers import callbacks
from dphsir.solvers.base import ADMMSolver
from dphsir.solvers.params import admm_log_descent
from dphsir.utils.io import loadmat

# ------------------------------------- #
#                Data                   #
# ------------------------------------- #

path = 'Lehavim_0910-1717.mat'
data = loadmat(path)
gt = data['gt']

sf = 2
downsample = GaussianDownsample(sf=sf)
low = downsample(gt)

# ------------------------------------- #
#                Init                   #
# ------------------------------------- #

device = torch.device('cuda:0')

# Create denoiser
model_path = 'grunet.pth'
denoiser = GRUNetDenoiser(model_path).to(device)
denoiser = Augment(denoiser)

# Create solver
init = partial(sisr.inits.interpolate, sf=sf, enable_shift_pixel=True)
prox = sisr.proxs.CloseFormedADMM(downsample.kernel, sf=sf).to(device)
denoise = denoiser
solver = ADMMSolver(init, prox, denoise).to(device)

# ------------------------------------- #
#                Solve                  #
# ------------------------------------- #

iter_num = 24

# Genreate parameters for ADMM
rhos, sigmas = admm_log_descent(sigma=max(0.255/255., 0),
                                iter_num=iter_num,
                                modelSigma1=35, modelSigma2=10,
                                w=1)
# Run the iterations
pred = solver.restore(low, iter_num=iter_num, rhos=rhos, sigmas=sigmas,
                      callbacks=[callbacks.ProgressBar(iter_num)])

# ------------------------------------- #
#                Show                   #
# ------------------------------------- #

print(pred.shape)
print(mpsnr(init(low), gt))
print(mpsnr(pred, gt))

# Expect: 47.5494
