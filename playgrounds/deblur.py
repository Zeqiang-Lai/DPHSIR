from functools import partial

import dphsir.solvers.fns.sisr as sisr
import torch
from dphsir.degrades import GaussianBlur
from dphsir.denoisers import Augment, GRUNetDenoiser
from dphsir.metrics import mpsnr
from dphsir.solvers import callbacks, ADMMSolver
from dphsir.solvers.params import admm_log_descent
from dphsir.utils.io import loadmat

# ------------------------------------- #
#                Data                   #
# ------------------------------------- #

path = 'Lehavim_0910-1717.mat'
data = loadmat(path)
gt = data['gt']

sf = 1
downsample = GaussianBlur()
low = downsample(gt)

# ------------------------------------- #
#                Init                   #
# ------------------------------------- #

device = torch.device('cuda:0')

# Create denoiser
model_path = 'unet_qrnn3d.pth'
denoiser = GRUNetDenoiser(model_path).to(device)
denoiser = Augment(denoiser)

# Create solver
init = partial(sisr.inits.interpolate, sf=sf, enable_shift_pixel=True)
prox = sisr.proxs.CloseFormedADMM(downsample.kernel(), sf=sf).to(device)
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

import matplotlib.pyplot as plt
import numpy as np
img = [i[:,:,20] for i in [init(low), pred, gt]]
plt.imshow(np.hstack(img))
plt.show()
# Expect: 55.10
