import torch
from dphsir.degrades import GaussianNoise
from dphsir.denoisers import  UNetDenoiser
from dphsir.metric import mpsnr
from dphsir.solvers.utils import single2tensor4, tensor2single
from dphsir.utils.io import loadmat


path = 'Lehavim_0910-1717.mat'
data = loadmat(path)
gt = data['gt']

sigma = 30 / 255
awgn = GaussianNoise(sigma)
low = awgn(gt)


device = torch.device('cuda:0')
model_path = 'unet_qrnn3d.pth'
denoiser = UNetDenoiser(model_path).to(device)

tmp = single2tensor4(low).to(device)
pred = denoiser(tmp, sigma)
pred = tensor2single(pred)

print(pred.shape)
print(mpsnr(low, gt))
print(mpsnr(pred, gt))

import matplotlib.pyplot as plt
import numpy as np
img = [i[:,:,20] for i in [low, pred, gt]]
plt.imshow(np.hstack(img))
plt.show()