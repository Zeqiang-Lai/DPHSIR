#%%
from dphsir.degrades.general import ClassicalDownsample, BiCubicDownsample, GaussianDownsample, UniformDownsample
from dphsir.utils.io import loadmat, show_hsi

import numpy as np
# %%
path = 'Lehavim_0910-1717.mat'
data = loadmat(path)
print(data.keys())
# %%
gt = data['gt']
print(gt.shape)
# %%
downsample = GaussianDownsample(sf=4)
low = downsample(gt)
print(low.shape)
# %%
show_hsi(low)
# %%
show_hsi(gt)
# %%
