from functools import partial
import munch

from dphsir.solvers import ADMMSolver, HQSSolver
from common import get_denoiser, restore

solvers = {'admm': ADMMSolver, 'hqs': HQSSolver}


def deblur(gt, device, cfg):
    import dphsir.solvers.fns.deblur as task
    from dphsir.degrades import GaussianBlur
    downsample = GaussianBlur()
    low = downsample(gt)
    init = task.inits.interpolate
    prox = task.proxs.CloseFormedADMM(downsample.kernel).to(device)

    denoiser = get_denoiser(cfg.denoiser).to(device)
    solver = solvers[cfg.solver](init, prox, denoiser).to(device)
    return low, init(low), solver


def sisr(gt, device, cfg):
    import dphsir.solvers.fns.sisr as task
    from dphsir.degrades import GaussianDownsample
    sf = 2
    downsample = GaussianDownsample(sf=sf)
    low = downsample(gt)
    init = partial(task.inits.interpolate, sf=sf, enable_shift_pixel=True)
    prox = task.proxs.CloseFormedADMM(downsample.kernel, sf=sf).to(device)

    denoiser = get_denoiser(cfg.denoiser).to(device)
    solver = solvers[cfg.solver](init, prox, denoiser).to(device)
    return low, init(low), solver


def cs(gt, device, cfg):
    import dphsir.solvers.fns.cs as task
    import dphsir.degrades.cs as cs
    degrade = cs.CASSI()
    low = degrade(gt)
    mask = degrade.mask.astype('float32')
    init = partial(task.init, mask=mask)
    prox = task.Prox(mask).to(device)

    denoiser = get_denoiser(cfg.denoiser).to(device)
    solver = solvers[cfg.solver](init, prox, denoiser).to(device)
    return low, init(low), solver


def inpaint(gt, device, cfg):
    import dphsir.solvers.fns.inpaint as task
    from dphsir.degrades.inpaint import FastHyStripe
    degrade = FastHyStripe()
    low, mask = degrade(gt)
    mask = mask.astype('float32')
    init = partial(task.inits.none, mask=mask)
    prox = task.Prox(mask).to(device)

    denoiser = get_denoiser(cfg.denoiser).to(device)
    solver = solvers[cfg.solver](init, prox, denoiser).to(device)
    return low, init(low), solver


def misr(gt, device, cfg):
    import dphsir.solvers.fns.misr as misr
    import dphsir.solvers.fns.sisr as sisr
    from dphsir.degrades import (HSI2RGB, BiCubicDownsample,
                                 GaussianDownsample, UniformDownsample)
    sf = 2
    spa_down = GaussianDownsample(sf=sf)
    spe_down = HSI2RGB()
    low = spa_down(gt)
    rgb = spe_down(gt)
    init = partial(sisr.inits.interpolate, sf=sf, enable_shift_pixel=True)
    prox_spa = sisr.proxs.CloseFormedADMM(spa_down.kernel, sf=sf).to(device)
    prox_spe = misr.SpeProx(spe_down.srf).to(device)

    denoiser = get_denoiser(cfg.denoiser).to(device)
    solver = misr.ADMMSolver(init, (prox_spe, prox_spa), denoiser).to(device)
    return (low, rgb), init(low), solver


if __name__ == '__main__':
    cfg = {
        'input_path': 'playgrounds/Lehavim_0910-1717.mat',
        'output_path': 'Lehavim_0910-1717_deblur',
        'denoiser': {
            'type': 'grunet',
            'model_path': 'playgrounds/grunet.pth',
        },
        'solver': 'admm',
        'device': 'cuda',
        'params': {
            'iter': 24,
            'sigma1': 30,
            'sigma2': 15,
            'w': 1,
            'lam': 0.23
        }
    }
    cfg = munch.munchify(cfg)

    restore(deblur, cfg)
