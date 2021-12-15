import numpy as np
import torch


def single2tensor4(img):
    """ convert single ndarray (H,W,C) to 4-D torch tensor (B,C,H,W)"""
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)


def tensor2single(img):
    """ convert torch tensor (B,C,H,W) to single ndarray (H,W,C)"""
    img = img.detach().cpu()
    img = img.squeeze().float().numpy().clip(0, 1)
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return img


class ConvergeChecker:
    def __init__(self, tol=1e-4):
        self.tol = tol

    def setup(self, **kwargs):
        self.last = self.convert(kwargs)

    def is_converged(self, **kwargs):
        current = self.convert(kwargs)
        avg_ratio = 0
        for k in current.keys():
            avg_ratio += np.mean(np.abs((current[k].flatten()-self.last[k].flatten())))
        avg_ratio /= len(current)
        return avg_ratio <= self.tol

    def convert(self, vars):
        return {k: v.detach().cpu().numpy() for k, v in vars.items()}
