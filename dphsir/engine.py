from abc import abstractmethod
import os
import logging
import time
import scipy
import scipy.io

from ..utils import utils_logger
from ..utils import utils_image as util
from ..utils.ani import save_ani

from .degrades.general import AbstractDownsample, AffineTransform, GaussianNoise, HSI2RGB, PerspectiveTransform
from .solver import DenoiseSolver, InpaintingADMMPnPSolver, MISRADMMPnPSolver, PnPSISRSolver, CompressSensingADMMPnPSolver
from .metrics import psnr_b_max, ssim_qrnn3d,sam_qrnn3d, pnsr_qrnn3d

"""
There are two types of dataset, all in [W,H,C] format

1. gt
This type of dataset only contain ground truth image, and low quality image is generated while running the pipeline.
We assume gt is in the range of [0,1], and gt can be load with `scipy.io.loadmat`, and access with key='gt'. 

contains: gt(single)

2. low
This type of dataset contain anything we need to perform a restoration task. 
- In deblur and super resolution task, we need both ground truth and low quality image, 
  which should be able to be accessed with key 'gt' and 'low'.
- In inpainting task, we need an extra mask that can be accessed with key 'mask'.

contains: gt(single), low(single), [mask(uint8)]
"""

SAVE_CHANNEL = 20   # The channel we choosed to visualize HSI

def format_dict(d:dict):
    """ All value in dict should be floating numbers """
    items = ["'{}': {:.4f}".format(k, v) for k, v in d.items()]
    return '{' + ", ".join(items) + '}'

class Engine:
    def __init__(self, checkpoint_dir='log', save_img=False, save_mat=False, save_ani=False, override=False):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.logger = self._get_logger()
        self.metric_fns = {
                        #   'psnr': util.calculate_psnr_01,
                          'psnr': pnsr_qrnn3d,
                          'psnr_qrnn3d': util.psnr_qrnn3d,
                        #   'psnr_b_max': psnr_b_max,
                          'ssim_qrnn3d': ssim_qrnn3d,
                          'sam_qrnn3d': sam_qrnn3d,
                          }

        self.save_img = save_img
        if self.save_img:
            self.img_dir = os.path.join(self.checkpoint_dir, 'img')
            os.makedirs(self.img_dir, exist_ok=override)
        
        self.save_mat = save_mat
        if self.save_mat:
            self.mat_dir = os.path.join(self.checkpoint_dir, 'mat')
            os.makedirs(self.mat_dir, exist_ok=override)

        self.save_ani = save_ani
        if self.save_ani:
            self.ani_dir = os.path.join(self.checkpoint_dir, 'ani')
            os.makedirs(self.ani_dir, exist_ok=override)
        
    def _get_logger(self) -> logging.Logger:
        logger_name = 'logging'
        log_path = os.path.join(self.checkpoint_dir, logger_name+'.log')
        utils_logger.logger_info(logger_name, log_path=log_path)
        logger = logging.getLogger(logger_name)
        return logger 

    def test_file(self, file_path, input_type, degrade, noise_level=0, iter_num=24):
        metrics = self._test_single(file_path, input_type, degrade, noise_level, iter_num)

        name, _ = os.path.splitext(os.path.basename(file_path))
        self.logger.info('{}: {}'.format(name, format_dict(metrics)))

        return metrics 

    def test_dir(self, dir_path, input_type, degrade, noise_level=0, iter_num=24):
        self.logger.info('------ degrade: {} -------'.format(str(degrade)))
        
        avg_metrics = {name: [] for name in self.metric_fns.keys()}
        avg_metrics['time'] = []

        for idx, input_path in enumerate(util.get_image_paths(dir_path)):
            metrics = self._test_single(input_path, input_type, degrade, noise_level, iter_num)

            name, _ = os.path.splitext(os.path.basename(input_path))
            self.logger.info('[{}]({}): {}'.format(idx+1, name, format_dict(metrics)))

            for name, value in metrics.items():
                avg_metrics[name].append(value)
        
        avg_metrics = {k: sum(v)/len(v) for k, v in avg_metrics.items()}
        self.logger.info('--- average ---: ' + format_dict(avg_metrics))
        
        return avg_metrics

    def _test_single(self, input_path, input_type, degrade, noise_level=0, iter_num=24):
        start = time.time()
        results = self._run_single(input_path, input_type, degrade, noise_level=noise_level, iter_num=iter_num)
        end = time.time()

        metrics = self._compute_metrics(results['mat']['pred'], results['mat']['gt'])

        name, _ = os.path.splitext(os.path.basename(input_path))
        if self.save_img: self._save_img(results['img'], name)
        if self.save_mat: self._save_mat(results['mat'], name)
        if self.save_ani: self._save_ani(results['ani'], name)
        
        metrics['time'] = end-start
        return metrics

    @abstractmethod
    def _run_single(self, input_path, input_type, degrade, noise_level=0, iter_num=24):
        pass

    def _compute_metrics(self, img_E, img_H):
        results = {}
        for name, fn in self.metric_fns.items():
            results[name] = fn(img_E, img_H)
        return results

    def _save_img(self, results:dict, name):
        for k, v in results.items():
            util.imsave(v, os.path.join(self.img_dir, name+'_'+k+'.png'))

    def _save_mat(self, results:dict, name):
        mat_path = os.path.join(self.mat_dir, name+'.mat')
        scipy.io.savemat(mat_path, results)

    def _save_ani(self, img_seq, name):
        save_ani(img_seq, os.path.join(self.ani_dir, name+'.gif'), fps=5)
    
class SISREngine(Engine):
    def __init__(self, solver: PnPSISRSolver, checkpoint_dir, save_img=False, save_mat=False, override=False):
        super().__init__(checkpoint_dir=checkpoint_dir, save_img=save_img, save_mat=save_mat, override=override)
        self.solver = solver

    def _run_single(self, input_path, input_type, degrade:AbstractDownsample, noise_level=0,iter_num=24):
        awgn = GaussianNoise(sigma=noise_level)
        data = scipy.io.loadmat(input_path)
        img_H = data['gt'] if input_type=='gt' else data['gt']
        img_L = awgn(degrade(img_H)) if input_type == 'gt' else data['low']
        img_E = self.solver.restore(img_L, 
                                    sf=degrade.scale_factor(), 
                                    k=degrade.kernel(), 
                                    classical_degradation=degrade.is_classical_downsample(), 
                                    iter_num=iter_num)

        imgs = {'C_{}_H'.format(SAVE_CHANNEL): util.single2uint(img_H)[:,:,SAVE_CHANNEL], 
                'C_{}_L'.format(SAVE_CHANNEL): util.single2uint(img_L)[:,:,SAVE_CHANNEL], 
                'C_{}_E'.format(SAVE_CHANNEL): util.single2uint(img_E)[:,:,SAVE_CHANNEL]}
        mats = {'gt': img_H, 'low': img_L, 'pred': img_E}
        return {'img': imgs, 'mat': mats}

class DeblurEngine(SISREngine):
    pass


class MISREngine(Engine):
    def __init__(self, solver: MISRADMMPnPSolver, checkpoint_dir, save_img=False, save_mat=False, override=False):
        super().__init__(checkpoint_dir=checkpoint_dir, save_img=save_img, save_mat=save_mat, override=override)
        self.solver = solver

    def _run_single(self, input_path, input_type, degrade:AbstractDownsample, noise_level=0,iter_num=24):
        awgn = GaussianNoise(sigma=noise_level)
        data = scipy.io.loadmat(input_path)
        img_H = data['gt'] if input_type=='gt' else data['gt']
        img_L = awgn(degrade(img_H)) if input_type == 'gt' else data['low']
        
        img_L_spe = HSI2RGB()(img_H)
        # img_L_spe = PerspectiveTransform(shift=5)(img_L_spe)
        
        img_E = self.solver.restore(img_L, 
                                    img_L_spe,
                                    T=HSI2RGB().srf,
                                    sf=degrade.scale_factor(), 
                                    k=degrade.kernel(), 
                                    classical_degradation=degrade.is_classical_downsample(), 
                                    iter_num=iter_num)

        imgs = {'C_{}_H'.format(SAVE_CHANNEL): util.single2uint(img_H)[:,:,SAVE_CHANNEL], 
                'C_{}_L'.format(SAVE_CHANNEL): util.single2uint(img_L)[:,:,SAVE_CHANNEL], 
                'C_{}_E'.format(SAVE_CHANNEL): util.single2uint(img_E)[:,:,SAVE_CHANNEL],
                'C_{}_SPE'.format(SAVE_CHANNEL): util.single2uint(img_L_spe),
                }
        mats = {'gt': img_H, 'low': img_L, 'pred': img_E}
        return {'img': imgs, 'mat': mats}

class InpaintingEngine(Engine):
    def __init__(self, solver: InpaintingADMMPnPSolver, checkpoint_dir, save_img=False, save_mat=False, override=False):
        super().__init__(checkpoint_dir=checkpoint_dir, save_img=save_img, save_mat=save_mat, override=override)
        self.solver = solver
    
    def _run_single(self, input_path, input_type, degrade, noise_level=0, iter_num=24):
        awgn = GaussianNoise(sigma=noise_level)
        data = scipy.io.loadmat(input_path)
        if input_type == 'gt':
            img_H = data['gt']
            img_L = img_H
            img_L = awgn(img_L)
            img_L, mask = degrade(img_L)
        else:
            img_H = data['gt']
            img_L = data['low']
            mask = data['mask']
            
        img_E = self.solver.restore(img_L, 
                                    mask=mask, 
                                    iter_num=iter_num)
        

        imgs = {'C_{}_H'.format(SAVE_CHANNEL): util.single2uint(img_H)[:,:,SAVE_CHANNEL], 
                'C_{}_L'.format(SAVE_CHANNEL): util.single2uint(img_L)[:,:,SAVE_CHANNEL], 
                'C_{}_E'.format(SAVE_CHANNEL): util.single2uint(img_E)[:,:,SAVE_CHANNEL]}
        mats = {'gt': img_H, 'low': img_L, 'pred': img_E, 'mask': mask}
        return {'img': imgs, 'mat': mats}

class CSEngine(Engine):
    def __init__(self, solver: CompressSensingADMMPnPSolver, checkpoint_dir, 
                 save_img=False, save_mat=False, save_ani=False, override=False):
        super().__init__(checkpoint_dir=checkpoint_dir, save_img=save_img, save_mat=save_mat, save_ani=save_ani, override=override)
        self.solver = solver
    
    def _run_single(self, input_path, input_type, degrade, noise_level=0, iter_num=24):
        awgn = GaussianNoise(sigma=noise_level)
        data = scipy.io.loadmat(input_path)
        if input_type == 'gt':
            img_H = data['gt']
            img_L = degrade(img_H)
            mask = degrade.mask
            img_L = awgn(img_L)
        else:
            img_H = data['orig'] / 255
            img_L = data['meas'] / 255
            mask = data['mask']
            
        img_E, img_seq = self.solver.restore(img_L, 
                                    mask=mask, 
                                    iter_num=iter_num)
        
        import numpy as np
        imgs = {'_L': util.single2uint(img_L/np.max(img_L)), 
                'C_{}_H'.format(SAVE_CHANNEL): util.single2uint(img_H)[:,:,SAVE_CHANNEL], 
                'C_{}_E'.format(SAVE_CHANNEL): util.single2uint(img_E)[:,:,SAVE_CHANNEL]}
        mats = {'gt': img_H, 'low': img_L, 'pred': img_E, 'mask': mask}
        return {'img': imgs, 'mat': mats, 'ani': img_seq}

class DenoiseEngine(Engine):
    def __init__(self, solver:DenoiseSolver, checkpoint_dir, low_key='input',
                 save_img=False, save_mat=False, override=False):
        super().__init__(checkpoint_dir=checkpoint_dir, save_img=save_img, save_mat=save_mat, override=override)
        self.solver = solver
        self.low_key = low_key
    
    def _run_single(self, input_path, input_type, degrade, noise_level=0, iter_num=24):
        awgn = GaussianNoise(sigma=noise_level)
        data = scipy.io.loadmat(input_path)
        if input_type == 'gt':
            img_H = data['gt']
            img_L = awgn(img_H)
        else:
            # img_H = data['gt'][192:320,192:320,:]
            # img_L = data[self.low_key][192:320,192:320,:]
            img_H = data['gt']
            img_L = data[self.low_key]
            noise_level = data['sigma'] / 255

        img_E = self.solver.restore(img_L, noise_level=noise_level)
        
        imgs = {'C_{}_H'.format(SAVE_CHANNEL): util.single2uint(img_H)[:,:,SAVE_CHANNEL], 
                'C_{}_L'.format(SAVE_CHANNEL): util.single2uint(img_L)[:,:,SAVE_CHANNEL], 
                'C_{}_E'.format(SAVE_CHANNEL): util.single2uint(img_E)[:,:,SAVE_CHANNEL]}
        mats = {'gt': img_H, 'low': img_L, 'pred': img_E}
        return {'img': imgs, 'mat': mats}
