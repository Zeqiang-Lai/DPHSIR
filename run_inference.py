import os
import argparse
import scipy.io as sio
import torch
import numpy as np

# Add base path if needed
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dphsir.denoisers import GRUNetDenoiser
from dphsir.metrics import mpsnr, mssim, sam
from dphsir.solvers.utils import single2tensor4

def main():
    parser = argparse.ArgumentParser(description='Run DPHSIR inference')
    parser.add_argument('--input_path', type=str, default='data/JasperRidge/case8/data.mat', help='Path to the input data.mat file')
    parser.add_argument('--output_dir', type=str, default='result', help='Directory to save the restored results')
    parser.add_argument('--model_type', type=str, default='grunet', choices=['grunet'], help='Type of model to use')
    parser.add_argument('--norm', type=str, default='clipped', choices=['minmax', 'clipped', 'raw'], help='Normalization method: minmax (scale 0-1), clipped (clamp 0-1), or raw (none)')
    parser.add_argument('--sigma', type=float, default=30.0/255.0, help='Noise standard deviation for the denoiser')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 1. Load model
    print(f'Loading model ({args.model_type})...')
    
    checkpoint_map = {
        'grunet': 'grunet.pth'
    }
    
    if args.model_type not in checkpoint_map:
        print(f"Error: Unknown model type {args.model_type}")
        return
        
    model_path = checkpoint_map[args.model_type]
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
        
    denoiser = GRUNetDenoiser(model_path).to(device)

    # 2. Load dataset
    print(f'Loading dataset from {args.input_path}...')
    if not os.path.exists(args.input_path):
        print(f"Error: Dataset file not found at {args.input_path}")
        return
        
    mat = sio.loadmat(args.input_path)
    
    # Try different key names for input
    if 'input' in mat:
        noisy_hsi = mat['input']
    elif 'HSI_noisy' in mat:
        noisy_hsi = mat['HSI_noisy']
    else:
        print("Error: Could not find 'input' or 'HSI_noisy' in the mat file.")
        return

    # Try different key names for GT
    has_gt = False
    gt_hsi = None
    if 'gt' in mat:
        gt_hsi = mat['gt']
        has_gt = True
    elif 'hsi_gt' in mat:
        gt_hsi = mat['hsi_gt']
        has_gt = True
        
    print('Input shape (H,W,C):', noisy_hsi.shape)

    # Prepare input tensor
    tmp = single2tensor4(noisy_hsi).to(device)

    # Apply normalization ONLY to input tensor
    print(f'Applying {args.norm} normalization to input...')
    if args.norm == 'minmax':
        t_min, t_max = tmp.min(), tmp.max()
        if t_max > t_min:
            tmp = (tmp - t_min) / (t_max - t_min)
    elif args.norm == 'clipped':
        tmp = torch.clamp(tmp, 0.0, 1.0)
    elif args.norm == 'raw':
        pass

    # Pad to multiple of 16
    b, c, h, w = tmp.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h > 0 or pad_w > 0:
        tmp = torch.nn.functional.pad(tmp, (0, pad_w, 0, pad_h), mode='reflect')

    # 3. Inference
    print('Running inference...')
    with torch.no_grad():
        pred = denoiser(tmp, args.sigma)

    if pad_h > 0 or pad_w > 0:
        pred = pred[:, :, :h, :w]
        
    # Convert tensor back to numpy (H, W, C)
    pred_hsi = pred.detach().cpu().squeeze().float().numpy()
    if pred_hsi.ndim == 3:
        pred_hsi = np.transpose(pred_hsi, (1, 2, 0))

    # 4. Calculate metrics
    if has_gt:
        print('Calculating metrics...')
        psnr_val = mpsnr(pred_hsi, gt_hsi)
        ssim_val = mssim(pred_hsi, gt_hsi, data_range=1.0)
        sam_val = sam(pred_hsi, gt_hsi)
        print(f'Results - MPSNR: {psnr_val:.4f}, MSSIM: {ssim_val:.4f}, SAM: {sam_val:.4f}')
    else:
        print('Warning: Could not find ground truth (gt) in the mat file. Skipping metrics.')
        gt_hsi = np.zeros_like(noisy_hsi)

    # 5. Save MAT
    print('Saving MAT file...')
    os.makedirs(args.output_dir, exist_ok=True)
    
    mat_name = f'{args.model_type}_{args.norm}_{args.sigma}.mat'
    save_path = os.path.join(args.output_dir, mat_name)
    
    # Save as MATLAB struct dict for params
    params_dict = {
        'norma': args.norm,
        'model_type': args.model_type,
        'sigma': args.sigma
    }
    
    sio.savemat(save_path, {
        'HSI_restored': pred_hsi, 
        'HSI_clean': gt_hsi, 
        'HSI_noisy': noisy_hsi,
        'params': params_dict,
    })
    print('Saved to:', save_path)

if __name__ == '__main__':
    main()
