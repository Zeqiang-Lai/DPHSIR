import torch

from dphsir.denoisers import GRUNetDenoiser
from dphsir.metrics import mpsnr
from dphsir.solvers.utils import single2tensor4, tensor2single
from dphsir.utils.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os

import sys

def main():
    case_num = sys.argv[1] if len(sys.argv) > 1 else '8'
    path = f'data/JasperRidge/case{case_num}/data.mat'
    if not os.path.exists(path):
        print(f"Error: Could not find {path}")
        return

    print(f"Loading data from {path}...")
    data = loadmat(path)
    
    gt = data.get('gt')
    low = data.get('input')

    if gt is None or low is None:
        print("Error: The .mat file must contain both 'gt' and 'input' keys.")
        return

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_path = 'grunet.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
        
    print("Loading GRUNet model...")
    denoiser = GRUNetDenoiser(model_path).to(device)

    # In denoise.py, the noise standard deviation (sigma) is typically estimated or known.
    # The paper often assumes some baseline sigma for noisy data or estimates it.
    # Assuming the input noise level roughly requires sigma around 30/255 for the network.
    sigma = 30 / 255.0

    print("Running denoiser...")
    tmp = single2tensor4(low).to(device)
    
    # Pad to multiple of 16
    b, c, h, w = tmp.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h > 0 or pad_w > 0:
        tmp = torch.nn.functional.pad(tmp, (0, pad_w, 0, pad_h), mode='reflect')
    
    with torch.no_grad():
        # Denoise expects a 5D tensor: [B, 1, C, H, W] for single images, or 4D depending on how denoiser wraps it.
        # But GRUNet expects [B, C_in, Bands, W, H], wait.
        # Let's check single2tensor4 again. It returns (1, C, H, W) where C is bands.
        # The wrapper models usually expect [B, C, H, W] then they rearrange it inside.
        pred = denoiser(tmp, sigma)
        
    if pad_h > 0 or pad_w > 0:
        pred = pred[:, :, :h, :w]
        
    pred = tensor2single(pred)

    print("\n--- Results ---")
    print(f"Shape: {pred.shape}")
    print(f"Initial Noisy Image PSNR: {mpsnr(low, gt):.2f} dB")
    print(f"Denoised Image PSNR:      {mpsnr(pred, gt):.2f} dB")

    # Plot results
    # Picking a specific band (e.g., band 20) for visualization
    band = 20
    if pred.shape[2] > band:
        img = [i[:, :, band] for i in [low, pred, gt]]
        
        plt.figure(figsize=(15, 5))
        titles = ['Noisy Input', 'Denoised (GRUNet)', 'Ground Truth']
        for i, (image, title) in enumerate(zip(img, titles)):
            plt.subplot(1, 3, i + 1)
            plt.imshow(image, cmap='gray')
            plt.title(title)
            plt.axis('off')
            
        plt.tight_layout()
        out_plot = f'output_jasper_case{case_num}_band20.png'
        plt.savefig(out_plot)
        print(f"\nSaved visualization to {out_plot}")
    else:
        print(f"\nWarning: Band {band} does not exist in the output for plotting.")

if __name__ == '__main__':
    main()
