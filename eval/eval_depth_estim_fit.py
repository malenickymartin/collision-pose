import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from copy import deepcopy

args = argparse.ArgumentParser()
args.add_argument("--dataset", type=str, default="ycbv")
args.add_argument("--estimator", type=str, default="zoe")
args.add_argument("--scene", type=int, default=0)
args.add_argument("--image", type=int, default=0)
args = args.parse_args()

# Fitting functions
def fit_linear(depth_meas, depth_estim):
    s = np.nanmedian(depth_meas / depth_estim)
    print(f"Linear fit: s={s}")
    return s*depth_estim

def fit_affine(depth_meas, depth_estim):
    def loss(x):
        s = x[0]
        a = x[1]
        error = ((depth_estim*s + a) - depth_meas)**2
        return np.nanmedian(error)

    res = minimize(loss, [1, 0], method='Nelder-Mead')
    s = res.x[0]
    a = res.x[1]

    print(f"Optimization result: s={s}, a={a}")

    return depth_estim*s + a

#Filtering functions
def filter_depth(depth_meas, depth_estim):
    mask = depth_meas <= 1e-6
    depth_meas[mask] = np.nan
    depth_estim[mask] = np.nan
    return depth_meas, depth_estim

# Plot functions
def error_hist(depth_meas, depth_estim, depth_lin, depth_aff, num_bins=100):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].hist((depth_meas - depth_estim).flatten(), bins=num_bins, color='blue', alpha=0.7)
    axs[0].set_title(f'Depth Measurement - Depth Estimation')

    axs[1].hist((depth_meas - depth_lin).flatten(), bins=num_bins, color='green', alpha=0.7)
    axs[1].set_title(f'Depth Measurement - Linear Fit Estimation')

    axs[2].hist((depth_meas - depth_aff).flatten(), bins=num_bins, color='red', alpha=0.7)
    axs[2].set_title(f'Depth Measurement - Affine Fit Estimation')

    for ax in axs:
        ax.grid()
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Difference (mm)')
    plt.tight_layout()
    plt.show(block=False)

def error_heatmap(depth_meas, depth_estim, depth_lin, depth_aff):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axs[0].imshow(depth_meas - depth_estim, cmap='viridis')
    axs[0].set_title('Depth Measurement - Depth Estimation')
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(depth_meas - depth_lin, cmap='viridis')
    axs[1].set_title('Depth Measurement - Linear Fit Estimation')
    plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(depth_meas - depth_aff, cmap='viridis')
    axs[2].set_title('Depth Measurement - Affine Fit Estimation')
    plt.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.show()

def compare_methods(depth_meas):
    methods = ["zoe", "da", "m3d"]
    fig, axs = plt.subplots(1, len(methods), figsize=(15, 5))
    original_depth_meas = deepcopy(depth_meas)

    for i, method in enumerate(methods):
        depth_meas = deepcopy(original_depth_meas)
        estim_depth_path = scene_path / f"depth_{method}"
        depth_estim = np.array(Image.open(estim_depth_path / img_name))/10 # depth in mm
        _, depth_estim = filter_depth(depth_meas, depth_estim)
        depth_aff = fit_affine(depth_meas, depth_estim)
        im = axs[i].imshow(depth_meas - depth_aff, cmap='gray')
        axs[i].set_title(f"Error {method}")
        plt.colorbar(im, ax=axs[i])

    plt.tight_layout()
    plt.show()
    

dataset_path = Path("/local2/homes/malenma3/collision-pose/data/datasets") / args.dataset
scene_path = sorted(dataset_path.iterdir())[args.scene]
meas_depth_path = scene_path / "depth"
estim_depth_path = scene_path / f"depth_{args.estimator}"
img_name = sorted(meas_depth_path.iterdir())[args.image].name
depth_meas = np.array(Image.open(meas_depth_path / img_name))/10 # depth in mm
depth_estim = np.array(Image.open(estim_depth_path / img_name))/10 # depth in mm
compare_methods(depth_meas)
depth_meas, depth_estim = filter_depth(depth_meas, depth_estim)
depth_lin = fit_linear(depth_meas, depth_estim)
depth_aff = fit_affine(depth_meas, depth_estim)
error_hist(depth_meas, depth_estim, depth_lin, depth_aff)
error_heatmap(depth_meas, depth_estim, depth_lin, depth_aff)
print(f"Median error: Depth Estimation: {np.nanmedian(abs(depth_meas - depth_estim))} mm, Linear Fit: {np.nanmedian(abs(depth_meas - depth_lin))} mm, Affine Fit: {np.nanmedian(abs(depth_meas - depth_aff))} mm")
