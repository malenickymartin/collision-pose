import time
from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm
import pinocchio as pin
from scene import read_poses_pickle

import pydiffcol
from pydiffcol.utils import (
    select_strategy
)
from pydiffcol.utils_render import create_visualizer

from scene import DiffColScene, draw_scene, poses_equilateral, poses_tetraedron, read_poses_pickle, SelectStrategyConfig
from optim import (
    perception_res_grad,
    update_est
)
from spatial import perturb_se3

MESHCAT_VIS = True

SEED = 13
pin.seed(SEED)
np.random.seed(SEED)
pydiffcol.set_seed(SEED)

# LOAD blenderproc/pybullet simulation
# TODO: include path to mesh or shape spec
####################
wMo_lst, wMc = read_poses_pickle('scene.pkl')
X_meas = perturb_se3(wMo_lst, sig_t=0.2, sig_o=0.0)
X_init = deepcopy(X_meas)
# X_init = perturb_se3(wMo_lst, sig_t=0.5, sig_o=0.5)
####################

# # CREATE simple scene with 3-4 objects
# #####################
# # equilateral config
# # wMo_lst = poses_equilateral(d=2.1)
# # X_meas = poses_equilateral(d=1.5)
# wMo_lst = poses_tetraedron(d=2.1)
# X_meas = poses_tetraedron(d=1.5)
# X_init = X_meas
# #####################

N_SHAPES = len(wMo_lst)
path_objs = N_SHAPES*["meshes/icosphere.obj"]

path_stat_objs = ["eval/data/floor.ply", "eval/data/floor.ply"]
wMs_lst = [pin.SE3(np.eye(4)), pin.SE3(np.eye(4))]
wMs_lst[0].translation[2] = 1.75
wMs_lst[1].rotation = np.array([[np.cos(np.pi/3),0,np.sin(np.pi/3)],[0,1,0],[-np.sin(np.pi/3),0,np.cos(np.pi/3)]])

dc_scene = DiffColScene(path_objs, path_stat_objs, wMs_lst)
    
args = SelectStrategyConfig()
args.noise = 1e-2
args.num_samples = 100
col_req, col_req_diff = select_strategy(args)

# Ref: https://cs231n.github.io/neural-networks-3/#sgd
# method = 'GD'
# method = 'MGD'
# method = 'NGD'
# method = 'adagrad'
# method = 'rmsprop'
method = 'adam'   # Seems to be the best

N_step = 20
cost_c_lst, grad_c_norm = [], []
cost_c_stat_lst, grad_c_stat_norm = [], []
cost_pt_lst, cost_po_lst, grad_p_norm = [], [], []
cost_pt_gt_lst, cost_po_gt_lst = [], []
X = deepcopy(X_init)
X_lst = []

# All
learning_rate = 0.01

# Momentum param MGD, NGD
mu = 0.90

# adagrad
eps_adagrad = 1e-8

# RMSprop
decay_rmsprop = 0.99
eps_rmsprop = 1e-8

# adam
beta1_adam = 0.99
beta2_adam = 0.999
eps_adam = 1e-8

cache_ada = np.zeros(6*N_SHAPES)
m_adam = np.zeros(6*N_SHAPES)
v_adam = np.zeros(6*N_SHAPES)
dx = np.zeros(6*N_SHAPES)
scale_grad_col = 1
# 
for i in tqdm(range(N_step)):
    X_lst.append(deepcopy(X))
    if i % 100 == 0: 
        print(f'i/N_step: {i}/{N_step}') 
        print('learning_rate: ', learning_rate)
        learning_rate /= 2

    if method == 'NGD':
        # Evaluate gradient at a look-ahead state
        X_eval = update_est(X, mu*dx)
    else:
        # Evaluate gradient at current state
        X_eval = X

    cost_c_obj, grad_c_obj = dc_scene.compute_diffcol(X_eval, col_req, col_req_diff)
    cost_c_stat, grad_c_stat = dc_scene.compute_diffcol_static(X_eval, col_req, col_req_diff)
    cost_c = cost_c_obj + cost_c_stat
    grad_c = grad_c_obj + grad_c_stat
    # cost_c, grad_c = 0.0, np.zeros(dx.shape)
    res_p, grad_p = perception_res_grad(X_eval, X_meas)
    # res_p, grad_p = np.zeros(dx.shape), np.zeros(dx.shape)
    grad = scale_grad_col*grad_c + grad_p

    if method == 'GD':
        dx = -learning_rate*grad

    if method in ['MGD', 'NGD']:
        dx = mu*dx - learning_rate*grad

    if method == 'adagrad':
        cache_ada += grad**2
        dx = -learning_rate * grad / (np.sqrt(cache_ada)+eps_adagrad)

    if method == 'rmsprop':
        cache_ada += decay_rmsprop*cache_ada + (1 - decay_rmsprop)*grad**2
        dx = -learning_rate * grad / (np.sqrt(cache_ada)+eps_rmsprop)

    if method == 'adam':
        m_adam = beta1_adam*m_adam + (1-beta1_adam)*grad
        v_adam = beta2_adam*v_adam + (1-beta2_adam)*(grad**2)
        dx = - learning_rate * m_adam / (np.sqrt(v_adam) + eps_adam)

    # state update
    X = update_est(X, dx)

    # Logs
    cost_c_lst.append(cost_c_obj)
    cost_c_stat_lst.append(cost_c_stat)
    grad_c_norm.append(norm(grad_c_obj))
    grad_c_stat_norm.append(norm(grad_c_stat))
    
    res2cost = lambda r: 0.5*sum(r**2)

    cost_pt, cost_po = res2cost(res_p[:3]), res2cost(res_p[3:])
    cost_pt_lst.append(cost_pt)
    cost_po_lst.append(cost_po)
    grad_p_norm.append(norm(grad_p))
    res_p_gt, _ = perception_res_grad(X, wMo_lst)
    cost_pt_gt, cost_po_gt = res2cost(res_p_gt[:3]), res2cost(res_p_gt[3:])
    cost_pt_gt_lst, cost_po_gt_lst
    cost_pt_gt_lst.append(cost_pt_gt)
    cost_po_gt_lst.append(cost_po_gt)



import matplotlib.pyplot as plt
steps = np.arange(len(cost_c_lst))

fig, ax = plt.subplots(2, 2)
ax[0,0].plot(steps, cost_c_lst)
ax[0,0].set_title('cost collision')
ax[1,0].plot(steps, grad_c_norm)
ax[1,0].set_title('grad norm collision')
ax[0,1].plot(steps, cost_c_stat_lst)
ax[0,1].set_title('cost collision static')
ax[1,1].plot(steps, grad_c_stat_norm)
ax[1,1].set_title('grad norm collision static')
fig.legend()

fig, ax = plt.subplots(3)
ax[0].plot(steps, cost_pt_lst, label='meas')
ax[0].plot(steps, cost_pt_gt_lst, label='gt')
ax[0].set_ylabel('err_t [m]')
ax[2].set_title('cost translation')
ax[0].legend()
ax[1].plot(steps, cost_po_lst, label='meas')
ax[1].plot(steps, cost_po_gt_lst, label='gt')
ax[1].set_ylabel('err_o [rad]')
ax[2].set_title('cost orientation')
ax[1].legend()
ax[2].plot(steps, grad_p_norm)
ax[2].set_title('grad norm perception')



if MESHCAT_VIS:
    print('Create vis')
    vis = create_visualizer()
    cost_c, grad_c = dc_scene.compute_diffcol(wMo_lst, col_req, col_req_diff)
    input("Continue?")
    print('GT!')
    draw_scene(vis, dc_scene.shapes, dc_scene.stat_shapes, wMo_lst, wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat, render_faces=False)
    time.sleep(4)
    print('measured!')
    draw_scene(vis, dc_scene.shapes, dc_scene.stat_shapes, X_meas, wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat, render_faces=False)
    time.sleep(4)
    print('optimized!')
    draw_scene(vis, dc_scene.shapes, dc_scene.stat_shapes, X, wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat, render_faces=False)
    time.sleep(4)

    # Process
    for Xtmp in X_lst:
        # Recompute collision between pairs for visualization (TODO: should be stored)
        dc_scene.compute_diffcol(Xtmp, col_req, col_req_diff, diffcol=False)
        dc_scene.compute_diffcol_static(Xtmp, col_req, col_req_diff, diffcol=False)
        draw_scene(vis, dc_scene.shapes, dc_scene.stat_shapes, Xtmp, wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat, render_faces=False)
    print("Animation done!")

plt.show()


