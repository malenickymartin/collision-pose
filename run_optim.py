import time
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm
import pinocchio as pin
import matplotlib.pyplot as plt
from typing import Union, List, Dict

import pydiffcol
from pydiffcol.utils import (
    select_strategy
)
from pydiffcol.utils_render import create_visualizer
import hppfcl

from scene import DiffColScene, draw_scene, read_poses_pickle, SelectStrategyConfig
from optim import (
    perception_res_grad,
    update_est,
    clip_grad,
    std_to_Q_aligned,
    cov_to_R
)
from spatial import perturb_se3
from scripts.cov import show_cov_ellipsoid


def optim(dc_scene: DiffColScene, wMo_lst_init: List[pin.SE3],
          col_req: hppfcl.DistanceRequest, col_req_diff: pydiffcol.DerivativeRequest,
          params: Union[Dict[str, Union[str,int,List]], None] = None,
          vis_meshes: Union[List, None] = None, vis_meshes_stat: Union[List, None] = None) -> List[pin.SE3]:
    """
    Optimize the poses of the objects in the scene to minimize the collision cost and the perception cost.

    Inputs:
    - dc_scene: the scene to optimize
    - wMo_lst_init: the initial poses of the objects
    - col_req: the collision request
    - col_req_diff: the collision request for the diff
    - params: the optimization parameters, a dictionary containing:
        - N_step: the number of optimization steps, default 1000
        - coll_grad_scale: the scaling factor for the collision gradient, default 1
        - learning_rate: the learning rate, default 0.01
        - step_lr_decay: the decay factor for the learning rate, default 0.5
        - step_lr_freq: the frequency of the learning rate decay, default 100
        - std_xy_z_theta: the standard deviations for the translation and rotation, default is [0.1, 0,245, 0.51] which corresponds to variation of [0.01, 0.06, 0.26]
        - method: the optimization method to use, one of "GD", "MGD", "NGD", "adagrad", "rmsprop", "adam" # Ref: https://cs231n.github.io/neural-networks-3/#sgd
        - method_params: the parameters for the optimization method, e.g. mu for MGD, NGD, eps for adagrad, [decay, eps] for rmsprop, [beta1, beta2, eps] for adam
    - vis_meshes: the meshes to visualize the scene, default None (no visualization)

    Returns:
    - the optimized poses of the objects of type List[pin.SE3]
    """
    # Check if optimization is needed
    cost_c_obj, _ = dc_scene.compute_diffcol(wMo_lst_init, col_req, col_req_diff)
    cost_c_stat, _ = dc_scene.compute_diffcol_static(wMo_lst_init, col_req, col_req_diff)
    if cost_c_obj + cost_c_stat < 1e-3:
        print("No collision detected, no need to optimize")
        return wMo_lst_init

    # Params
    if params is None:
        params = {
            "N_step": 1000,
            "coll_grad_scale": 1,
            "learning_rate": 0.01,
            "step_lr_decay": 0.5,
            "step_lr_freq": 50,
            "std_xy_z_theta": [0.1, 0,245, 0.51],
            "method": "GD",
            "method_params": None
        }

    # Logs
    visualize = vis_meshes is not None
    if visualize:
        cost_c_lst, grad_c_norm = [], []
        cost_c_stat_lst, grad_c_stat_norm = [], []
        cost_pt_lst, cost_po_lst, grad_p_norm = [], [], []

    X = deepcopy(wMo_lst_init)
    X_lst = []

    N_SHAPES = len(dc_scene.shapes_convex)

    # All
    std_xy_z_theta = params["std_xy_z_theta"]
    coll_grad_scale = params["coll_grad_scale"]
    N_step = params["N_step"]
    learning_rate = params["learning_rate"]
    lr_decay = params["step_lr_decay"]
    lr_freq = params["step_lr_freq"]
    method = params["method"]

    # Momentum param MGD, NGD
    if method in ['MGD', 'NGD']:
        mu = params["method_params"] #0.90

    # adagrad
    elif method == 'adagrad':
        eps_adagrad = params["method_params"] #1e-8

    # RMSprop
    elif method == 'rmsprop':
        decay_rmsprop, eps_rmsprop = params["method_params"] #[0.99, 1e-8]

    # adam
    elif method == 'adam':
        beta1_adam, beta2_adam, eps_adam = params["method_params"] #[0.99, 0.999, 1e-8]

    cache_ada = np.zeros(6*N_SHAPES)
    m_adam = np.zeros(6*N_SHAPES)
    v_adam = np.zeros(6*N_SHAPES)
    dx = np.zeros(6*N_SHAPES)

    Q_lst = [std_to_Q_aligned(std_xy_z_theta, wMo_lst_init[i]) for i in range(N_SHAPES)]
    R_lst = [cov_to_R(Q) for Q in Q_lst]
    
    for i in tqdm(range(N_step)):
        if i % lr_freq == 0 and i != 0:
            learning_rate *= lr_decay

        if method == 'NGD':
            # Evaluate gradient at a look-ahead state
            X_eval = update_est(X, mu*dx)
        else:
            # Evaluate gradient at current state
            X_eval = X

        cost_c_obj, grad_c_obj = dc_scene.compute_diffcol(X_eval, col_req, col_req_diff)
        if len(dc_scene.statics_convex) > 0:
            cost_c_stat, grad_c_stat = dc_scene.compute_diffcol_static(X_eval, col_req, col_req_diff)
        else:
            cost_c_stat, grad_c_stat = 0.0, np.zeros(6*N_SHAPES)
        res_p, grad_p = perception_res_grad(X_eval, wMo_lst_init, R_lst)

        grad_c_obj = clip_grad(grad_c_obj)
        grad_c_stat = clip_grad(grad_c_stat)
        grad_c = grad_c_obj + grad_c_stat
        grad = coll_grad_scale*grad_c + grad_p

        if method == 'GD':
            dx = -learning_rate*grad
        elif method in ['MGD', 'NGD']:
            dx = mu*dx - learning_rate*grad
        elif method == 'adagrad':
            cache_ada += grad**2
            dx = -learning_rate * grad / (np.sqrt(cache_ada)+eps_adagrad)
        elif method == 'rmsprop':
            cache_ada += decay_rmsprop*cache_ada + (1 - decay_rmsprop)*grad**2
            dx = -learning_rate * grad / (np.sqrt(cache_ada)+eps_rmsprop)
        elif method == 'adam':
            m_adam = beta1_adam*m_adam + (1-beta1_adam)*grad
            v_adam = beta2_adam*v_adam + (1-beta2_adam)*(grad**2)
            dx = - learning_rate * m_adam / (np.sqrt(v_adam) + eps_adam)
        else:
            raise ValueError(f"Unknown method {method}")

        # state update
        X = update_est(X, dx)

        # Logs
        if visualize:
            X_lst.append(deepcopy(X))
            cost_c_lst.append(cost_c_obj)
            cost_c_stat_lst.append(cost_c_stat)
            grad_c_norm.append(norm(grad_c_obj))
            grad_c_stat_norm.append(norm(grad_c_stat))
            
            res2cost = lambda r: 0.5*sum(r**2)

            cost_pt, cost_po = res2cost(res_p.reshape((-1,2,3))[:,0].reshape(-1)), res2cost(res_p.reshape((-1,2,3))[:,1].reshape(-1))
            cost_pt_lst.append(cost_pt)
            cost_po_lst.append(cost_po)
            grad_p_norm.append(norm(grad_p))

    if visualize:
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
        ax[0].plot(steps, cost_pt_lst)
        ax[0].set_ylabel('err_t [m]')
        ax[0].set_title('cost translation')
        ax[1].plot(steps, cost_po_lst)
        ax[1].set_ylabel('err_o [rad]')
        ax[1].set_title('cost orientation')
        ax[2].plot(steps, grad_p_norm)
        ax[2].set_title('grad norm perception')
        plt.show(block=False)
        print('Create vis')
        vis = create_visualizer(grid=True, axes=True)
        input("Continue?")
        print('Init!')
        for j in range(N_SHAPES):
            show_cov_ellipsoid(vis, wMo_lst_init[j].translation, Q_lst[j][:3,:3], ellipsoid_id=j, nstd=3)
        dc_scene.compute_diffcol(wMo_lst_init, col_req, col_req_diff)
        dc_scene.compute_diffcol_static(wMo_lst_init, col_req, col_req_diff, diffcol=False)
        draw_scene(vis, vis_meshes, vis_meshes_stat, wMo_lst_init, dc_scene.wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat)
        time.sleep(4)
        print('optimized!')
        for j in range(N_SHAPES):
            show_cov_ellipsoid(vis, X[j].translation, Q_lst[j][:3,:3], ellipsoid_id=j, nstd=3)
        dc_scene.compute_diffcol(X_lst[-1], col_req, col_req_diff)
        dc_scene.compute_diffcol_static(X_lst[-1], col_req, col_req_diff, diffcol=False)
        draw_scene(vis, vis_meshes, vis_meshes_stat, X, dc_scene.wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat)
        time.sleep(4)
        # Process
        print("Animation start!")
        for i, Xtmp in enumerate(tqdm(X_lst)):
            if i % 10 != 0:
                continue
            dc_scene.compute_diffcol(Xtmp, col_req, col_req_diff, diffcol=False)
            dc_scene.compute_diffcol_static(Xtmp, col_req, col_req_diff, diffcol=False)
            draw_scene(vis, vis_meshes, vis_meshes_stat, Xtmp, dc_scene.wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat)
            time.sleep(0.5)
        print("Animation done!")

    return X

if __name__ == "__main__":
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

    optim(dc_scene, X_meas, col_req, col_req_diff, visualize=MESHCAT_VIS)