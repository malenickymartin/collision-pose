import time
from copy import deepcopy
import numpy as np
from numpy.linalg import norm
import pinocchio as pin
import matplotlib.pyplot as plt
from typing import Union, List, Dict
import meshcat
import hppfcl

import pydiffcol

from src.scene import DiffColScene
from src.vis import draw_scene
from src.optim_tools import (
    perception_res_grad,
    update_est,
    clip_grad,
    std_to_Q_aligned,
    cov_to_sqrt_prec,
    error_se3, 
    error_r3_so3
)
from src.vis import show_cov_ellipsoid

def three_phase_optim(dc_scene: DiffColScene, wMo_lst_init: List[pin.SE3],
                      col_req: hppfcl.DistanceRequest, col_req_diff: pydiffcol.DerivativeRequest,
                      params: Union[Dict[str, Union[str,int,List]], None] = None,
                      vis_meshes: List = [], vis_meshes_stat: List = []) -> List[pin.SE3]:
    """
    Optimize the poses of the objects in the scene.
    The optimization is done in three phases: first without gravity only with collisions, then with gravity, and again only with collisions.
    Inputs: same as optim
    Returns: same as optim
    """
    if params is None:
        params = {
            "N_step": 1000,
            "g_grad_scale": 2,
            "coll_grad_scale": 2,
            "coll_exp_scale": 0,
            "learning_rate": 0.0001,
            "step_lr_decay": 1,
            "step_lr_freq": 1000,
            "std_xy_z_theta": [0.05, 0.49, 0.26],
            "method": "GD",
            "method_params": None
        }
    ces = params["coll_exp_scale"]
    ggs = params["g_grad_scale"]
    params["coll_exp_scale"] = 0
    params["g_grad_scale"] = 0
    X = optim(dc_scene, wMo_lst_init, col_req, col_req_diff, params, vis_meshes, vis_meshes_stat)
    params["coll_exp_scale"] = ces
    params["g_grad_scale"] = ggs
    X = optim(dc_scene, X, col_req, col_req_diff, params, vis_meshes, vis_meshes_stat)
    params["coll_exp_scale"] = ces
    params["g_grad_scale"] = 0
    X = optim(dc_scene, X, col_req, col_req_diff, params, vis_meshes, vis_meshes_stat)
    params["g_grad_scale"] = ggs
    return X


def optim(dc_scene: DiffColScene, wMo_lst_init: List[pin.SE3],
          col_req: hppfcl.DistanceRequest, col_req_diff: pydiffcol.DerivativeRequest,
          params: Union[Dict[str, Union[str,int,List]], None] = None,
          vis_meshes: List = [], vis_meshes_stat: List = []) -> List[pin.SE3]:
    """
    Optimize the poses of the objects in the scene to minimize the collision, perception and gravity costs.

    Inputs:
    - dc_scene: the scene to optimize
    - wMo_lst_init: the initial poses of the objects
    - col_req: the collision request
    - col_req_diff: the collision request for the diff
    - params: the optimization parameters, a dictionary containing:
        - N_step: the number of optimization steps, default 1000
        - g_grad_scale: the scaling factor for the gravity gradient, default 5
        - coll_grad_scale: the scaling factor for the collision gradient, default 1
        - learning_rate: the learning rate, default 0.01
        - step_lr_decay: the decay factor for the learning rate, default 0.75
        - step_lr_freq: the frequency of the learning rate decay, default 100
        - std_xy_z_theta: the standard deviations for the translation and rotation, default is [0.1, 0,245, 0.51] which corresponds to variation of [0.01, 0.06, 0.26]
        - method: the optimization method to use, one of "GD", "MGD", "NGD", "adagrad", "rmsprop", "adam" # Ref: https://cs231n.github.io/neural-networks-3/#sgd
        - method_params: the parameters for the optimization method, e.g. mu for MGD, NGD, eps for adagrad, [decay, eps] for rmsprop, [beta1, beta2, eps] for adam
        - coll_exp_scale: the scaling factor for the collision exponential, default 0
    - vis_meshes: the meshes to visualize the scene, default None (no visualization)
    - vis_meshes_stat: the static meshes to visualize the scene, default None (no visualization)
    Returns:
    - the optimized poses of the objects of type List[pin.SE3]
    """

    # Params
    if params is None:
        params = {
            "N_step": 1000,
            "g_grad_scale": 2,
            "coll_grad_scale": 2,
            "coll_exp_scale": 0,
            "learning_rate": 0.0001,
            "step_lr_decay": 1,
            "step_lr_freq": 1000,
            "std_xy_z_theta": [0.05, 0.49, 0.26],
            "method": "GD",
            "method_params": None
        }

    # Check if optimization is needed
    if not params["g_grad_scale"]:
        cost_c_obj, _ = dc_scene.compute_diffcol(wMo_lst_init, col_req, col_req_diff, diffcol=False)
        cost_c_stat, _ = dc_scene.compute_diffcol_static(wMo_lst_init, col_req, col_req_diff, diffcol=False)
        if np.sum(cost_c_obj) + np.sum(cost_c_stat) < 1e-3:
            print("No collision detected, no need to optimize")
            return wMo_lst_init

    # Logs
    visualize = len(vis_meshes) > 0
    if visualize:
        cost_c_lst, grad_c_norm = [], []
        cost_c_stat_lst, grad_c_stat_norm = [], []
        cost_pt_lst, cost_po_lst, grad_p_norm = [], [], []

    X = deepcopy(wMo_lst_init)
    X_lst = []

    N_SHAPES = len(dc_scene.shapes_convex)

    # All
    std_xy_z_theta = params["std_xy_z_theta"]
    g_grad_scale = params["g_grad_scale"]
    coll_exp_scale = params["coll_exp_scale"]
    coll_grad_scale = params["coll_grad_scale"]
    N_step = params["N_step"]
    learning_rate = params["learning_rate"]
    lr_decay = params["step_lr_decay"]
    lr_freq = params["step_lr_freq"]
    method = params["method"]

    diffcol_flag = True
    if coll_grad_scale == 0:
        diffcol_flag = False

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
    L_lst = [cov_to_sqrt_prec(Q) for Q in Q_lst]
    
    for i in range(N_step):
        if i % lr_freq == 0 and i != 0:
            learning_rate *= lr_decay

        if method == 'NGD':
            # Evaluate gradient at a look-ahead state
            X_eval = update_est(X, mu*dx)
        else:
            # Evaluate gradient at current state
            X_eval = X

        # Compute obj-obj collision gradients
        cost_c_obj, grad_c_obj = dc_scene.compute_diffcol(X_eval, col_req, col_req_diff, coll_exp_scale, diffcol=diffcol_flag)

        # Compute obj-static collision gradients
        if len(dc_scene.statics_convex) > 0:
            cost_c_stat, grad_c_stat = dc_scene.compute_diffcol_static(X_eval, col_req, col_req_diff, coll_exp_scale, diffcol=diffcol_flag)
        else:
            cost_c_stat, grad_c_stat = np.zeros(N_SHAPES), np.zeros(6*N_SHAPES)

        # Compute gravity gradient
        if g_grad_scale and len(dc_scene.statics_convex) > 0:
            grad_g = dc_scene.compute_gravity(X_eval, cost_c_stat, cost_c_obj)
        else:
            grad_g = np.zeros(6*N_SHAPES)

        # Compute perception gradients
        res_p, grad_p = perception_res_grad(X_eval, wMo_lst_init, L_lst, error_fun=error_r3_so3)

        grad_c = grad_c_obj + grad_c_stat
        grad = coll_grad_scale*grad_c + grad_p + g_grad_scale * grad_g
        grad = clip_grad(grad)

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
            cost_c_lst.append(np.sum(cost_c_obj))
            cost_c_stat_lst.append(np.sum(cost_c_stat))
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
        input("Continue to init pose?")
        vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        vis.delete()
        print('Init!')
        # for j in range(N_SHAPES):
        #     show_cov_ellipsoid(vis, wMo_lst_init[j].translation, Q_lst[j][:3,:3], ellipsoid_id=j, nstd=3)
        dc_scene.compute_diffcol(wMo_lst_init, col_req, col_req_diff)
        dc_scene.compute_diffcol_static(wMo_lst_init, col_req, col_req_diff, diffcol=False)
        draw_scene(vis, vis_meshes, vis_meshes_stat, wMo_lst_init, dc_scene.wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat)
        input("Continue to optimized pose?")
        time.sleep(2)
        print('optimized!')
        # for j in range(N_SHAPES):
        #     show_cov_ellipsoid(vis, X[j].translation, Q_lst[j][:3,:3], ellipsoid_id=j, nstd=3)
        dc_scene.compute_diffcol(X_lst[-1], col_req, col_req_diff, diffcol=False)
        dc_scene.compute_diffcol_static(X_lst[-1], col_req, col_req_diff, diffcol=False)
        draw_scene(vis, vis_meshes, vis_meshes_stat, X, dc_scene.wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat)
        
        # input("Continue to animation?")
        # time.sleep(4)
        # # Process
        # print("Animation start!")
        # for i, Xtmp in enumerate(tqdm(X_lst)):
        #     if i % 10 != 0:
        #         continue
        #     dc_scene.compute_diffcol(Xtmp, col_req, col_req_diff, diffcol=False)
        #     dc_scene.compute_diffcol_static(Xtmp, col_req, col_req_diff, diffcol=False)
        #     draw_scene(vis, vis_meshes, vis_meshes_stat, Xtmp, dc_scene.wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat)
        #     time.sleep(0.1)
        # print("Animation done!")
        

    return X