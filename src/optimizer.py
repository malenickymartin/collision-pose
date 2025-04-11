import time
from copy import deepcopy
import numpy as np
from numpy.linalg import norm
import pinocchio as pin
import matplotlib.pyplot as plt
from typing import Union, List, Dict
import meshcat

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


def optim(dc_scene: DiffColScene, wMo_lst_init: List[pin.SE3],
          params: Union[Dict[str, Union[str,int,List]], None] = None,
          vis_meshes: List = [], vis_meshes_stat: List = []) -> List[pin.SE3]:
    """
    Optimize the poses of the objects in the scene to minimize the collision, perception and gravity costs.

    Inputs:
    - dc_scene: the scene to optimize
    - wMo_lst_init: the initial poses of the objects
    - params: the optimization parameters, a dictionary containing:
        - N_step: the number of optimization steps, default 1000
        - g_grad_scale: the scaling factor for the gravity gradient, default 5
        - coll_grad_scale: the scaling factor for the collision gradient, default 1
        - learning_rate: the learning rate, default 0.01
        - step_lr_decay: the decay factor for the learning rate, default 0.75
        - step_lr_freq: the frequency of the learning rate decay, default 100
        - std_xy_z_theta: the standard deviations for the translation and rotation, default is [0.1, 0,245, 0.51] which corresponds to variation of [0.01, 0.06, 0.26]
    - vis_meshes: the meshes to visualize the scene, default None (no visualization)
    - vis_meshes_stat: the static meshes to visualize the scene, default None (no visualization)
    Returns:
    - the optimized poses of the objects of type List[pin.SE3]
    """

    # Params
    if params is None:
        params = {
            "N_step": 3000,
            "g_grad_scale": 1,
            "coll_grad_scale": 1,
            "learning_rate": 0.0001,
            "step_lr_decay": 1,
            "step_lr_freq": 1000,
            "std_xy_z_theta": [0.05, 0.49, 0.26],
        }

    # Check if optimization is needed
    if not params["g_grad_scale"]:
        cost_c_obj, _ = dc_scene.compute_diffcol(wMo_lst_init)
        cost_c_stat, _ = dc_scene.compute_diffcol_static(wMo_lst_init)
        if np.sum(cost_c_obj) + np.sum(cost_c_stat) < 1e-3:
            print("No collision detected, no need to optimize")
            return wMo_lst_init

    # Logs
    visualize = len(vis_meshes) > 0
    if visualize:
        cost_c_lst, grad_c_norm = [], []
        cost_g_lst, grad_g_norm = [], []
        cost_p_lst, grad_p_norm = [], []
        grad_lst = []
        X_lst = []

    # Params
    std_xy_z_theta = params["std_xy_z_theta"]
    g_grad_scale = params["g_grad_scale"]
    coll_grad_scale = params["coll_grad_scale"]
    N_step = params["N_step"]
    learning_rate = params["learning_rate"]
    lr_decay = params["step_lr_decay"]
    lr_freq = params["step_lr_freq"]

    N_SHAPES = len(dc_scene.shapes_convex)
    X = deepcopy(wMo_lst_init)
    dx = np.zeros(6*N_SHAPES)

    Q_lst = [std_to_Q_aligned(std_xy_z_theta, wMo_lst_init[i]) for i in range(N_SHAPES)]
    L_lst = [cov_to_sqrt_prec(Q) for Q in Q_lst]
    
    # Start of Gradient Descent
    for i in range(N_step):
        if i % lr_freq == 0 and i != 0:
            learning_rate *= lr_decay

        # Compute obj-obj collision gradient
        cost_c_obj, grad_c_obj, num_colls_obj = dc_scene.compute_diffcol(X)

        # Compute obj-static collision gradient
        if len(dc_scene.statics_convex) > 0:
            cost_c_stat, grad_c_stat, num_colls_stat = dc_scene.compute_diffcol_static(X)
        else:
            cost_c_stat, grad_c_stat, num_colls_stat = np.zeros(N_SHAPES), np.zeros(6*N_SHAPES), np.zeros(N_SHAPES)

        # Compute collision gradient
        grad_c = grad_c_obj + grad_c_stat
        num_colls = num_colls_obj + num_colls_stat
        for j in range(N_SHAPES):
            if num_colls[j] > 0:
                grad_c[6*j:6*j+6] = grad_c[6*j:6*j+6]/num_colls[j]

        # Compute gravity gradient
        if g_grad_scale and len(dc_scene.statics_convex) > 0:
            cost_g, grad_g = dc_scene.compute_gravity(X, num_colls)
        else:
            cost_g, grad_g = 0, np.zeros(6*N_SHAPES)

        # Compute perception gradient
        res_p, grad_p = perception_res_grad(X, wMo_lst_init, L_lst, error_fun=error_r3_so3)

        grad = grad_p + coll_grad_scale*grad_c + g_grad_scale * grad_g
        grad = clip_grad(grad)
        dx = -learning_rate*grad
        X = update_est(X, dx)

        # Logs
        if visualize:
            res2cost = lambda r: 0.5*sum(r**2)

            X_lst.append(deepcopy(X))
            cost_c_lst.append(np.sum(cost_c_obj)+np.sum(cost_c_stat))
            cost_g_lst.append(np.sum(cost_g))
            cost_p_lst.append(res2cost(res_p))

            grad_c_norm.append(norm(grad_c))
            grad_g_norm.append(norm(grad_g))
            grad_p_norm.append(norm(grad_p))
            grad_lst.append(norm(grad))

    if visualize:
        steps = np.arange(len(cost_c_lst))
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(steps, cost_c_lst)
        ax[0].set_title('cost collision')
        ax[1].plot(steps, cost_g_lst)
        ax[1].set_title('gravity cost')
        ax[2].plot(steps, cost_p_lst)
        ax[2].set_title('perception cost')
        fig.suptitle('Costs')
        fig.legend()

        fig, ax = plt.subplots(4,1)
        ax[0].plot(steps, grad_c_norm)
        ax[0].set_title('collision gradient norm')
        ax[1].plot(steps, grad_g_norm)
        ax[1].set_title('gravity gradient norm')
        ax[2].plot(steps, grad_p_norm)
        ax[2].set_title('perception gradient norm')
        ax[3].plot(steps, grad_lst)
        ax[3].set_title('total gradient norm')
        fig.suptitle('Gradient norms')
        plt.show(block=False)
        
        print('Create vis')
        input("Continue to init pose?")
        vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6008")
        vis.delete()
        print('Init!')
        # for j in range(N_SHAPES):
        #     show_cov_ellipsoid(vis, wMo_lst_init[j].translation, Q_lst[j][:3,:3], ellipsoid_id=j, nstd=3)
        dc_scene.compute_diffcol(wMo_lst_init)
        dc_scene.compute_diffcol_static(wMo_lst_init)
        draw_scene(vis, vis_meshes, vis_meshes_stat, wMo_lst_init, dc_scene.wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat)
        input("Continue to optimized pose?")
        time.sleep(2)
        print('optimized!')
        # for j in range(N_SHAPES):
        #     show_cov_ellipsoid(vis, X[j].translation, Q_lst[j][:3,:3], ellipsoid_id=j, nstd=3)
        dc_scene.compute_diffcol(X_lst[-1])
        dc_scene.compute_diffcol_static(X_lst[-1])
        draw_scene(vis, vis_meshes, vis_meshes_stat, X, dc_scene.wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat)
        
        # input("Continue to animation?")
        # time.sleep(4)
        # # Process
        # print("Animation start!")
        # for i, Xtmp in enumerate(tqdm(X_lst)):
        #     if i % 10 != 0:
        #         continue
        #     dc_scene.compute_diffcol(Xtmp)
        #     dc_scene.compute_diffcol_static(Xtmp)
        #     draw_scene(vis, vis_meshes, vis_meshes_stat, Xtmp, dc_scene.wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat)
        #     time.sleep(0.1)
        # print("Animation done!")
        

    return X