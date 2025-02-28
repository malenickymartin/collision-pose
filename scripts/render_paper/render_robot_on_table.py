import example_robot_data
import numpy as np
import pinocchio as pin
from pathlib import Path
from robomeshcat import Object, Robot, Scene
from scripts.render_paper.robot_model import RobotModel, T_RC
from scripts.render_paper.handles import HANDLES, PRE_HANDLES

meshes_path = Path(__file__).parent / "meshes"
model = RobotModel(Path("/local2/homes/malenma3/robotic_experiment/data"))

OBJECT = ["cheezit", "sugar", "mustard"][2]
METHODS = ["collision", "megapose"]

scene = Scene()
scene.vis["/Grid"].set_property("Visible", False)
scene.vis["/Axes"].set_property("Visible", False)
clr = np.array([1] * 3)
scene.set_background_color(np.array([1] * 3), clr)

T_BR = pin.SE3(np.array([[ 0.92470941,  0.02978314,  0.37950689,  0.30299043],
       [ 0.02119414, -0.99941636,  0.0267909 ,  0.00726584],
       [ 0.38008331, -0.01673047, -0.92480093,  0.59372245],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]))
o_poses = {"cheezit":{"collision":[[-0.9789210997465049, -0.18748984920348416, -0.08100004089005432, 0.06954457953862614], [-0.13321602895679585, 0.8867643970542906, -0.4426087019392986, 0.11976330303886519], [0.15481258573647896, -0.42248853352925686, -0.8930490032996315, 0.5192527285497807], [0.0, 0.0, 0.0, 1.0]],
           "megapose":[[-0.9789173007011414, -0.1880781352519989, -0.0796712189912796, 0.06869687885046005], [-0.13159096240997314, 0.8790283799171448, -0.45824992656707764, 0.15177293121814728], [0.15622004866600037, -0.4381048083305359, -0.8852454423904419, 0.6554021239280701], [0.0, 0.0, 0.0, 1.0]]},
           "sugar":{"collision":[[-0.8102443691871557, -0.5765602708006294, -0.10527149690075578, -0.18834595813813862], [-0.48103998062125436, 0.7568098260617021, -0.4425373101847464, -0.12064819179544625], [0.3348199773583709, -0.30792357476168264, -0.8905496697134423, 0.6455819595130672], [0.0, 0.0, 0.0, 1.0]],
                  "megapose":[[-0.8104779124259949, -0.576562762260437, -0.10344421863555908, -0.18590474128723145], [-0.4812919795513153, 0.7561164498329163, -0.44344761967658997, -0.11946244537830353], [0.3338913023471832, -0.30961763858795166, -0.8903109431266785, 0.6363437175750732], [0.0, 0.0, 0.0, 1.0]]},
            "mustard":{"collision":[[-0.873360578560648, -0.4866833545957937, -0.0195120531523742, 0.3422973802809757], [-0.4378687557060481, 0.8020470071537305, -0.4061916736985692, 0.15251163283160848], [0.2133362883824788, -0.34620807838747825, -0.9135796728628767, 0.5128842211618484], [0.0, 0.0, 0.0, 1.0]],
                       "megapose":[[-0.8733340501785278, -0.4867442846298218, -0.01917671039700508, 0.3541298806667328], [-0.4375462830066681, 0.8011476993560791, -0.40830838680267334, 0.1575288325548172], [0.21410512924194336, -0.34819892048835754, -0.9126427173614502, 0.5298656225204468], [0.0, 0.0, 0.0, 1.0]]}
}

o1 = Object.create_mesh(
    path_to_mesh=meshes_path / "IKEA_INGO_2.obj",
    name="robot/table",
    texture=meshes_path / "wood_maple.jpg",
    scale=2,
    opacity=0.5,
)
scene.add_object(o1)
o = Object.create_mesh(
    path_to_mesh=meshes_path / f"{OBJECT}.ply",
    name="robot/movable_obj",
    texture=meshes_path / f"{OBJECT}.png",
    scale=0.001)
scene.add_object(o)
robot = example_robot_data.load("panda")
r = Robot(
    pinocchio_model=robot.model,
    pinocchio_data=robot.data,
    pinocchio_geometry_model=robot.visual_model,
    pinocchio_geometry_data=robot.visual_data,
)
scene.add_robot(r)

for i in [1,0]:
    METHOD = METHODS[i]

    # Camera
    T_CO = pin.SE3(np.array(o_poses[OBJECT][METHOD]))
    T_Z = np.eye(4)
    T_Z[2,3] = 0.09
    T_Z = pin.SE3(T_Z)
    T_BC = T_BR * T_RC

    # Robot
    T_OH = pin.SE3(pin.XYZQUATToSE3(HANDLES[OBJECT]["handleZpx"]).homogeneous)
    T_X = np.eye(4)
    T_X[1,3] = 0.015
    T_X = pin.SE3(T_X)
    T_BH = T_BR * T_RC * T_CO * T_Z * T_X * T_OH
    q_9 = model.ik_num(np.array([ 0,-0.78539816,0,-2.35619449,0,1.57079633,0.78539816]), T_BH)
    q_9[7:] = [0.047, 0.047]
    r[:9] = q_9

    # Object
    T_BO = T_BC * T_CO * T_Z
    o.pose = T_BO.homogeneous

    # Table
    o1.rot = pin.exp(np.array([np.pi / 2, 0, 0]))
    o1.pos = np.array([0.4, 0, -0.214 * 2])

    # Render
    scene.render_image().save(f"{OBJECT}_{METHOD}.png")
