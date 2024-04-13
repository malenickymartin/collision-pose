import json
import numpy as np
from config import FLOOR_POSES_PATH

path = FLOOR_POSES_PATH / "tless_bop_floor_poses_1mm_res_dilation_optimized_del.json"

with open(path, "r") as f:
    floor_poses = json.load(f)

del_scenes = ["14","15","16","17","19","20"]
del_im = {"4": [448, 455, 462, 469, 479],
        "13": [450, 462, 478, 486, 500],
        "9": [437, 451, 465, 477, 484, 490, 495],
        "5": [450, 458, 463, 468, 475, 493],
        "2": [434, 439, 446, 455, 459, 463, 470, 478, 489, 494],
        "8": [459, 465, 471, 479, 484, 495],
        "10": [441, 450, 457, 465, 475, 482, 496],
        "12": [438, 455, 471, 482, 490],
        "1": [434, 442, 455, 464, 474, 488, 493],
        "3": [446, 454, 459, 466, 476, 493],
        "6": [434, 442, 462, 469, 475, 485, 496, 501],
        "7": [435, 447, 450, 458, 468, 478, 486, 493],
        "18": [439, 452, 461, 466, 473, 480, 487, 491, 496],
        "11": [441, 452, 459, 469, 476, 486, 497]}

for scene in floor_poses:
    for im in floor_poses[scene]:
        if scene in del_scenes:
            floor_poses[scene][im] = None
        if scene in del_im:
            if int(im) in del_im[scene]:
                floor_poses[scene][im] = None

save_name = "tless_bop_floor_poses_1mm_res_dilation_optimized_del.json"
with open(FLOOR_POSES_PATH / save_name, "w") as f:
    json.dump(floor_poses, f, indent=2)