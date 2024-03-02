import os
from pathlib import Path

PROJECT_PATH = Path(os.path.realpath(__file__)).parent

EVAL_DATA_PATH = PROJECT_PATH / "eval" / "data"

MESHES_PATH = EVAL_DATA_PATH / "meshes"
MESHES_DECOMP_PATH = EVAL_DATA_PATH / "meshes_decomp"
FLOOR_MESH_PATH = EVAL_DATA_PATH / "floor.ply"

DATASETS_PATH = EVAL_DATA_PATH / "datasets"
POSES_OUTPUT_PATH = EVAL_DATA_PATH / "poses_output"
FLOOR_POSES_PATH = EVAL_DATA_PATH / "floor_poses"
