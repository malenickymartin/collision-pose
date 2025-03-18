# collision-pose
Object Pose Estimation from Images with Geometrical and Physical Consistency

# Installation
### I. Collision-pose set-up
First install the repository and the required packages:
```
git clone git@github.com:malenickymartin/collision-pose.git
cd collision-pose
pip install -r setup/requirements.txt
```
Next [Diffcol](https://lmontaut.github.io/diffcol_rs.github.io/) needs to be installed, to do so you can follow guide in `setup/diffcol_intall_guide.txt`.\
Then add Diffcol to PYTHONPATH:\
``export PYTHONPATH="${PYTHONPATH}:`pwd`/../diffcol/build/bindings/"``

### II. Dataset setup
The following steps have to be done for each dataset used:
1. Download meshes in format data/meshes/{dataset name}/{object id}/{mesh file name}, or use the following script to download meshes from [BOP Website](https://bop.felk.cvut.cz/datasets/):\
`sh setup/download_meshes_bop.sh {dataset name}`
1. Make a convex decomposition of the meshes and store it in data/meshes_decomp/{dataset name}/{object id}/{decomposed mesh files}. For this you can use script:\
`python3 setup/mesh_decomposition.py {dataset name}`
1. Some scripts (table pose estimation, point-cloud visualization) also require downloading the entire BOP dataset to the data/datasets/{dataset name} directory. Usually this is **not required**. If you are planning on using the mentioned scripts, you can download the datasets using:\
`sh setup/download_test_dataset.sh {dataset name}`

### III. Support plane poses
The support plane (floor or table) poses are stored in `data/floor_poses/{file name}.json`. The support plane poses are used for the optimization of the object poses for datasets that contain support plane (either floor or table), that can be estimated or is otherwise known. These support planes can be used for BOP submission evaluation if argument --floor with support plane .json file name is passed. The support plane poses are stored in the following format (transformations are w.r.t. the camera coordinate system):
```
floor_poses/{filename}.json = 
{
    "{scene id}": {
        "{image id}": {
            "R": [[...], [...], [...]], # 3x3 rotation matrix
            "t": [...], # 3x1 translation vector
        },
        ...
    },
    ...
}
```

### IV. Data paths
Note that all paths are defined in `config.py` and can be adjusted there.

# Inference
There are two main ways to use this repository:
1. Evaluate your BOP submission .csv file:\
`python3 eval/eval_with_optim.py --dataset={dataset name} --init_poses={file name of your submission}`\
Fore more arguments run:\
`python3 eval/eval_with_optim.py --help`

1. Use the optimizer directly in your code:\
`from src.optimizer import optim as collision_pose_optim`