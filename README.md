# collision-pose
Object Pose Estimation from Images with Geometrical and Physical Consistency

# Install
First Diffcol needs to be installed, to do so you can follow guide in `setup/diffcol_intall_guide.txt`.
### I. Collision-pose set-up
The following steps only have to be done one time after downloading the repository:
1. Add Diffcol to PYTHONPATH:\
``export PYTHONPATH="${PYTHONPATH}:`pwd`/../diffcol/build/bindings/"``
1. Install requirements:\
`pip install -r setup/requirements.txt`
### II. Dataset setup
Do the following steps for each dataset used:
1. Download your meshes to the data/meshes directory, or use the following example script to download meshes from [BOP Website](https://bop.felk.cvut.cz/datasets/):\
`sh setup/download_meshes.sh {dataset name}`
2. Make a convex decomposition of the meshes and store it in the data/meshes_decomp folder you can use script:\
`python3 setup/mesh_decomposition.py {dataset name}`
3. Some scripts (table pose estimation, point-cloud visualization) also require downloading the entire BOP dataset to the data/datasets folder, for this you can use the script:\
`sh setup/download_test_dataset.sh {dataset name}`

# Inference
Several scripts are prepared for inference. You can use the complete pipeline using [Happypose](https://agimus-project.github.io/happypose/index.html) for inference on individual images, evaluate the entire dataset on the output of your pose estimator using `eval/eval_with_optim.py` or embed our method directly into your code using `src/optimizer`.
If you are evaluating the entire dataset, the poses should be stored in [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md).