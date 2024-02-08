# collision-pose
Object Pose Estimation from Images with Geometrical and Physical Consistency

=============================================
# Install
Assuming diffcol already installed. Everything regarding evaluation is in "eval" directory, you may skip steps regarding this directory.
1. Install requirements.  
`pip install -r requirements.txt`  
2. Download meshes.  
`sh download_meshes.sh`  
`sh eval/download_meshes_ycbv.sh`  
3. Download test dataset from BOP.  
`sh eval/download_test_dataset.sh`  

=============================================
# Evaluation

It is assumed that everything related to the "eval" directory was downloaded in the "Install" chapter.
Next you will have to make convex decomposition of meshes with:  
`python eval/mesh_decomposition.py`  
You can either evaluated real scenes downloaded from BOP website or synthetic scenes created using Blenderproc. In the latter case, it is necessary that the ground truth scene information is stored in the [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md#3d-object-models), same as the real scenes. For the real dataset Megapose results are saved in CSV file. Columns of this file are: scene_id, im_id, obj_id, score, R, t, time. The synthetic dataset has to have the results saved in separate JSON file for each image. Format of this file is taken from happypose/happypose/pose_estimators/megapose/src/megapose/scripts/run_inference_on_example.py.  
If you have the results to evaluate you can run the following script:  
`python eval/collision_evaluation.py`  
You can then select which type of evaluation you want to proceed with. Note that the synthetic dataset evaluation will be much faster, since the pose of the floor is known (origin). For the read dataset the pose of floor is calculated from depth images using RANSAC.

=============================================
# Example simulated scene

## Generate scene (stored in pickle)

`blenderproc run simulate_object_cluster.py`  
Outputs a scene.pkl storing object and camera poses and a hdf5 storing the rendered view.

## bird-eye render of the scene
`blenderproc vis hdf5 out/0.hdf5`

# Scene optim
`python run_optim.py`