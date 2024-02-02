# collision-pose
Object Pose Estimation from Images with Geometrical and Physical Consistency



=============================================
# Example simulated scene
## Install extra dependencies
Assuming diffcol already installed:  
`pip install -r requirements.txt`  

## Generate scene (stored in pickle)
`sh download_meshes.py`  
`blenderproc run simulate_object_cluster.py`  
Outputs a scene.pkl storing object and camera poses and a hdf5 storing the rendered view.

## bird-eye render of the scene
`blenderproc vis hdf5 out/0.hdf5`

# Scene optim
`python run_optim.py`