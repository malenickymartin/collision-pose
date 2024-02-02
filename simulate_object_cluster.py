import blenderproc as bproc  # keep at the top!
import numpy as np
import pickle

"""
Use blenderproc to simulate objects falling on a rugged surface.
"""

# run sh download_meshes.sh
sphere_path = 'meshes/icosphere.obj'
ground_path = "meshes/passive.obj"

bproc.init()

N_OBJECTS = 70

# Load active and passive objects into the scene
# load_obj always return a list of objects, even if only one defined in .obj file
sphere = bproc.loader.load_obj(sphere_path)[0]
objects = [sphere.duplicate() for _ in range(N_OBJECTS)]
for i in range(len(objects)):
    objects[i].set_name(f'sphere_{i}')
ground = bproc.loader.load_obj(ground_path)[0]

# Create a SUN light and set its properties
light = bproc.types.Light()
light.set_type("SUN")
light.set_location([0, 0, 0])
light.set_rotation_euler([-0.063, 0.6177, -0.1985])
light.set_energy(1)
light.set_color([1, 0.978, 0.407])

# define the pose using opencv camera frame convention: right-down-front = XYZ
# blender uses a right-handed, Z=top representation of the world 
# -> camera on top of the scene, looking down
wMc = bproc.math.build_transformation_mat([0, 0, 70], [np.pi, 0, 0])
# convert to blender=opengl camera frame convention: right-up-back = XYZ
wMc_ogl = bproc.math.change_source_coordinate_frame_of_transformation_matrix(wMc, ["X", "-Y", "-Z"])
bproc.camera.add_camera_pose(wMc_ogl)

# Define a function that samples the pose of a given sphere
def sample_pose(obj: bproc.types.MeshObject):
    obj.set_location(np.random.uniform([-3, -3, 8], [4, 4, 20]))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

# Sample the poses of all objects above the ground without any collisions in-between
bproc.object.sample_poses(
    objects,
    sample_pose_func=sample_pose
)

# Make all objects actively participate in the simulation
for obj in objects:
    obj.enable_rigidbody(active=True)
# The ground should only act as an obstacle and is therefore marked passive.
# To let the objects fall into the valleys of the ground, make the collision shape MESH instead of CONVEX_HULL.
ground.enable_rigidbody(active=False, 
                        collision_shape="MESH",
                        # friction=5.0,
                        # angular_damping=0.1,
                        linear_damping=0.4)

# Run the simulation and fix the poses of the objects at the end
bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=4, max_simulation_time=20, check_object_interval=1)

# Retrieve scene object poses after physics simulation
wMo_lst = [s.get_local2world_mat() for s in objects]
# Filter objects that fell into the void
wMo_lst = [M for M in wMo_lst if M[2,3] > -10]
scene_dic = {
    'wMo_lst': wMo_lst,
    'wMc': wMc
}

with open('scene.pkl', 'wb') as f:
    pickle.dump(scene_dic, f)

# render the whole pipeline
data = bproc.renderer.render()

# write the data to a .hdf5 container
bproc.writer.write_hdf5('out', data)
