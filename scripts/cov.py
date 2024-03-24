import meshcat
import pinocchio as pin
import numpy as np

GREEN = np.array([110, 250, 90, 200]) / 255
RED = np.array([200, 11, 50, 100]) / 255

def meshcat_material(r, g, b, a):
    material = meshcat.geometry.MeshPhongMaterial()
    material.color = int(r * 255) * 256 ** 2 + int(g * 255) * 256 + \
        int(b * 255)
    material.opacity = a
    return material

def get_ellipsoid(cov, nstd=2):
    # https://users.cs.utah.edu/~tch/CS4640/resources/A%20geometric%20interpretation%20of%20the%20covariance%20matrix.pdf
    eigvals, eigvecs = np.linalg.eigh(cov)
    radii = np.sqrt(eigvals) * nstd
    return radii, eigvecs

def show_cov_ellipsoid(vis, mean, cov, ellipsoid_id=1, nstd=2):
    radii, R_ellipsoid = get_ellipsoid(cov, nstd)
    T_ellipsoid = pin.SE3(R_ellipsoid, mean)
    ellipsoid_mesh = meshcat.geometry.Ellipsoid(radii)
    vis[f"ellipsoid_{ellipsoid_id}"].set_object(ellipsoid_mesh, meshcat_material(*RED))
    vis[f"ellipsoid_{ellipsoid_id}"].set_transform(T_ellipsoid.homogeneous)

def test_change_frame():
    from optim import change_Q_frame
    from eval.eval_utils import load_meshes, draw_shape
    from config import MESHES_PATH

    Q = [0.0002, 0.08, 0.26]
    grid = True
    pin.seed(0)
    np.random.seed(0)
    DS_NAME = "hopevideo"

    num_meshes = 26 if grid else np.random.randint(1, 20)
    meshes_all = load_meshes(MESHES_PATH / DS_NAME, convex=False)
    #meshes = np.random.choice(list(meshes_all.values()), num_meshes)
    meshes = [meshes_all["1"]]*26

    if grid:
        # Generate poses in a 3x3x3 grid without the center pose
        poses = []
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    pose = pin.SE3.Random()
                    pose.translation = np.array([x, y, z])
                    poses.append(pose)
    else:
        # Generate random poses
        poses = [pin.SE3.Random() for _ in range(len(meshes))]
        for pose in poses:
            pose.translation[2] = 1
    
    covs = [change_Q_frame(Q, pose) for pose in poses]

    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    vis.delete()

    for i, (mesh, pose, cov) in enumerate(zip(meshes, poses, covs)):
        cov_t = cov[:3, :3]
        show_cov_ellipsoid(vis, pose.translation, cov_t, i)
        draw_shape(vis, mesh, f"mesh_{i}", pose, GREEN)


if __name__ == "__main__":
    i = int(input("Press 1 to test_change_frame, 2 to plot points and ellipsoid: "))
    if i == 1:
        test_change_frame()
    elif i == 2:
        vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        vis.delete()

        point_mesh = meshcat.geometry.Sphere(0.005)
        points_T = [pin.SE3.Identity() for _ in range(1000)]
        mean_gt = np.random.uniform(-1, 1, 3)
        R = pin.exp3(np.random.random(3))
        cov_sqrt = R @ np.diag([0.1, 0.2, 0.4])
        cov_gt = cov_sqrt @ cov_sqrt.T
        for T in points_T:
            T.translation = np.random.multivariate_normal(mean_gt, cov_gt)
        for i, T in enumerate(points_T):
            vis[f"point_{i}"].set_object(point_mesh, meshcat_material(*GREEN))
            vis[f"point_{i}"].set_transform(T.homogeneous)


        mean = np.mean([T.translation for T in points_T], axis=0)
        cov = np.cov([T.translation for T in points_T], rowvar=False)
        nstd = 2

        radii, R_ellipsoid = get_ellipsoid(cov, nstd)
        T_ellipsoid = pin.SE3(R_ellipsoid, mean)
        ellipsoid_mesh = meshcat.geometry.Ellipsoid(radii)
        vis["ellipsoid"].set_object(ellipsoid_mesh, meshcat_material(*RED))
        vis["ellipsoid"].set_transform(T_ellipsoid.homogeneous)
