import sys
import coacd
import trimesh
from config import MESHES_PATH, MESHES_DECOMP_PATH

if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
else:
    dataset_name = input("Enter the dataset name: ")

source_path = MESHES_PATH / dataset_name
destination_path = MESHES_DECOMP_PATH / dataset_name
for i in range(1, len(list(source_path.iterdir())) + 1):
    input_file = str(source_path / f"{i}/obj_{i:06d}.ply")
    mesh = trimesh.load(input_file, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    result = coacd.run_coacd(mesh, max_convex_hull=32) # a list of convex hulls.
    mesh_parts = []
    for vs, fs in result:
        mesh_parts.append(trimesh.Trimesh(vs, fs))
    (destination_path / f"{i}").mkdir(parents=True, exist_ok=True)
    for j, part in enumerate(mesh_parts):
        part.export(destination_path / f"{i}" /f"obj_{i:06d}_{j:06d}.obj")