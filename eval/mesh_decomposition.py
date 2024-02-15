import coacd
import trimesh
from pathlib import Path

source_path = Path("eval/data/meshes")
destination_path = source_path.parent / "meshes_convex_multi"
for i in range(1, 22):
    input_file = str(source_path / f"{i}/obj_{i:06d}.ply")
    mesh = trimesh.load(input_file, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    result = coacd.run_coacd(mesh) # a list of convex hulls.
    mesh_parts = []
    for vs, fs in result:
        mesh_parts.append(trimesh.Trimesh(vs, fs))
    (destination_path / f"{i}").mkdir(parents=True, exist_ok=True)
    for j, part in enumerate(mesh_parts):
        part.export(destination_path / f"{i}" /f"obj_{i:06d}_{j:06d}.obj")