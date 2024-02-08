import pyvista as pv
import pyVHACD
from tqdm import tqdm
from pathlib import Path

source_path = Path("eval/data/meshes")
destination_path = source_path.parent / "meshes_convex_multi"

for i in tqdm(range(1, 22)):
    mesh = pv.read(str(source_path / f"{i}/obj_{i:06d}.ply"))
    Path(destination_path / f"{i}").mkdir(parents=True, exist_ok=True)
    outputs = pyVHACD.compute_vhacd(mesh.points, mesh.faces)
    for j, (mesh_points, mesh_faces) in enumerate(outputs):
        plotter = pv.Plotter()
        plotter.add_mesh(pv.PolyData(mesh_points, mesh_faces), color=list(pv.hexcolors.keys())[j])
        plotter.export_obj(str(destination_path / f"{i}/obj_{j:06d}_{i:06d}.obj"))