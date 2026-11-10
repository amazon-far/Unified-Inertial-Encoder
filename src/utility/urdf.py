"""URDF loading and rerun logging of robot visuals."""
import yourdfpy
import rerun as rr
import numpy as np

from pathlib import Path
from typing import Literal


Manufactor = Literal["unitree"]


def get_urdf(manufactor: Manufactor, path: str) -> yourdfpy.URDF:
    match manufactor:
        case "unitree":
            load_path = Path("assets/unitree/robots", path)
        case _:
            raise ValueError(f"Unrecognized manufactor {manufactor=}")
    assert load_path.exists()
    
    model = yourdfpy.URDF.load(str(load_path), build_collision_scene_graph=False)
    return model


def log_urdf_initial(path: str, model: yourdfpy.URDF, vis_override_urdf=None):
    """Log each visual mesh statically under `{path}/{link_name}/{visual_node}`.

    Per-frame link poses logged at `{path}/{link_name}` then propagate to the visuals through
    Rerun's transform hierarchy.
    """
    scene       = model.scene
    parents     = scene.graph.transforms.parents
    base_frame  = scene.graph.base_frame

    visual_nodes = list(scene.graph.nodes_geometry)
    for visual_node in visual_nodes:
        link_frame = parents.get(visual_node, base_frame)
        link_to_visual, geom_name = scene.graph.get(frame_to=visual_node, frame_from=link_frame)
        
        mesh = scene.geometry[geom_name]
        rr.log(
            f"{path}/{link_frame}/{visual_node}",
            rr.Transform3D(
                mat3x3=np.asarray(link_to_visual[:3, :3], dtype=np.float32),
                translation=np.asarray(link_to_visual[:3, 3], dtype=np.float32),
            ),
            rr.Mesh3D(
                vertex_positions=np.asarray(mesh.vertices, dtype=np.float32),
                triangle_indices=np.asarray(mesh.faces, dtype=np.uint32),
                vertex_normals  =np.asarray(mesh.vertex_normals, dtype=np.float32),
                albedo_factor   =[1.0, 1.0, 1.0, 0.05] if vis_override_urdf is None else vis_override_urdf.albedo,
            ),
            static=True,
        )
    return visual_nodes
