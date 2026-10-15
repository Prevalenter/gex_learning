"""Colored local-axis arrows for spherical objects and orientation targets."""
import torch
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply


class OrientationAxes:
    def __init__(self, path, length, device):
        prototypes = {}
        offsets = []
        for index, (axis, color) in enumerate(zip("XYZ", ((1., .05, .05), (.05, 1., .05), (.05, .25, 1.)))):
            material = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=.65)
            prototypes[f"{axis}_shaft"] = sim_utils.CylinderCfg(
                radius=length * .018, height=length * .8, axis=axis, visual_material=material)
            prototypes[f"{axis}_tip"] = sim_utils.ConeCfg(
                radius=length * .05, height=length * .2, axis=axis, visual_material=material)
            for distance in (.4, .9):
                offset = [0., 0., 0.]
                offset[index] = length * distance
                offsets.append(offset)
        self.offsets = torch.tensor(offsets, device=device)
        self.markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path=path, markers=prototypes))

    def update(self, positions, orientations):
        count = positions.shape[0]
        rotations = orientations[:, None, :].expand(count, 6, 4).reshape(-1, 4)
        offsets = self.offsets[None, :, :].expand(count, 6, 3).reshape(-1, 3)
        centers = positions[:, None, :].expand(count, 6, 3).reshape(-1, 3)
        self.markers.visualize(centers + quat_apply(rotations, offsets), rotations,
                               marker_indices=torch.arange(6, device=positions.device).repeat(count))
