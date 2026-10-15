"""Edit the GX4M initial joint positions and DexCube pose without Isaac Sim."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/gex_learning"))
from gex_learning.tasks.direct.inhand_reorient.initial_pose import (
    DEFAULT_PATH, JOINT_NAMES, joint_limits, load_pose, resolve_path, save_pose,
)
from gex_learning.tasks.direct.dexanytwist.dataset import load_hand_spec
from pose_math import _quat_from_euler_xyz_deg, _euler_xyz_deg_from_quat


def run_viser(args):
    try:
        import numpy as np
        import trimesh
        import viser
        from viser.extras import ViserUrdf
    except ImportError as error:
        raise SystemExit(f'Missing {error.name}; run: python -m pip install -e ".[viser]"') from error

    path = resolve_path(args.config)
    state = load_pose(path)
    lock = threading.RLock()
    server = viser.ViserServer(host=args.host, port=args.port, label="InHand · 初始抓握配置")
    server.scene.set_up_direction("+z")
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    server.scene.add_grid("/ground", width=1.0, height=1.0, cell_size=.05, section_size=.25)
    server.scene.add_frame("/hand", position=(0., 0., .55),
                           wxyz=_quat_from_euler_xyz_deg([-90., -90., 0.]), show_axes=False)
    hand = ViserUrdf(server, load_hand_spec().urdf_path, root_node_name="/hand")
    hand_names = hand.get_actuated_joint_names()
    limits = joint_limits()

    # Color each face differently so a cube's orientation is visible without a simulator.
    mesh = trimesh.creation.box(extents=(.06, .06, .06))
    colors = np.array([[233, 88, 80, 255], [59, 173, 218, 255], [235, 179, 65, 255],
                       [140, 109, 216, 255], [77, 193, 133, 255], [232, 139, 192, 255]])
    normals = mesh.face_normals
    face_axes = np.abs(normals).argmax(axis=1)
    mesh.visual.face_colors = colors[2 * face_axes + (normals[np.arange(len(normals)), face_axes] < 0)]
    cube = server.scene.add_frame("/cube", axes_length=.055, axes_radius=.0015,
                                  position=tuple(state["object_position_m"]),
                                  wxyz=tuple(state["object_orientation_wxyz"]))
    cube_mesh = server.scene.add_mesh_trimesh("/cube/mesh", mesh, scale=0.85)
    transform = server.scene.add_transform_controls(
        "/controls/cube", scale=.11, depth_test=False,
        position=tuple(state["object_position_m"]), wxyz=tuple(state["object_orientation_wxyz"]))

    with server.gui.add_folder("保存与载入"):
        server.gui.add_markdown("调整关节滑条或拖动方块坐标轴，再点击**保存初始姿态**。\n\n"
                                "保存后，新启动的 InHand 训练和回放会从此关节位置、方块位姿重置。")
        save_button = server.gui.add_button("保存初始姿态")
        reload_button = server.gui.add_button("重新载入已保存配置")
        disable_button = server.gui.add_button("恢复原始随机初始化")
        status = server.gui.add_markdown("已载入保存姿态。" if state["enabled"] else
                                         "尚未启用保存姿态；当前训练使用原始随机初始化。")

    with server.gui.add_folder("方块位姿"):
        position = server.gui.add_vector3("位置 XYZ（米）", initial_value=tuple(state["object_position_m"]), step=.001)
        rotation = server.gui.add_vector3("旋转 XYZ（度）",
            initial_value=tuple(_euler_xyz_deg_from_quat(state["object_orientation_wxyz"])), step=1.)
        preview_scale = server.gui.add_slider("预览尺寸倍率", min=0.82, max=0.88, step=.001, initial_value=0.85)
        server.gui.add_markdown("倍率只影响预览；训练在 0.82–0.88 倍之间随机尺寸。\n\n"
                                "姿态编辑不运行物理仿真，接触稳定性需在仿真中验证。")

    sliders = {}
    for group in range(4):
        with server.gui.add_folder(f"关节 {group * 4 + 1}–{group * 4 + 4}（度）"):
            for name in JOINT_NAMES[group * 4:group * 4 + 4]:
                low, high = limits[name]
                sliders[name] = server.gui.add_slider(name, min=math.degrees(low), max=math.degrees(high),
                    step=.1, initial_value=math.degrees(state["joint_positions_rad"][name]))

    def update_hand():
        hand.update_cfg(np.array([state["joint_positions_rad"][name] for name in hand_names]))

    def update_cube():
        cube.position = tuple(state["object_position_m"])
        cube.wxyz = tuple(state["object_orientation_wxyz"])

    def mark_dirty():
        status.content = "**有未保存修改。** 点击“保存初始姿态”后生效。"

    for name, slider in sliders.items():
        def on_joint(event, joint_name=name, handle=slider):
            if event.client is None:
                return
            with lock:
                low, high = limits[joint_name]
                state["joint_positions_rad"][joint_name] = min(high, max(low, math.radians(handle.value)))
                update_hand()
                mark_dirty()
        slider.on_update(on_joint)

    @position.on_update
    def on_position(event):
        if event.client is None:
            return
        with lock:
            state["object_position_m"] = list(position.value)
            update_cube()
            transform.position = cube.position
            mark_dirty()

    @rotation.on_update
    def on_rotation(event):
        if event.client is None:
            return
        with lock:
            state["object_orientation_wxyz"] = list(_quat_from_euler_xyz_deg(rotation.value))
            update_cube()
            transform.wxyz = cube.wxyz
            mark_dirty()

    @transform.on_update
    def on_transform(event):
        if event.client is None:
            return
        with lock:
            state["object_position_m"] = [float(v) for v in transform.position]
            state["object_orientation_wxyz"] = [float(v) for v in transform.wxyz]
            update_cube()
            position.value = tuple(state["object_position_m"])
            rotation.value = tuple(_euler_xyz_deg_from_quat(state["object_orientation_wxyz"]))
            mark_dirty()

    @preview_scale.on_update
    def on_scale(_):
        cube_mesh.scale = preview_scale.value

    @save_button.on_click
    def on_save(_):
        with lock:
            try:
                payload = deepcopy(state)
                payload["enabled"] = True
                save_pose(payload, path)
                state["enabled"] = True
                status.content = "**已保存。** 新启动的训练和回放将使用此姿态。"
                print(f"[inhand-config] saved {path}", flush=True)
            except (ValueError, OSError) as error:
                status.content = f"**保存失败：** {error}"

    @reload_button.on_click
    def on_reload(_):
        with lock:
            try:
                restored = load_pose(path)
                state.clear()
                state.update(restored)
                for name, slider in sliders.items():
                    slider.value = math.degrees(state["joint_positions_rad"][name])
                update_hand()
                update_cube()
                transform.position, transform.wxyz = cube.position, cube.wxyz
                position.value = tuple(state["object_position_m"])
                rotation.value = tuple(_euler_xyz_deg_from_quat(state["object_orientation_wxyz"]))
                status.content = "已重新载入；保存姿态已启用。" if state["enabled"] else "已重新载入；当前使用原始随机初始化。"
            except (ValueError, OSError) as error:
                status.content = f"**载入失败：** {error}"

    @disable_button.on_click
    def on_disable(_):
        with lock:
            try:
                saved = load_pose(path)
                saved["enabled"] = False
                save_pose(saved, path)
                state["enabled"] = False
                status.content = "**已恢复原始随机初始化。** 保存姿态仍保留，可重新载入后再次保存启用。"
            except (ValueError, OSError) as error:
                status.content = f"**停用失败：** {error}"

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (-.35, .65, .88)
        client.camera.look_at = (.015, .065, .56)
        client.camera.up_direction = (0., 0., 1.)

    update_hand()
    print(f"[inhand-config] ready http://{args.host}:{server.get_port()}", flush=True)
    print(f"[inhand-config] config: {path}", flush=True)
    try:
        while True:
            time.sleep(.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--config", type=Path, default=DEFAULT_PATH)
    run_viser(parser.parse_args())


if __name__ == "__main__":
    main()
