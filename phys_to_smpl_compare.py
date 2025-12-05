#!/usr/bin/env python3
import sys
import pathlib
sys.path.append("./")

import os
import os.path as osp
import torch
import numpy as np
import yaml
import torchgeometry as tgm
from tqdm import tqdm

# rendering libs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio
from matplotlib import rcParams

# adjust this fallback to actual tokenhsi repo path
TOKENHSI_ROOT = "/home/leo/experiment/retarget/TokenHSI"
TOKENHSI_ROOT = pathlib.Path(TOKENHSI_ROOT).resolve()
sys.path.insert(0, str(TOKENHSI_ROOT))

from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from lpanlib.poselib.core.rotation3d import quat_mul_norm

from body_models.model_loader import get_body_model

# scenepic renderer (optional)
from lpanlib.isaacgym_utils.vis.api import vis_motion_use_scenepic_animation
from lpanlib.others.colors import name_to_rgb

# increase default figure DPI for nicer frames
rcParams["figure.dpi"] = 100

def render_skeleton_motion_to_video(motion: SkeletonMotion, skeleton: SkeletonTree, out_path: str, fps: int = 30,
                                    size=(640, 480), elev=20, azim=120, line_color="tab:blue", bgcolor="white"):
    """
    Render a SkeletonMotion into an MP4 (and GIF) using matplotlib 3D lines.
    - motion.global_translation: (T, N, 3)
    - skeleton.parent_indices gives the edges
    """
    # ensure numpy arrays
    pos = motion.global_translation.cpu().numpy()  # (T, N, 3)
    T, N, _ = pos.shape

    # compute stable axis limits from whole motion
    mins = pos.reshape(-1, 3).min(axis=0)
    maxs = pos.reshape(-1, 3).max(axis=0)
    spans = maxs - mins
    # add small padding
    pad = spans.max() * 0.15 if spans.max() > 0 else 0.5
    xlim = (mins[0] - pad, maxs[0] + pad)
    ylim = (mins[1] - pad, maxs[1] + pad)
    zlim = (mins[2] - pad, maxs[2] + pad)

    # edges from parent indices
    parent_indices = skeleton.parent_indices.numpy()
    edges = [(i, int(parent_indices[i])) for i in range(len(parent_indices)) if parent_indices[i] != -1]

    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', quality=8)

    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bgcolor)
    plt.tight_layout(pad=0)

    # static styling
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    # invert y if needed to match left-right orientation; keep as is for now
    ax.set_axis_off()

    for t in range(T):
        ax.cla()
        ax.set_facecolor(bgcolor)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_axis_off()

        pts = pos[t]  # (N,3)

        # draw joints
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=line_color, s=12)

        # draw bones
        for a, b in edges:
            pa = pts[a]
            pb = pts[b]
            xs = [pa[0], pb[0]]
            ys = [pa[1], pb[1]]
            zs = [pa[2], pb[2]]
            ax.plot(xs, ys, zs, c=line_color, linewidth=2)

        # optionally draw a ground plane grid
        # plane at z = zlim[0]
        # draw frame number
        ax.text2D(0.02, 0.95, f"frame {t+1}/{T}", transform=ax.transAxes)

        # capture frame
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        w, h = fig.canvas.get_width_height()
        img = img.reshape((h, w, 3))
        writer.append_data(img)

    writer.close()
    plt.close(fig)

    # also save a GIF next to mp4
    try:
        gif_path = osp.splitext(out_path)[0] + ".gif"
        with imageio.get_writer(gif_path, mode='I', duration=1.0 / fps) as gif_writer:
            vid = imageio.get_reader(out_path)
            for frame in vid:
                gif_writer.append_data(frame)
            vid.close()
    except Exception:
        # gif optional; ignore failures
        pass

if __name__ == "__main__":
    # --- load skeletons ----------------------------------------------------
    phys_humanoid_v3_xml_path = osp.join(TOKENHSI_ROOT, "tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml")
    phys_humanoid_v3_skeleton = SkeletonTree.from_mjcf(phys_humanoid_v3_xml_path)

    # build an smpl-like skeleton tree (same as original scripts)
    bm = get_body_model("SMPL", "NEUTRAL", batch_size=1, debug=False)
    jts_global_trans = bm().joints[0, :24, :].cpu().detach().numpy()
    jts_local_trans = np.zeros_like(jts_global_trans)
    for i in range(jts_local_trans.shape[0]):
        parent = bm.parents[i]
        if parent == -1:
            jts_local_trans[i] = jts_global_trans[i]
        else:
            jts_local_trans[i] = jts_global_trans[i] - jts_global_trans[parent]

    skel_dict = phys_humanoid_v3_skeleton.to_dict()
    skel_dict["node_names"] = [
        "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine", "L_Ankle", "R_Ankle",
        "Chest", "L_Toe", "R_Toe", "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
        "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
    ]
    skel_dict["parent_indices"]["arr"] = bm.parents.numpy()
    skel_dict["local_translation"]["arr"] = jts_local_trans
    smpl_original_skeleton = SkeletonTree.from_dict(skel_dict)

    # --- create tposes ----------------------------------------------------
    phys_humanoid_v3_tpose = SkeletonState.zero_pose(phys_humanoid_v3_skeleton)
    local_rotation = phys_humanoid_v3_tpose.local_rotation
    local_rotation[phys_humanoid_v3_skeleton.index("left_upper_arm")] = quat_mul_norm(
        torch.tensor([0.5, 0.5, 0.5, 0.5]), local_rotation[phys_humanoid_v3_skeleton.index("left_upper_arm")]
    )
    local_rotation[phys_humanoid_v3_skeleton.index("right_upper_arm")] = quat_mul_norm(
        torch.tensor([0.5, -0.5, -0.5, 0.5]), local_rotation[phys_humanoid_v3_skeleton.index("right_upper_arm")]
    )
    smpl_original_tpose = SkeletonState.zero_pose(smpl_original_skeleton)

    # --- load motion list from yaml (same as your file) --------------------
    yaml_path = osp.join(osp.dirname(__file__), "dataset_amass_loco.yaml")
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    motion_files = []
    for category, motions in cfg["motions"].items():
        for motion in motions:
            motion_files.append(motion["file"])

    output_dir = osp.join(osp.dirname(__file__), "smpl_motions_from_phys")
    os.makedirs(output_dir, exist_ok=True)

    # mapping: must match phys node names exactly
    joint_mapping = {
        "pelvis": "Pelvis",
        "torso": "Spine",
        "head": "Head",
        "right_upper_arm": "R_Shoulder",
        "right_lower_arm": "R_Elbow",
        "right_hand": "R_Wrist",
        "left_upper_arm": "L_Shoulder",
        "left_lower_arm": "L_Elbow",
        "left_hand": "L_Wrist",
        "right_thigh": "R_Hip",
        "right_shin": "R_Knee",
        "right_foot": "R_Ankle",
        "left_thigh": "L_Hip",
        "left_shin": "L_Knee",
        "left_foot": "L_Ankle",
    }

    for motion_file in tqdm(motion_files):
        phys_motion_path = osp.join(osp.dirname(__file__), motion_file)
        phys_motion = SkeletonMotion.from_file(phys_motion_path)
        fps = phys_motion.fps

        # retarget whole motion
        retargeted_motion = phys_motion.retarget_to(
            joint_mapping=joint_mapping,
            source_tpose_local_rotation=phys_humanoid_v3_tpose.local_rotation,
            source_tpose_root_translation=phys_humanoid_v3_tpose.root_translation,
            target_skeleton_tree=smpl_original_skeleton,
            target_tpose_local_rotation=smpl_original_tpose.local_rotation,
            target_tpose_root_translation=smpl_original_tpose.root_translation,
            rotation_to_target_skeleton=torch.tensor([-0.5, -0.5, -0.5, 0.5]),
            scale_to_target_skeleton=1.0,
            z_up=True,
        )

        # ground correction (same heuristics as original)
        if "stair" not in motion_file:
            min_h = torch.min(retargeted_motion.global_translation[:, :, 2], dim=-1)[0].mean()
        else:
            min_h = torch.min(retargeted_motion.global_translation[:, :, 2], dim=-1)[0].min()
        retargeted_motion.root_translation[:, 2] += -min_h

        # save retargeted motion
        save_name = osp.basename(motion_file).replace("phys_humanoid_v3", "smpl")
        save_path = osp.join(output_dir, save_name)
        retargeted_motion.to_file(save_path)

        # also save SMPL params (angle-axis poses + trans)
        poses_quat = retargeted_motion.local_rotation.clone()
        poses_quat = poses_quat[:, :, [3, 0, 1, 2]]  # xyzw -> wxyz
        poses_axis = tgm.quaternion_to_angle_axis(poses_quat.reshape(-1, 4)).reshape(poses_quat.shape[0], -1, 3)
        trans = retargeted_motion.root_translation.clone()
        params_save_path = save_path.replace("ref_motion.npy", "smpl_params.npy")
        np.save(params_save_path, {"poses": poses_axis.cpu().numpy(), "trans": trans.cpu().numpy(), "fps": fps})

        # -------------------- video visualization --------------------
        mp4_out = save_path.replace(".npy", "_smpl_render.mp4")
        try:
            render_skeleton_motion_to_video(retargeted_motion, smpl_original_skeleton, mp4_out, fps=min(30, int(fps)),
                                            size=(800, 600), elev=18, azim=120, line_color="tab:blue", bgcolor="white")
            print(f"Saved video: {mp4_out}")
        except Exception as e:
            print("Video rendering failed:", e)
            # fallback: save scenepic html if available
            try:
                smpl_xml_path = osp.join(TOKENHSI_ROOT, "tokenhsi/data/assets/mjcf/smpl_humanoid.xml")
                vis_motion_use_scenepic_animation(
                    asset_filename=smpl_xml_path,
                    rigidbody_global_pos=retargeted_motion.global_translation,
                    rigidbody_global_rot=retargeted_motion.global_rotation,
                    fps=fps,
                    up_axis="z",
                    color=name_to_rgb['AliceBlue'] * 255,
                    output_path=save_path.replace(".npy", "_render.html"),
                )
                print("Saved scenepic HTML as fallback.")
            except Exception as e2:
                print("Fallback scenepic render failed:", e2)

    print("Done")