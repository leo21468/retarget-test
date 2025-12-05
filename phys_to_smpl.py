# (only the full script; unchanged parts kept same as your last script)
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

# adjust this fallback to actual tokenhsi repo path
TOKENHSI_ROOT = "/home/leo/experiment/retarget/TokenHSI"
TOKENHSI_ROOT = pathlib.Path(TOKENHSI_ROOT).resolve()
sys.path.insert(0, str(TOKENHSI_ROOT))

from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from lpanlib.poselib.visualization.common import plot_skeleton_motion_interactive
from lpanlib.poselib.core.rotation3d import quat_mul_norm

from body_models.model_loader import get_body_model

from lpanlib.isaacgym_utils.vis.api import vis_motion_use_scenepic_animation
from lpanlib.others.colors import name_to_rgb

if __name__ == "__main__":
    # --- load skeletons ----------------------------------------------------
    phys_humanoid_v3_xml_path = osp.join(TOKENHSI_ROOT, "tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml")
    phys_humanoid_v3_skeleton = SkeletonTree.from_mjcf(phys_humanoid_v3_xml_path)

    # build an smpl-like skeleton tree (same as your original script)
    bm = get_body_model("SMPL", "NEUTRAL", batch_size=1, debug=False)
    jts_global_trans = bm().joints[0, :24, :].cpu().detach().numpy()
    jts_local_trans = np.zeros_like(jts_global_trans)
    for i in range(jts_local_trans.shape[0]):
        parent = bm.parents[i]
        if parent == -1:
            jts_local_trans[i] = jts_global_trans[i]
        else:
            jts_local_trans[i] = jts_global_trans[i] - jts_global_trans[parent]

    skel_dict = phys_humanoid_v3_skeleton.to_dict()  # reuse structure container
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
    # adjust arms similar to original pipeline (if needed)
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

    # --- IMPORTANT: inspect source/target node names -----------------------
    print("phys nodes (source):", phys_humanoid_v3_skeleton.node_names)
    print("smpl nodes (target):", smpl_original_skeleton.node_names)

    # --- joint_mapping: keys must be EXACT source names in phys_humanoid_v3_skeleton.node_names
    # Completed mapping: every phys node mapped to a corresponding SMPL node
    joint_mapping = {
        "pelvis": "Pelvis",
        "torso": "Spine",            # phys 'torso' -> SMPL 'Spine'
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

    # --- main loop: use SkeletonMotion.retarget_to (vectorized, recommended) ---
    for motion_file in tqdm(motion_files):
        phys_motion_path = osp.join(osp.dirname(__file__), motion_file)
        phys_motion = SkeletonMotion.from_file(phys_motion_path)
        fps = phys_motion.fps

        # Directly retarget the whole motion
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

        # ground correction
        if "stair" not in motion_file:
            min_h = torch.min(retargeted_motion.global_translation[:, :, 2], dim=-1)[0].mean()
        else:
            min_h = torch.min(retargeted_motion.global_translation[:, :, 2], dim=-1)[0].min()
        retargeted_motion.root_translation[:, 2] += -min_h

        save_path = osp.join(output_dir, osp.basename(motion_file).replace("phys_humanoid_v3", "smpl"))
        retargeted_motion.to_file(save_path)

        # convert local_rotation (xyzw) -> wxyz for quaternion->angle_axis
        poses_quat = retargeted_motion.local_rotation.clone()
        poses_quat = poses_quat[:, :, [3, 0, 1, 2]]  # xyzw -> wxyz
        poses_axis = tgm.quaternion_to_angle_axis(poses_quat.reshape(-1, 4)).reshape(poses_quat.shape[0], -1, 3)
        
        # Reshape poses from (num_frames, 24, 3) to (num_frames, 72) for SMPL format
        poses_reshaped = poses_axis.reshape(poses_axis.shape[0], -1)
        
        trans = retargeted_motion.root_translation.clone()
        params_save_path = save_path.replace("ref_motion.npy", "smpl_params.npy")
        
        # Save with proper shape validation
        try:
            np.save(params_save_path, {"poses": poses_reshaped.cpu().numpy(), "trans": trans.cpu().numpy(), "fps": fps})
            print(f"Saved smpl_params with poses shape: {poses_reshaped.shape}")
        except Exception as e:
            print(f"Error saving smpl_params to {params_save_path}: {e}")
            import traceback
            traceback.print_exc()

    print("Done")