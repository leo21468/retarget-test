import sys
sys.path.append("./")

import os
import pathlib
import numpy as np
import torch

# adjust this fallback to actual TokenHSI repo path
tokenhsi_ROOT = "/home/leo/experiment/retarget/TokenHSI"
tokenhsi_ROOT = pathlib.Path(tokenhsi_ROOT).resolve()
sys.path.insert(0, str(tokenhsi_ROOT))

from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from lpanlib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from lpanlib.poselib.core.rotation3d import quat_mul, quat_from_angle_axis, quat_mul_norm, quat_rotate, quat_identity

def process_amass_seq(fname, output_path):
    """
    Process AMASS sequence data and convert to target format.
    
    Args:
        fname: Input file path
        output_path: Output file path
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # load raw params from AMASS dataset
        from npy_handler import load_npz, save_npy
        
        # Load file based on extension
        if fname.endswith('.npz'):
            raw_params = load_npz(fname)
        elif fname.endswith('.npy'):
            raw_params = dict(np.load(fname, allow_pickle=True))
        else:
            raise ValueError(f"Unsupported file format: {fname}")

        # Validate required keys
        if "poses" not in raw_params:
            print(f"Error: Missing 'poses' key in {fname}")
            print(f"Available keys: {list(raw_params.keys())}")
            print(f"Hint: This file may not be in SMPL format. Check if it's the correct input file.")
            raise ValueError(f"Missing required key 'poses' in {fname}")
        
        if "trans" not in raw_params:
            print(f"Error: Missing 'trans' key in {fname}")
            print(f"Available keys: {list(raw_params.keys())}")
            print(f"Hint: This file may not be in SMPL format. Check if it's the correct input file.")
            raise ValueError(f"Missing required key 'trans' in {fname}")

        poses = raw_params["poses"]  # rotations
        trans = raw_params["trans"]  # translations
        
        # Validate poses shape
        if poses.ndim == 3 and poses.shape[1] == 24 and poses.shape[2] == 3:
            print(f"Warning: Detected poses with shape {poses.shape}, reshaping to (N, 72)")
            poses = poses.reshape(poses.shape[0], -1)

        # downsample from source fps to 30hz
        source_fps = raw_params.get("mocap_frame_rate", raw_params.get("mocap_framerate", 30))
        target_fps = 30
        
        # Validate source_fps to prevent invalid skip values
        if not isinstance(source_fps, (int, float)) or source_fps <= 0:
            print(f"Warning: Invalid source_fps {source_fps}, using default 30")
            source_fps = 30
        
        skip = max(1, int(source_fps // target_fps))
        
        # Ensure we have data to downsample
        if len(poses) == 0 or len(trans) == 0:
            raise ValueError(f"Empty poses or trans arrays in {fname}")
        
        poses = poses[::skip]
        trans = trans[::skip]

        # extract 24 SMPL joints from 55 SMPL-X joints
        joints_to_use = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 40]
        )
        joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)  # convert joint indices to 72 indexes (3*24)
        
        # Handle different pose dimensions with bounds checking
        if poses.shape[1] >= 156:
            # Validate that indices are within bounds
            if np.max(joints_to_use) >= poses.shape[1]:
                print(f"Error: Index out of bounds. Max index {np.max(joints_to_use)} >= poses shape[1] {poses.shape[1]}")
                raise IndexError(f"Joint indices out of bounds for poses shape {poses.shape}")
            poses = poses[:, joints_to_use]  # take out corresponding x, y, z rotations
        elif poses.shape[1] == 72:
            # Already in SMPL format
            pass
        else:
            print(f"Warning: Unexpected pose dimension {poses.shape[1]}, keeping as is")

        required_params = {}
        required_params["poses"] = poses
        required_params["trans"] = trans
        required_params["fps"] = target_fps
        
        # save
        save_npy(output_path, required_params, allow_overwrite=True)
        
        return True
        
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        import traceback
        traceback.print_exc()
        return False

def project_joints_simple(motion):
    """ This is the our revised function used by TokenHSI, designed for phys_humanoid_v3.xml 

    The difference is that we only project the arms, not the legs.
    The reason is that the leg joints have been modified to 3 DoF spherical joints.

    """

    right_upper_arm_id = motion.skeleton_tree._node_indices["right_upper_arm"]
    right_lower_arm_id = motion.skeleton_tree._node_indices["right_lower_arm"]
    right_hand_id = motion.skeleton_tree._node_indices["right_hand"]
    left_upper_arm_id = motion.skeleton_tree._node_indices["left_upper_arm"]
    left_lower_arm_id = motion.skeleton_tree._node_indices["left_lower_arm"]
    left_hand_id = motion.skeleton_tree._node_indices["left_hand"]
    
    right_thigh_id = motion.skeleton_tree._node_indices["right_thigh"] # upper leg
    right_shin_id = motion.skeleton_tree._node_indices["right_shin"] # lower leg
    right_foot_id = motion.skeleton_tree._node_indices["right_foot"]
    left_thigh_id = motion.skeleton_tree._node_indices["left_thigh"]
    left_shin_id = motion.skeleton_tree._node_indices["left_shin"]
    left_foot_id = motion.skeleton_tree._node_indices["left_foot"]
    
    device = motion.global_translation.device

    # right arm
    right_upper_arm_pos = motion.global_translation[..., right_upper_arm_id, :]
    right_lower_arm_pos = motion.global_translation[..., right_lower_arm_id, :]
    right_hand_pos = motion.global_translation[..., right_hand_id, :]
    right_shoulder_rot = motion.local_rotation[..., right_upper_arm_id, :]
    right_elbow_rot = motion.local_rotation[..., right_lower_arm_id, :]
    
    right_arm_delta0 = right_upper_arm_pos - right_lower_arm_pos
    right_arm_delta1 = right_hand_pos - right_lower_arm_pos
    right_arm_delta0 = right_arm_delta0 / torch.norm(right_arm_delta0, dim=-1, keepdim=True)
    right_arm_delta1 = right_arm_delta1 / torch.norm(right_arm_delta1, dim=-1, keepdim=True)
    right_elbow_dot = torch.sum(-right_arm_delta0 * right_arm_delta1, dim=-1)
    right_elbow_dot = torch.clamp(right_elbow_dot, -1.0, 1.0)
    right_elbow_theta = torch.acos(right_elbow_dot)
    right_elbow_q = quat_from_angle_axis(-torch.abs(right_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                            device=device, dtype=torch.float32))
    
    right_elbow_local_dir = motion.skeleton_tree.local_translation[right_hand_id]
    right_elbow_local_dir = right_elbow_local_dir / torch.norm(right_elbow_local_dir)
    right_elbow_local_dir_tile = torch.tile(right_elbow_local_dir.unsqueeze(0), [right_elbow_rot.shape[0], 1])
    right_elbow_local_dir0 = quat_rotate(right_elbow_rot, right_elbow_local_dir_tile)
    right_elbow_local_dir1 = quat_rotate(right_elbow_q, right_elbow_local_dir_tile)
    right_arm_dot = torch.sum(right_elbow_local_dir0 * right_elbow_local_dir1, dim=-1)
    right_arm_dot = torch.clamp(right_arm_dot, -1.0, 1.0)
    right_arm_theta = torch.acos(right_arm_dot)
    right_arm_theta = torch.where(right_elbow_local_dir0[..., 1] <= 0, right_arm_theta, -right_arm_theta)
    right_arm_q = quat_from_angle_axis(right_arm_theta, right_elbow_local_dir.unsqueeze(0))
    right_shoulder_rot = quat_mul(right_shoulder_rot, right_arm_q)
    
    # left arm
    left_upper_arm_pos = motion.global_translation[..., left_upper_arm_id, :]
    left_lower_arm_pos = motion.global_translation[..., left_lower_arm_id, :]
    left_hand_pos = motion.global_translation[..., left_hand_id, :]
    left_shoulder_rot = motion.local_rotation[..., left_upper_arm_id, :]
    left_elbow_rot = motion.local_rotation[..., left_lower_arm_id, :]
    
    left_arm_delta0 = left_upper_arm_pos - left_lower_arm_pos
    left_arm_delta1 = left_hand_pos - left_lower_arm_pos
    left_arm_delta0 = left_arm_delta0 / torch.norm(left_arm_delta0, dim=-1, keepdim=True)
    left_arm_delta1 = left_arm_delta1 / torch.norm(left_arm_delta1, dim=-1, keepdim=True)
    left_elbow_dot = torch.sum(-left_arm_delta0 * left_arm_delta1, dim=-1)
    left_elbow_dot = torch.clamp(left_elbow_dot, -1.0, 1.0)
    left_elbow_theta = torch.acos(left_elbow_dot)
    left_elbow_q = quat_from_angle_axis(-torch.abs(left_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))

    left_elbow_local_dir = motion.skeleton_tree.local_translation[left_hand_id]
    left_elbow_local_dir = left_elbow_local_dir / torch.norm(left_elbow_local_dir)
    left_elbow_local_dir_tile = torch.tile(left_elbow_local_dir.unsqueeze(0), [left_elbow_rot.shape[0], 1])
    left_elbow_local_dir0 = quat_rotate(left_elbow_rot, left_elbow_local_dir_tile)
    left_elbow_local_dir1 = quat_rotate(left_elbow_q, left_elbow_local_dir_tile)
    left_arm_dot = torch.sum(left_elbow_local_dir0 * left_elbow_local_dir1, dim=-1)
    left_arm_dot = torch.clamp(left_arm_dot, -1.0, 1.0)
    left_arm_theta = torch.acos(left_arm_dot)
    left_arm_theta = torch.where(left_elbow_local_dir0[..., 1] <= 0, left_arm_theta, -left_arm_theta)
    left_arm_q = quat_from_angle_axis(left_arm_theta, left_elbow_local_dir.unsqueeze(0))
    left_shoulder_rot = quat_mul(left_shoulder_rot, left_arm_q)

    new_local_rotation = motion.local_rotation.clone()
    new_local_rotation[..., right_upper_arm_id, :] = right_shoulder_rot
    new_local_rotation[..., right_lower_arm_id, :] = right_elbow_q
    new_local_rotation[..., left_upper_arm_id, :] = left_shoulder_rot
    new_local_rotation[..., left_lower_arm_id, :] = left_elbow_q
    
    new_local_rotation[..., left_hand_id, :] = quat_identity([1])
    new_local_rotation[..., right_hand_id, :] = quat_identity([1])

    new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, new_local_rotation, motion.root_translation, is_local=True)
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)

    return new_motion
