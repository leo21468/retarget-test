import os
import argparse
import numpy as np

# Make tqdm optional
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

def convert_smpl_to_smplx(input_path, output_path, gender='neutral'):
    """
    Convert SMPL format motion data to SMPL-X format.
    
    Args:
        input_path: Path to input SMPL file
        output_path: Path to save SMPL-X file
        gender: Gender for SMPL-X model ('male', 'female', or 'neutral')
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from npy_handler import load_npy, save_npy
        
        # Load SMPL data
        smpl_data = load_npy(input_path)
        
        # Handle dict inside array
        if isinstance(smpl_data, np.ndarray) and smpl_data.dtype == object:
            data_dict = dict(smpl_data.item())
        else:
            data_dict = dict(smpl_data)

        # Handle betas padding for SMPL-X (pad from 10 to 16 if necessary)
        if 'betas' in data_dict:
            betas = data_dict['betas']
            if betas.shape == (10,):
                data_dict['betas'] = np.concatenate([betas, np.zeros(6, dtype=betas.dtype)])
                print(f"Padded betas from 10 to 16 for {input_path}")
            elif betas.ndim == 2 and betas.shape[1] == 10:
                data_dict['betas'] = np.concatenate([betas, np.zeros((betas.shape[0], 6), dtype=betas.dtype)], axis=1)
                print(f"Padded betas from 10 to 16 for {input_path}")
            elif betas.shape not in [(16,), (1, 16)]:
                if betas.ndim == 2 and betas.shape[1] != 16:
                    print(f"Warning: Unexpected betas shape: {betas.shape}. Expected (10,), (16,), or (N,10/16)")

        # Handle mocap_frame_rate variations
        if 'mocap_framerate' in data_dict:
            data_dict['mocap_frame_rate'] = data_dict.pop('mocap_framerate')
            print(f"Renamed 'mocap_framerate' to 'mocap_frame_rate' for {input_path}")

        if 'poses' not in data_dict:
            print(f"Error: Input file does not contain 'poses' key.")
            print(f"Available keys: {list(data_dict.keys())}")
            print(f"This might be a motion file (ref_motion.npy) rather than an SMPL params file.")
            print(f"Hint: Make sure you're converting the correct file (usually smpl_params.npy, not ref_motion.npy)")
            raise ValueError("Input file does not contain 'poses' key. Is this an SMPL file?")

        poses = data_dict['poses']
        
        # Handle different pose dimensions and shapes
        if poses.ndim == 3:
            # Handle shape (N, 24, 3) - reshape to (N, 72)
            if poses.shape[1] == 24 and poses.shape[2] == 3:
                print(f"Detected poses shape {poses.shape}, reshaping from (N, 24, 3) to (N, 72)")
                poses = poses.reshape(poses.shape[0], -1)
            else:
                print(f"Warning: Unexpected 3D poses shape: {poses.shape}")
                poses = poses.reshape(poses.shape[0], -1)
        elif poses.ndim == 2 and poses.shape[1] > 72:
            poses = poses[:, :72]
        elif poses.ndim == 1 and poses.shape[0] > 72:
            poses = poses[:72]
        
        # Validate poses shape after processing
        if poses.ndim == 2 and poses.shape[1] != 72:
            print(f"Warning: Expected poses shape (N, 72), got {poses.shape}")
            if poses.shape[1] < 72:
                # Pad with zeros if too short
                padding = np.zeros((poses.shape[0], 72 - poses.shape[1]), dtype=poses.dtype)
                poses = np.concatenate([poses, padding], axis=1)
                print(f"Padded poses to shape {poses.shape}")

        # Map to SMPL-X format
        if poses.ndim == 2:
            data_dict['root_orient'] = poses[:, :3]
            data_dict['pose_body'] = poses[:, 3:66]  # 21 joints x 3 = 63, ignoring SMPL hand poses
        else:
            data_dict['root_orient'] = poses[:3]
            data_dict['pose_body'] = poses[3:66]

        # Ensure gender is set
        if 'gender' not in data_dict:
            data_dict['gender'] = np.array(gender)

        # Remove original poses key
        del data_dict['poses']

        # Save as SMPL-X npy
        save_npy(output_path, data_dict, allow_overwrite=True)
        print(f"Converted {input_path} to {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_directory(src_folder, tgt_folder, gender='neutral'):
    os.makedirs(tgt_folder, exist_ok=True)
    for filename in tqdm(os.listdir(src_folder)):
        if filename.endswith('.npy'):
            input_path = os.path.join(src_folder, filename)
            output_path = os.path.join(tgt_folder, filename)
            convert_smpl_to_smplx(input_path, output_path, gender)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SMPL motion data to SMPL-X format.")
    parser.add_argument("--src_folder", type=str, help="Source directory of SMPL .npy files")
    parser.add_argument("--tgt_folder", type=str, help="Target directory for SMPL-X .npy files")
    parser.add_argument("--input_file", type=str, help="Single input SMPL .npy file")
    parser.add_argument("--output_file", type=str, help="Single output SMPL-X .npy file")
    parser.add_argument("--gender", type=str, default="neutral", choices=["male", "female", "neutral"],
                        help="Gender for SMPL-X model if not present in file.")
    args = parser.parse_args()

    if args.src_folder and args.tgt_folder:
        process_directory(args.src_folder, args.tgt_folder, args.gender)
    elif args.input_file and args.output_file:
        convert_smpl_to_smplx(args.input_file, args.output_file, args.gender)
    else:
        parser.print_help()
