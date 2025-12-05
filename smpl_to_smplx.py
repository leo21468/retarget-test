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
        
        # Debugging: Log shape of the input file
        print(f"Processing file: {input_path}")

        # Handle dict inside array
        if isinstance(smpl_data, np.ndarray) and smpl_data.dtype == object:
            try:
                data_dict = dict(smpl_data.item())
            except Exception as e:
                print(f"Failed to process smpl_data.item(): {e}")
                raise ValueError("Input file structure is invalid or corrupted.")
        else:
            data_dict = dict(smpl_data)

        # Handle betas padding for SMPL-X (pad from 10 to 16 if necessary)
        if 'betas' in data_dict:
            betas = data_dict['betas']
            print(f"Betas in the file: {betas}")  # Debugging output
            if betas.shape == (10,):
                if betas.dtype.kind != 'f':  # Check for floating-point type
                    print("Warning: Unexpected dtype for betas. Expected float.")
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
            print(f"Keys in the loaded file: {data_dict.keys()}")  # Debugging output
            raise ValueError("Input file does not contain 'poses' key. Is this an SMPL file?")

        poses = data_dict['poses']

        # Validate pose dimensions
        if poses.ndim not in [1, 2]:
            print(f"Unexpected dimensionality for poses: {poses.ndim}. Expected 1D or 2D arrays.")
            raise ValueError("Unexpected poses format. Ensure poses have 1D or 2D shape.")
        
        # Handle different pose dimensions
        if poses.ndim == 2 and poses.shape[1] > 72:
            poses = poses[:, :72]
        elif poses.ndim == 1 and poses.shape[0] > 72:
            poses = poses[:72]

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