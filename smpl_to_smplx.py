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
                print(f"ERROR: Failed to process smpl_data.item(): {e}")
                raise ValueError("Input file structure is invalid or corrupted.")
        else:
            data_dict = dict(smpl_data)
        
        # Log available keys for debugging
        print(f"Available keys in file: {list(data_dict.keys())}")

        # Handle betas padding for SMPL-X (pad from 10 to 16 if necessary)
        if 'betas' in data_dict:
            betas = data_dict['betas']
            print(f"Betas shape: {betas.shape}, dtype: {betas.dtype}")
            
            # Validate betas data type
            if betas.dtype.kind not in ['f', 'i', 'u']:  # float, int, or uint
                print(f"WARNING: Unexpected dtype for betas: {betas.dtype}. Expected numeric type.")
            
            # Handle 1D betas
            if betas.ndim == 1:
                if betas.shape[0] == 10:
                    data_dict['betas'] = np.concatenate([betas, np.zeros(6, dtype=betas.dtype)])
                    print(f"INFO: Padded betas from shape {betas.shape} to (16,)")
                elif betas.shape[0] == 16:
                    print(f"INFO: Betas already have correct shape (16,)")
                else:
                    print(f"WARNING: Unexpected betas shape: {betas.shape}. Expected (10,) or (16,).")
                    print(f"         Attempting to pad/truncate to (16,)...")
                    if betas.shape[0] < 16:
                        # Pad with zeros
                        padding = np.zeros(16 - betas.shape[0], dtype=betas.dtype)
                        data_dict['betas'] = np.concatenate([betas, padding])
                        print(f"         Padded from {betas.shape[0]} to 16 elements")
                    else:
                        # Truncate to 16
                        data_dict['betas'] = betas[:16]
                        print(f"         Truncated from {betas.shape[0]} to 16 elements")
            # Handle 2D betas
            elif betas.ndim == 2:
                if betas.shape[1] == 10:
                    data_dict['betas'] = np.concatenate([betas, np.zeros((betas.shape[0], 6), dtype=betas.dtype)], axis=1)
                    print(f"INFO: Padded betas from shape {betas.shape} to ({betas.shape[0]}, 16)")
                elif betas.shape[1] == 16:
                    print(f"INFO: Betas already have correct shape ({betas.shape[0]}, 16)")
                else:
                    print(f"WARNING: Unexpected betas shape: {betas.shape}. Expected (N, 10) or (N, 16).")
                    print(f"         Attempting to pad/truncate to (N, 16)...")
                    if betas.shape[1] < 16:
                        # Pad with zeros
                        padding = np.zeros((betas.shape[0], 16 - betas.shape[1]), dtype=betas.dtype)
                        data_dict['betas'] = np.concatenate([betas, padding], axis=1)
                        print(f"         Padded from shape {betas.shape} to ({betas.shape[0]}, 16)")
                    else:
                        # Truncate to 16
                        data_dict['betas'] = betas[:, :16]
                        print(f"         Truncated from shape {betas.shape} to ({betas.shape[0]}, 16)")
            else:
                print(f"ERROR: Betas has unexpected number of dimensions: {betas.ndim}. Expected 1 or 2.")
                raise ValueError(f"Betas must be 1D or 2D array, got shape {betas.shape}")

        # Handle mocap_frame_rate variations
        if 'mocap_framerate' in data_dict:
            data_dict['mocap_frame_rate'] = data_dict.pop('mocap_framerate')
            print(f"Renamed 'mocap_framerate' to 'mocap_frame_rate' for {input_path}")

        if 'poses' not in data_dict:
            print(f"ERROR: Missing 'poses' key in file.")
            print(f"       Available keys: {list(data_dict.keys())}")
            print(f"       This may not be a valid SMPL file.")
            raise ValueError("Input file does not contain 'poses' key. Is this an SMPL file?")

        poses = data_dict['poses']
        print(f"Poses shape: {poses.shape}, dtype: {poses.dtype}")

        # Validate pose dimensions
        if poses.ndim not in [1, 2]:
            print(f"ERROR: Unexpected dimensionality for poses: {poses.ndim}. Expected 1D or 2D arrays.")
            print(f"       Poses shape: {poses.shape}")
            raise ValueError("Unexpected poses format. Ensure poses have 1D or 2D shape.")
        
        # Check for empty poses
        if poses.size == 0:
            print(f"ERROR: Poses array is empty.")
            print(f"       Cannot convert empty motion data.")
            raise ValueError("Poses array is empty. Cannot convert empty motion data.")
        
        # Validate pose data contains valid numbers
        if not np.isfinite(poses).all():
            nan_count = np.isnan(poses).sum()
            inf_count = np.isinf(poses).sum()
            print(f"WARNING: Poses contain {nan_count} NaN and {inf_count} Inf values.")
            print(f"         This may cause issues in downstream processing.")
        
        # Handle different pose dimensions
        if poses.ndim == 2:
            if poses.shape[1] > 72:
                print(f"INFO: Truncating poses from {poses.shape[1]} to 72 dimensions (SMPL format)")
                poses = poses[:, :72]
            elif poses.shape[1] < 66:
                print(f"WARNING: Poses have only {poses.shape[1]} dimensions, expected at least 66 for SMPL body.")
        elif poses.ndim == 1:
            if poses.shape[0] > 72:
                print(f"INFO: Truncating poses from {poses.shape[0]} to 72 dimensions (SMPL format)")
                poses = poses[:72]
            elif poses.shape[0] < 66:
                print(f"WARNING: Poses have only {poses.shape[0]} dimensions, expected at least 66 for SMPL body.")

        # Map to SMPL-X format
        if poses.ndim == 2:
            data_dict['root_orient'] = poses[:, :3]
            data_dict['pose_body'] = poses[:, 3:66]  # 21 joints x 3 = 63, ignoring SMPL hand poses
            print(f"INFO: Extracted root_orient: {data_dict['root_orient'].shape}, pose_body: {data_dict['pose_body'].shape}")
        else:
            data_dict['root_orient'] = poses[:3]
            data_dict['pose_body'] = poses[3:66]
            print(f"INFO: Extracted root_orient: {data_dict['root_orient'].shape}, pose_body: {data_dict['pose_body'].shape}")
        
        # Validate trans if present
        if 'trans' in data_dict:
            trans = data_dict['trans']
            print(f"Trans shape: {trans.shape}, dtype: {trans.dtype}")
            if trans.ndim not in [1, 2]:
                print(f"WARNING: Unexpected trans dimensionality: {trans.ndim}. Expected 1D or 2D.")
            if trans.size > 0 and not np.isfinite(trans).all():
                print(f"WARNING: Trans contains NaN or Inf values.")

        # Ensure gender is set
        if 'gender' not in data_dict:
            data_dict['gender'] = np.array(gender)
            print(f"INFO: Set gender to '{gender}'")
        else:
            print(f"INFO: Gender already set to '{data_dict['gender']}'")

        # Remove original poses key
        del data_dict['poses']

        # Save as SMPL-X npy
        save_npy(output_path, data_dict, allow_overwrite=True)
        print(f"SUCCESS: Converted {input_path} to {output_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to convert {input_path}")
        print(f"       Exception: {e}")
        import traceback
        print("       Traceback:")
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