"""
Enhanced NPY/NPZ File Reader

This script demonstrates reading and inspecting .npy and .npz files
with proper error handling and detailed information display.
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from npy_handler import NpyNpzHandler


def read_and_display_file(filepath: str, verbose: bool = False):
    """
    Read and display information about a .npy or .npz file.
    
    Args:
        filepath: Path to the file
        verbose: Whether to show detailed information
    """
    handler = NpyNpzHandler(allow_pickle=True)
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return False
    
    try:
        # Get file info
        print(f"\n{'='*60}")
        print(f"File: {filepath}")
        print(f"{'='*60}")
        
        info = handler.get_info(filepath)
        print(f"Size: {info['size_bytes']:,} bytes ({info['size_bytes']/1024/1024:.2f} MB)")
        print(f"Type: {info['extension']}")
        
        if filepath.suffix.lower() == '.npy':
            # Load and display npy file
            data = handler.load_npy(filepath)
            print(f"\nData Type: {data.dtype}")
            print(f"Array Shape: {data.shape}")
            print(f"Number of Elements: {data.size:,}")
            
            if data.size > 0:
                print(f"\nValue Statistics:")
                if np.issubdtype(data.dtype, np.floating):
                    print(f"  Min: {data.min():.6f}")
                    print(f"  Max: {data.max():.6f}")
                    print(f"  Mean: {data.mean():.6f}")
                    print(f"  Std: {data.std():.6f}")
                
                if verbose and data.size < 100:
                    print(f"\nData Content:")
                    print(data)
                elif verbose:
                    print(f"\nFirst few elements:")
                    print(data.flat[:10])
            
            # If it's a dict inside the array
            if data.dtype == object and isinstance(data.item() if data.shape == () else data[0], dict):
                print("\nDetected dictionary structure:")
                data_dict = data.item() if data.shape == () else data[0]
                for key, value in data_dict.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: {type(value).__name__} = {value}")
        
        elif filepath.suffix.lower() == '.npz':
            # Load and display npz file
            data = handler.load_npz(filepath)
            print(f"\nNumber of Arrays: {len(data)}")
            print(f"Keys: {list(data.keys())}")
            
            for key, arr in data.items():
                print(f"\n{key}:")
                if hasattr(arr, 'shape'):
                    print(f"  Shape: {arr.shape}")
                    print(f"  Dtype: {arr.dtype}")
                    print(f"  Size: {arr.size:,} elements")
                    
                    if arr.size > 0 and np.issubdtype(arr.dtype, np.floating):
                        print(f"  Min: {arr.min():.6f}")
                        print(f"  Max: {arr.max():.6f}")
                        print(f"  Mean: {arr.mean():.6f}")
                    
                    if verbose and arr.size < 100:
                        print(f"  Data: {arr}")
                else:
                    print(f"  Type: {type(arr).__name__}")
                    print(f"  Value: {arr}")
        
        # Validation
        print(f"\n{'='*60}")
        is_valid = handler.validate_motion_data(data)
        print(f"Validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Read and display information about .npy and .npz files"
    )
    parser.add_argument(
        'filepath',
        type=str,
        nargs='?',
        default='ref_motion.npy',
        help='Path to .npy or .npz file (default: ref_motion.npy)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information including data values'
    )
    
    args = parser.parse_args()
    
    success = read_and_display_file(args.filepath, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()