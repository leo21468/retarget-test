"""
Examples demonstrating the usage of NPY/NPZ handlers and SMPLX visualization.

This script provides practical examples of:
1. Reading and inspecting NPY/NPZ files
2. Converting between formats
3. Visualizing SMPLX data
4. Error handling and validation
"""

import sys
from pathlib import Path
import numpy as np


def example_1_read_npy_file():
    """Example 1: Read and inspect a .npy file"""
    print("\n" + "="*60)
    print("Example 1: Reading NPY Files")
    print("="*60)
    
    from npy_handler import NpyNpzHandler
    
    handler = NpyNpzHandler(allow_pickle=True)
    
    # Check if ref_motion.npy exists
    filepath = Path("ref_motion.npy")
    if filepath.exists():
        try:
            # Load the file
            data = handler.load_npy(filepath)
            print(f"✓ Loaded {filepath}")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            
            # Get detailed info
            info = handler.get_info(filepath)
            print(f"  Size: {info['size_bytes']:,} bytes")
            
            # Validate
            is_valid = handler.validate_motion_data(data)
            print(f"  Validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print(f"File not found: {filepath}")
        print("Creating example .npy file...")
        
        # Create example data
        example_data = np.random.randn(100, 24, 3)
        handler.save_npy("example_motion.npy", example_data, allow_overwrite=True)
        print("✓ Created example_motion.npy")


def example_2_read_npz_file():
    """Example 2: Read and inspect a .npz file"""
    print("\n" + "="*60)
    print("Example 2: Reading NPZ Files")
    print("="*60)
    
    from npy_handler import NpyNpzHandler
    
    handler = NpyNpzHandler(allow_pickle=True)
    
    # Check if g1.npz exists
    filepath = Path("g1.npz")
    if filepath.exists():
        try:
            # Load the file
            data = handler.load_npz(filepath)
            print(f"✓ Loaded {filepath}")
            print(f"  Keys: {list(data.keys())}")
            
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: type={type(value).__name__}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print(f"File not found: {filepath}")
        print("Creating example .npz file...")
        
        # Create example data
        example_data = {
            'poses': np.random.randn(100, 72),
            'trans': np.random.randn(100, 3),
            'fps': np.array(30)
        }
        handler.save_npz("example_data.npz", example_data, allow_overwrite=True)
        print("✓ Created example_data.npz")


def example_3_smplx_conversion():
    """Example 3: Convert SMPL to SMPLX format"""
    print("\n" + "="*60)
    print("Example 3: SMPL to SMPLX Conversion")
    print("="*60)
    
    from smpl_to_smplx import convert_smpl_to_smplx
    from npy_handler import NpyNpzHandler
    
    # Create example SMPL data
    handler = NpyNpzHandler()
    
    example_smpl = {
        'poses': np.random.randn(100, 72),  # 24 joints x 3 DOF
        'trans': np.random.randn(100, 3),
        'betas': np.random.randn(10),  # Shape parameters
        'mocap_framerate': 30
    }
    
    input_path = "example_smpl.npy"
    output_path = "example_smplx.npy"
    
    print("Creating example SMPL file...")
    handler.save_npy(input_path, example_smpl, allow_overwrite=True)
    print(f"✓ Created {input_path}")
    
    print(f"Converting to SMPLX format...")
    success = convert_smpl_to_smplx(input_path, output_path, gender='neutral')
    
    if success:
        print(f"✓ Converted to {output_path}")
        
        # Verify the output
        smplx_data = handler.load_npy(output_path)
        if isinstance(smplx_data, np.ndarray) and smplx_data.dtype == object:
            smplx_data = smplx_data.item()
        
        print(f"SMPLX file contains:")
        for key in smplx_data.keys():
            if hasattr(smplx_data[key], 'shape'):
                print(f"  {key}: shape={smplx_data[key].shape}")
            else:
                print(f"  {key}: {smplx_data[key]}")
    else:
        print("✗ Conversion failed")


def example_4_smplx_info():
    """Example 4: Get SMPLX file information"""
    print("\n" + "="*60)
    print("Example 4: SMPLX File Information")
    print("="*60)
    
    from smplx_visualizer import get_smplx_info
    
    # Try to load existing SMPLX file or use example
    test_files = ["example_smplx.npy", "ref_motion.npy"]
    
    for filepath in test_files:
        if Path(filepath).exists():
            try:
                print(f"\nAnalyzing {filepath}:")
                info = get_smplx_info(filepath)
                
                for key, value in info.items():
                    print(f"  {key}:")
                    if isinstance(value, dict):
                        for subkey, subval in value.items():
                            print(f"    {subkey}: {subval}")
                    else:
                        print(f"    {value}")
                
                break
            except Exception as e:
                print(f"  ✗ Error: {e}")
    else:
        print("No SMPLX files found to analyze")


def example_5_error_handling():
    """Example 5: Demonstrate error handling"""
    print("\n" + "="*60)
    print("Example 5: Error Handling")
    print("="*60)
    
    from npy_handler import NpyNpzHandler
    
    handler = NpyNpzHandler()
    
    # Test 1: Non-existent file
    print("\nTest 1: Loading non-existent file")
    try:
        handler.load_npy("nonexistent.npy")
        print("  ✗ Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"  ✓ Correctly caught: {e.__class__.__name__}")
    
    # Test 2: Wrong extension
    print("\nTest 2: Wrong file extension")
    try:
        handler.load_npy("file.txt")
        print("  ✗ Should have raised ValueError")
    except (ValueError, FileNotFoundError) as e:
        print(f"  ✓ Correctly caught: {e.__class__.__name__}")
    
    # Test 3: Overwrite protection
    print("\nTest 3: Overwrite protection")
    test_data = np.array([1, 2, 3])
    test_path = "test_overwrite.npy"
    
    handler.save_npy(test_path, test_data, allow_overwrite=True)
    print(f"  ✓ Created {test_path}")
    
    try:
        handler.save_npy(test_path, test_data, allow_overwrite=False)
        print("  ✗ Should have raised FileExistsError")
    except FileExistsError as e:
        print(f"  ✓ Correctly caught: {e.__class__.__name__}")
    
    # Clean up
    Path(test_path).unlink(missing_ok=True)
    print(f"  ✓ Cleaned up {test_path}")


def example_6_validation():
    """Example 6: Data validation"""
    print("\n" + "="*60)
    print("Example 6: Data Validation")
    print("="*60)
    
    from npy_handler import NpyNpzHandler
    
    handler = NpyNpzHandler()
    
    # Test valid data
    print("\nTest 1: Valid array")
    valid_data = np.random.randn(10, 3)
    is_valid = handler.validate_motion_data(valid_data)
    print(f"  Result: {'✓ VALID' if is_valid else '✗ INVALID'}")
    
    # Test empty array
    print("\nTest 2: Empty array")
    empty_data = np.array([])
    is_valid = handler.validate_motion_data(empty_data)
    print(f"  Result: {'✓ VALID' if is_valid else '✗ INVALID (expected)'}")
    
    # Test data with NaN
    print("\nTest 3: Array with NaN")
    nan_data = np.array([1.0, 2.0, np.nan, 4.0])
    is_valid = handler.validate_motion_data(nan_data)
    print(f"  Result: {'✓ VALID' if is_valid else '✗ INVALID (expected)'}")
    
    # Test valid dict
    print("\nTest 4: Valid dictionary")
    valid_dict = {
        'poses': np.random.randn(10, 72),
        'trans': np.random.randn(10, 3)
    }
    is_valid = handler.validate_motion_data(valid_dict, expected_keys=['poses', 'trans'])
    print(f"  Result: {'✓ VALID' if is_valid else '✗ INVALID'}")
    
    # Test dict with missing keys
    print("\nTest 5: Dictionary with missing keys")
    incomplete_dict = {'poses': np.random.randn(10, 72)}
    is_valid = handler.validate_motion_data(incomplete_dict, expected_keys=['poses', 'trans'])
    print(f"  Result: {'✓ VALID' if is_valid else '✗ INVALID (expected)'}")


def main():
    """Run all examples"""
    print("\n" + "#"*60)
    print("# NPY/NPZ Handler and SMPLX Visualization Examples")
    print("#"*60)
    
    examples = [
        ("Read NPY Files", example_1_read_npy_file),
        ("Read NPZ Files", example_2_read_npz_file),
        ("SMPL to SMPLX Conversion", example_3_smplx_conversion),
        ("SMPLX File Information", example_4_smplx_info),
        ("Error Handling", example_5_error_handling),
        ("Data Validation", example_6_validation),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"\n✗ Example {i} ({name}) failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "#"*60)
    print("# Examples Complete!")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
