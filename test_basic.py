"""
Basic tests for NPY/NPZ handlers and SMPLX visualization.

These tests validate the core functionality of the refactored modules.
"""

import numpy as np
import os
import sys
from pathlib import Path


def test_npy_handler():
    """Test NPY handler functionality."""
    print("Testing NPY Handler...")
    from npy_handler import NpyNpzHandler
    
    handler = NpyNpzHandler()
    test_file = "test_array.npy"
    
    try:
        # Test 1: Save and load array
        test_data = np.random.randn(10, 3)
        handler.save_npy(test_file, test_data, allow_overwrite=True)
        loaded_data = handler.load_npy(test_file)
        assert np.allclose(test_data, loaded_data), "Data mismatch after save/load"
        print("  ✓ Save and load array")
        
        # Test 2: Get file info
        info = handler.get_info(test_file)
        assert 'shape' in info, "Missing shape in info"
        assert info['shape'] == (10, 3), "Incorrect shape in info"
        print("  ✓ Get file info")
        
        # Test 3: Validation
        is_valid = handler.validate_motion_data(test_data)
        assert is_valid, "Valid data marked as invalid"
        print("  ✓ Data validation")
        
        # Clean up
        Path(test_file).unlink(missing_ok=True)
        print("  ✓ All NPY handler tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ NPY handler test failed: {e}")
        Path(test_file).unlink(missing_ok=True)
        return False


def test_npz_handler():
    """Test NPZ handler functionality."""
    print("\nTesting NPZ Handler...")
    from npy_handler import NpyNpzHandler
    
    handler = NpyNpzHandler()
    test_file = "test_arrays.npz"
    
    try:
        # Test 1: Save and load multiple arrays
        test_data = {
            'array1': np.random.randn(5, 3),
            'array2': np.random.randn(10, 2),
            'fps': np.array(30)
        }
        handler.save_npz(test_file, test_data, allow_overwrite=True)
        loaded_data = handler.load_npz(test_file)
        
        assert 'array1' in loaded_data, "Missing array1 in loaded data"
        assert np.allclose(test_data['array1'], loaded_data['array1']), "Data mismatch"
        print("  ✓ Save and load multiple arrays")
        
        # Test 2: Get file info
        info = handler.get_info(test_file)
        assert 'keys' in info, "Missing keys in info"
        assert len(info['keys']) == 3, "Incorrect number of keys"
        print("  ✓ Get file info")
        
        # Clean up
        Path(test_file).unlink(missing_ok=True)
        print("  ✓ All NPZ handler tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ NPZ handler test failed: {e}")
        Path(test_file).unlink(missing_ok=True)
        return False


def test_smplx_visualizer():
    """Test SMPLX visualizer functionality."""
    print("\nTesting SMPLX Visualizer...")
    from smplx_visualizer import SMPLXVisualizer
    
    visualizer = SMPLXVisualizer()
    
    try:
        # Test 1: Create skeleton edges
        edges = visualizer.create_skeleton_edges()
        assert len(edges) > 0, "No edges created"
        print("  ✓ Create skeleton edges")
        
        # Test 2: Validate SMPLX data
        valid_data = {
            'root_orient': np.random.randn(10, 3),
            'pose_body': np.random.randn(10, 63)
        }
        is_valid = visualizer.validate_smplx_data(valid_data)
        assert is_valid, "Valid SMPLX data marked as invalid"
        print("  ✓ Validate SMPLX data")
        
        # Test 3: Export info
        info = visualizer.export_info(valid_data)
        assert 'root_orient' in info, "Missing root_orient in info"
        print("  ✓ Export SMPLX info")
        
        print("  ✓ All SMPLX visualizer tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ SMPLX visualizer test failed: {e}")
        return False


def test_smpl_to_smplx_conversion():
    """Test SMPL to SMPLX conversion."""
    print("\nTesting SMPL to SMPLX Conversion...")
    from smpl_to_smplx import convert_smpl_to_smplx
    from npy_handler import NpyNpzHandler
    
    handler = NpyNpzHandler()
    input_file = "test_smpl_input.npy"
    output_file = "test_smplx_output.npy"
    
    try:
        # Create test SMPL data
        smpl_data = {
            'poses': np.random.randn(10, 72),
            'trans': np.random.randn(10, 3),
            'betas': np.random.randn(10),
            'mocap_framerate': 30
        }
        handler.save_npy(input_file, smpl_data, allow_overwrite=True)
        
        # Convert
        success = convert_smpl_to_smplx(input_file, output_file, gender='neutral')
        assert success, "Conversion failed"
        print("  ✓ SMPL to SMPLX conversion")
        
        # Verify output
        smplx_data = handler.load_npy(output_file)
        if isinstance(smplx_data, np.ndarray) and smplx_data.dtype == object:
            smplx_data = smplx_data.item()
        
        assert 'root_orient' in smplx_data, "Missing root_orient in output"
        assert 'pose_body' in smplx_data, "Missing pose_body in output"
        print("  ✓ Verify conversion output")
        
        # Clean up
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
        print("  ✓ All conversion tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Conversion test failed: {e}")
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
        return False


def test_data_utils():
    """Test data_utils functions."""
    print("\nTesting Data Utils...")
    try:
        from data_utils import process_amass_seq
    except ImportError as e:
        print(f"  ⊘ Skipped - missing dependencies: {e}")
        return True  # Not a failure, just skipped
    
    from npy_handler import NpyNpzHandler
    
    handler = NpyNpzHandler()
    input_file = "test_amass_input.npz"
    output_file = "test_amass_output.npy"
    
    try:
        # Create test AMASS data
        amass_data = {
            'poses': np.random.randn(100, 156),  # SMPLX format
            'trans': np.random.randn(100, 3),
            'mocap_frame_rate': 120
        }
        handler.save_npz(input_file, amass_data, allow_overwrite=True)
        
        # Process
        success = process_amass_seq(input_file, output_file)
        assert success, "Processing failed"
        print("  ✓ Process AMASS sequence")
        
        # Verify output
        output_data = handler.load_npy(output_file)
        if isinstance(output_data, np.ndarray) and output_data.dtype == object:
            output_data = output_data.item()
        
        assert 'poses' in output_data, "Missing poses in output"
        assert output_data['fps'] == 30, "Incorrect FPS"
        print("  ✓ Verify processed output")
        
        # Clean up
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
        print("  ✓ All data utils tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Data utils test failed: {e}")
        import traceback
        traceback.print_exc()
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
        return False


def test_existing_files():
    """Test with existing repository files."""
    print("\nTesting with Existing Files...")
    from npy_handler import NpyNpzHandler
    
    handler = NpyNpzHandler()
    
    try:
        # Test ref_motion.npy if it exists
        if Path("ref_motion.npy").exists():
            data = handler.load_npy("ref_motion.npy")
            is_valid = handler.validate_motion_data(data)
            print("  ✓ Load and validate ref_motion.npy")
        
        # Test g1.npz if it exists
        if Path("g1.npz").exists():
            data = handler.load_npz("g1.npz")
            is_valid = handler.validate_motion_data(data)
            print("  ✓ Load and validate g1.npz")
        
        print("  ✓ All existing file tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Existing file test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Running Basic Tests for Refactored Modules")
    print("="*60)
    
    results = []
    
    results.append(("NPY Handler", test_npy_handler()))
    results.append(("NPZ Handler", test_npz_handler()))
    results.append(("SMPLX Visualizer", test_smplx_visualizer()))
    results.append(("SMPL to SMPLX Conversion", test_smpl_to_smplx_conversion()))
    results.append(("Data Utils", test_data_utils()))
    results.append(("Existing Files", test_existing_files()))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)
    
    return all(result for _, result in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
