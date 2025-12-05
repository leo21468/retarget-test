"""
Test suite for debugging improvements in smpl_to_smplx.py

This test suite validates the enhanced error-checking mechanisms,
detailed logging, and graceful handling of malformed data.
"""

import numpy as np
import os
import sys
import io
from pathlib import Path
from smpl_to_smplx import convert_smpl_to_smplx
from npy_handler import save_npy, load_npy


def test_empty_poses_rejection():
    """Test that empty pose arrays are properly rejected."""
    print("\n=== Test: Empty Poses Rejection ===")
    
    input_file = "test_empty_poses.npy"
    output_file = "test_empty_poses_out.npy"
    
    try:
        # Create test data with empty poses
        test_data = {
            'poses': np.array([]),
            'trans': np.array([]),
            'betas': np.random.randn(10)
        }
        save_npy(input_file, test_data, allow_overwrite=True)
        
        # Attempt conversion (should fail)
        success = convert_smpl_to_smplx(input_file, output_file)
        
        assert not success, "Empty poses should fail conversion"
        print("  ✓ Empty poses properly rejected")
        
        # Clean up
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
        return False


def test_malformed_betas_handling():
    """Test that betas with unexpected shapes are handled gracefully."""
    print("\n=== Test: Malformed Betas Handling ===")
    
    test_cases = [
        ("5 elements", 5),
        ("3 elements", 3),
        ("20 elements", 20),
        ("16 elements", 16),  # Valid case
    ]
    
    all_passed = True
    
    for desc, size in test_cases:
        input_file = f"test_betas_{size}.npy"
        output_file = f"test_betas_{size}_out.npy"
        
        try:
            # Create test data with specific betas size
            test_data = {
                'poses': np.random.randn(10, 72),
                'trans': np.random.randn(10, 3),
                'betas': np.random.randn(size)
            }
            save_npy(input_file, test_data, allow_overwrite=True)
            
            # Convert
            success = convert_smpl_to_smplx(input_file, output_file)
            
            if success:
                # Verify betas were padded/truncated to 16
                output_data = load_npy(output_file)
                if isinstance(output_data, np.ndarray) and output_data.dtype == object:
                    output_data = output_data.item()
                
                assert 'betas' in output_data, "Missing betas in output"
                assert output_data['betas'].shape[0] == 16, f"Betas not converted to 16 elements, got {output_data['betas'].shape}"
                print(f"  ✓ Betas with {desc} handled correctly")
            else:
                print(f"  ✗ Failed to convert betas with {desc}")
                all_passed = False
            
            # Clean up
            Path(input_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)
            
        except Exception as e:
            print(f"  ✗ Test failed for {desc}: {e}")
            Path(input_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)
            all_passed = False
    
    return all_passed


def test_nan_inf_detection():
    """Test that NaN and Inf values in poses are detected and logged."""
    print("\n=== Test: NaN/Inf Detection ===")
    
    input_file = "test_nan_inf.npy"
    output_file = "test_nan_inf_out.npy"
    
    try:
        # Create test data with NaN and Inf
        test_data = {
            'poses': np.random.randn(10, 72),
            'trans': np.random.randn(10, 3),
            'betas': np.random.randn(10)
        }
        test_data['poses'][2, 5] = np.nan
        test_data['poses'][3, 10] = np.inf
        save_npy(input_file, test_data, allow_overwrite=True)
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        # Convert
        success = convert_smpl_to_smplx(input_file, output_file)
        
        # Restore stdout
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        
        # Check that warning was logged
        assert "WARNING: Poses contain" in output, "NaN/Inf warning not logged"
        assert "NaN" in output, "NaN count not in warning"
        assert "Inf" in output, "Inf count not in warning"
        
        print("  ✓ NaN/Inf values detected and logged")
        
        # Clean up
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
        return False


def test_3d_poses_rejection():
    """Test that 3D pose arrays are properly rejected."""
    print("\n=== Test: 3D Poses Rejection ===")
    
    input_file = "test_3d_poses.npy"
    output_file = "test_3d_poses_out.npy"
    
    try:
        # Create test data with 3D poses
        test_data = {
            'poses': np.random.randn(5, 10, 72),
            'trans': np.random.randn(10, 3)
        }
        save_npy(input_file, test_data, allow_overwrite=True)
        
        # Attempt conversion (should fail)
        success = convert_smpl_to_smplx(input_file, output_file)
        
        assert not success, "3D poses should fail conversion"
        print("  ✓ 3D poses properly rejected")
        
        # Clean up
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
        return False


def test_keys_logging():
    """Test that available keys are logged."""
    print("\n=== Test: Available Keys Logging ===")
    
    input_file = "test_keys_logging.npy"
    output_file = "test_keys_logging_out.npy"
    
    try:
        # Create test data
        test_data = {
            'poses': np.random.randn(10, 72),
            'trans': np.random.randn(10, 3),
            'betas': np.random.randn(10),
            'custom_key': np.array([1, 2, 3])
        }
        save_npy(input_file, test_data, allow_overwrite=True)
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        # Convert
        success = convert_smpl_to_smplx(input_file, output_file)
        
        # Restore stdout
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        
        # Check that keys were logged
        assert "Available keys in file:" in output, "Keys not logged"
        assert "poses" in output, "'poses' key not in log"
        assert "custom_key" in output, "'custom_key' not in log"
        
        print("  ✓ Available keys properly logged")
        
        # Clean up
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
        return False


def test_trans_validation():
    """Test that trans array is validated."""
    print("\n=== Test: Trans Array Validation ===")
    
    input_file = "test_trans_validation.npy"
    output_file = "test_trans_validation_out.npy"
    
    try:
        # Create test data with trans containing NaN
        test_data = {
            'poses': np.random.randn(10, 72),
            'trans': np.random.randn(10, 3),
            'betas': np.random.randn(10)
        }
        test_data['trans'][2, 1] = np.nan
        save_npy(input_file, test_data, allow_overwrite=True)
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        # Convert
        success = convert_smpl_to_smplx(input_file, output_file)
        
        # Restore stdout
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        
        # Check that trans was validated and logged
        assert "Trans shape:" in output, "Trans shape not logged"
        assert "WARNING: Trans contains NaN or Inf values" in output, "Trans NaN warning not logged"
        
        print("  ✓ Trans array properly validated")
        
        # Clean up
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
        return False


def main():
    """Run all debugging improvement tests."""
    print("="*60)
    print("Testing Debugging Improvements in smpl_to_smplx.py")
    print("="*60)
    
    results = []
    
    results.append(("Empty Poses Rejection", test_empty_poses_rejection()))
    results.append(("Malformed Betas Handling", test_malformed_betas_handling()))
    results.append(("NaN/Inf Detection", test_nan_inf_detection()))
    results.append(("3D Poses Rejection", test_3d_poses_rejection()))
    results.append(("Keys Logging", test_keys_logging()))
    results.append(("Trans Validation", test_trans_validation()))
    
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
