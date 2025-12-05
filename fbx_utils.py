"""
FBX Utilities Module

This module provides helper functions for working with FBX files and
includes improved error handling for missing FBX SDK.
"""

import sys

def check_fbx_available():
    """
    Check if FBX SDK is available and provide helpful error messages if not.
    
    Returns:
        tuple: (is_available: bool, error_message: str or None)
    """
    try:
        import FbxCommon
        return True, None
    except ImportError as e:
        error_msg = """
FBX SDK is not installed or not found in Python path.

To use FBX functionality, you need to install the Autodesk FBX Python SDK:

1. Download the FBX Python SDK:
   Visit: https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-3-2
   
2. Select the appropriate version:
   - Match your Python version (e.g., Python 3.8, 3.9, 3.10)
   - Match your operating system (Windows, Linux, macOS)
   
3. Install the SDK:
   - Windows: Run the installer and follow the wizard
   - Linux/Mac: Extract and run the installation script
   
4. Verify installation:
   python -c "import FbxCommon; print('FBX SDK installed successfully')"

Alternative:
If you don't need FBX support, you can work with NPY/NPZ files directly.
Most functionality in this repository works without FBX SDK.

Original error: {error}
""".format(error=str(e))
        return False, error_msg


def import_fbx_with_fallback(raise_on_error=True):
    """
    Import FBX SDK with helpful error messages.
    
    Args:
        raise_on_error: If True, raises ImportError with helpful message.
                       If False, returns None and prints warning.
    
    Returns:
        FbxCommon module or None
        
    Raises:
        ImportError: If FBX is not available and raise_on_error is True
    """
    is_available, error_msg = check_fbx_available()
    
    if is_available:
        import FbxCommon
        return FbxCommon
    else:
        if raise_on_error:
            raise ImportError(error_msg)
        else:
            print(f"Warning: FBX SDK not available. FBX functionality will be disabled.")
            print(f"See README.md for installation instructions.")
            return None


def load_fbx_file(filepath, raise_on_error=True):
    """
    Load an FBX file with proper error handling.
    
    Args:
        filepath: Path to the FBX file
        raise_on_error: If True, raises errors. If False, returns None on error.
    
    Returns:
        FBX scene object or None
        
    Raises:
        ImportError: If FBX SDK is not available
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be loaded
    """
    import os
    
    # Check if FBX is available
    FbxCommon = import_fbx_with_fallback(raise_on_error=raise_on_error)
    if FbxCommon is None:
        return None
    
    # Check if file exists
    if not os.path.exists(filepath):
        error_msg = f"FBX file not found: {filepath}"
        if raise_on_error:
            raise FileNotFoundError(error_msg)
        else:
            print(f"Error: {error_msg}")
            return None
    
    try:
        # Import FBX manager and scene
        manager, scene = FbxCommon.InitializeSdkObjects()
        
        # Load the file
        result = FbxCommon.LoadScene(manager, scene, filepath)
        
        if not result:
            error_msg = f"Failed to load FBX file: {filepath}"
            if raise_on_error:
                raise ValueError(error_msg)
            else:
                print(f"Error: {error_msg}")
                return None
        
        return scene
        
    except Exception as e:
        error_msg = f"Error loading FBX file {filepath}: {e}"
        if raise_on_error:
            raise ValueError(error_msg)
        else:
            print(error_msg)
            return None


# Example usage and testing
if __name__ == "__main__":
    print("Testing FBX SDK availability...")
    is_available, error_msg = check_fbx_available()
    
    if is_available:
        print("✓ FBX SDK is installed and available")
    else:
        print("✗ FBX SDK is not available")
        print(error_msg)
