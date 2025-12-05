# Retarget-Test Repository

A refactored Python codebase for motion retargeting with robust NPY/NPZ file handling and SMPLX format visualization.

## Overview

This repository contains tools for:
- Reading and writing `.npy` and `.npz` files with error handling
- Converting between SMPL and SMPLX motion formats
- Visualizing SMPLX skeleton data
- Processing AMASS motion capture datasets

## Recent Improvements

### 1. Robust NPY/NPZ File Handling
- **New Module**: `npy_handler.py` - Comprehensive file handler with:
  - Error handling and validation
  - File information extraction
  - Data validation for motion data
  - Support for both `.npy` and `.npz` formats

### 2. SMPLX Visualization
- **New Module**: `smplx_visualizer.py` - Visualization utilities featuring:
  - 3D skeleton plotting
  - Motion sequence rendering to video/GIF
  - Data structure inspection
  - Support for matplotlib-based visualizations

### 3. Enhanced Existing Scripts
- **`read_np.py`**: Upgraded with CLI interface and detailed file inspection
- **`data_utils.py`**: Improved error handling and support for multiple formats
- **`smpl_to_smplx.py`**: Enhanced conversion with better error handling

### 4. Code Quality Improvements
- Added comprehensive error handling
- Improved modularity and reusability
- Added logging for better debugging
- Created documentation and examples

## Installation

### Basic Dependencies
```bash
pip install numpy torch matplotlib imageio imageio-ffmpeg
```

### Optional Dependencies (for full functionality)
```bash
pip install scipy pyyaml tqdm natsort rich torchgeometry
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

### FBX Library (Optional - for FBX file support)

If you encounter errors related to missing `FbxCommon` module, you need to install the Autodesk FBX Python SDK:

1. **Download the FBX Python SDK**:
   - Visit: https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-3-2
   - Download the appropriate version for your platform and Python version
   - Note: Make sure to match your Python version (e.g., Python 3.8, 3.9, 3.10)

2. **Install the SDK**:
   - Follow the installation instructions provided with the SDK
   - On Windows: Run the installer and follow the wizard
   - On Linux/Mac: Extract and run the installation script
   
3. **Verify Installation**:
   ```bash
   python -c "import FbxCommon; print('FBX SDK installed successfully')"
   ```

**Note**: FBX support is optional. If you don't need to work with FBX files, you can skip this installation. The core functionality of this repository (NPY/NPZ handling, SMPL/SMPLX conversion) works without FBX SDK.

## Usage

### Reading NPY/NPZ Files

#### Command Line
```bash
# Basic usage
python read_np.py ref_motion.npy

# Verbose output with data values
python read_np.py ref_motion.npy --verbose

# Read NPZ files
python read_np.py g1.npz
```

#### Python API
```python
from npy_handler import NpyNpzHandler

# Create handler
handler = NpyNpzHandler(allow_pickle=True)

# Load NPY file
data = handler.load_npy('ref_motion.npy')

# Load NPZ file
data = handler.load_npz('g1.npz')

# Get file information without loading all data
info = handler.get_info('ref_motion.npy')

# Validate motion data
is_valid = handler.validate_motion_data(data)

# Save data
handler.save_npy('output.npy', data, allow_overwrite=True)
handler.save_npz('output.npz', data_dict, compressed=True)
```

### SMPL to SMPLX Conversion

#### Command Line
```bash
# Convert single file
python smpl_to_smplx.py --input_file input.npy --output_file output.npy

# Convert directory
python smpl_to_smplx.py --src_folder smpl_data/ --tgt_folder smplx_data/

# Specify gender
python smpl_to_smplx.py --input_file input.npy --output_file output.npy --gender female
```

#### Python API
```python
from smpl_to_smplx import convert_smpl_to_smplx

# Convert file
success = convert_smpl_to_smplx(
    input_path='input.npy',
    output_path='output.npy',
    gender='neutral'
)
```

### SMPLX Visualization

```python
from smplx_visualizer import SMPLXVisualizer, get_smplx_info

# Create visualizer
visualizer = SMPLXVisualizer()

# Load and inspect data
data = visualizer.load_smplx_data('motion.npy')
info = visualizer.export_info(data)

# Get file information
info = get_smplx_info('motion.npy')

# Plot skeleton (requires joint positions)
# positions shape: (num_joints, 3)
visualizer.plot_skeleton_3d(positions, save_path='skeleton.png')

# Render motion sequence (requires joint positions over time)
# positions shape: (num_frames, num_joints, 3)
visualizer.render_motion_sequence(
    positions,
    output_path='motion.mp4',
    fps=30
)
```

### Running Examples

```bash
python examples.py
```

This will run all example scenarios demonstrating:
1. Reading NPY files
2. Reading NPZ files
3. SMPL to SMPLX conversion
4. SMPLX file information extraction
5. Error handling
6. Data validation

## Module Reference

### npy_handler.py

Main class: `NpyNpzHandler`

Methods:
- `load_npy(filepath)` - Load .npy file
- `load_npz(filepath)` - Load .npz file
- `save_npy(filepath, data)` - Save to .npy file
- `save_npz(filepath, data)` - Save to .npz file
- `validate_motion_data(data)` - Validate motion data
- `get_info(filepath)` - Get file information

Convenience functions:
- `load_npy(filepath, allow_pickle=True)`
- `load_npz(filepath, allow_pickle=True)`
- `save_npy(filepath, data, allow_overwrite=False)`
- `save_npz(filepath, data, compressed=False, allow_overwrite=False)`

### smplx_visualizer.py

Main class: `SMPLXVisualizer`

Methods:
- `load_smplx_data(filepath)` - Load SMPLX data
- `validate_smplx_data(data)` - Validate SMPLX format
- `plot_skeleton_3d(positions, save_path)` - Plot 3D skeleton
- `render_motion_sequence(positions, output_path, fps)` - Render motion video
- `export_info(data)` - Export data summary

Convenience functions:
- `visualize_smplx_file(filepath, output_path, title)`
- `get_smplx_info(filepath)`

### data_utils.py

Functions:
- `process_amass_seq(fname, output_path)` - Process AMASS sequence
- `project_joints_simple(motion)` - Project joints for retargeting

### smpl_to_smplx.py

Functions:
- `convert_smpl_to_smplx(input_path, output_path, gender)` - Convert SMPL to SMPLX
- `process_directory(src_folder, tgt_folder, gender)` - Batch convert directory

## File Formats

### SMPL Format
```python
{
    'poses': np.ndarray,  # Shape (N, 72) - 24 joints × 3 DOF
    'trans': np.ndarray,  # Shape (N, 3) - root translation
    'betas': np.ndarray,  # Shape (10,) - shape parameters
    'fps': int,           # Frame rate
}
```

### SMPLX Format
```python
{
    'root_orient': np.ndarray,  # Shape (N, 3) - root orientation
    'pose_body': np.ndarray,    # Shape (N, 63) - body pose
    'trans': np.ndarray,        # Shape (N, 3) - translation
    'betas': np.ndarray,        # Shape (16,) - shape parameters
    'gender': str,              # Gender ('male', 'female', 'neutral')
    'fps': int,                 # Frame rate
}
```

## Error Handling

All modules implement comprehensive error handling:
- `FileNotFoundError` - File doesn't exist
- `ValueError` - Invalid file format or data
- `TypeError` - Incorrect data type
- `FileExistsError` - Attempting to overwrite without permission

Enable verbose logging for debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Testing

The codebase includes validation at multiple levels:
1. File format validation
2. Data structure validation
3. Value range checking (NaN, empty arrays)
4. Key presence validation for dictionaries

Run examples to test functionality:
```bash
python examples.py
```

## Best Practices

1. **Always use error handling** when loading files:
   ```python
   try:
       data = handler.load_npy('file.npy')
   except FileNotFoundError:
       print("File not found")
   except Exception as e:
       print(f"Error: {e}")
   ```

2. **Validate data** before processing:
   ```python
   if handler.validate_motion_data(data):
       # Process data
       pass
   ```

3. **Use allow_overwrite** cautiously:
   ```python
   # Safe - won't overwrite existing files
   handler.save_npy('output.npy', data, allow_overwrite=False)
   
   # Careful - will overwrite
   handler.save_npy('output.npy', data, allow_overwrite=True)
   ```

4. **Check file info** before loading large files:
   ```python
   info = handler.get_info('large_file.npy')
   print(f"File size: {info['size_bytes'] / 1024 / 1024:.2f} MB")
   ```

## Contributing

When adding new features:
1. Include error handling
2. Add logging statements
3. Update documentation
4. Add examples if applicable
5. Follow existing code style

## License

See repository for license information.

## Acknowledgments

This repository builds upon motion retargeting research and includes utilities for working with:
- AMASS dataset
- SMPL/SMPLX body models
- Various motion capture formats
