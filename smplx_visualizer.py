"""
SMPLX Visualization Module

This module provides visualization utilities for SMPLX format data including
skeleton rendering, motion visualization, and export capabilities.
"""

import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SMPLXVisualizer:
    """
    Visualizer for SMPLX format data with support for static poses
    and animated motion sequences.
    """
    
    def __init__(self):
        """Initialize the SMPLX visualizer."""
        # SMPLX has 55 joints total (22 body + 30 hand + 3 head joints)
        # This list contains only the 22 body joints for basic skeleton visualization
        # Hand and head joints are excluded for simplicity
        self.body_joint_names = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
        ]
        
        # Parent indices for kinematic chain (SMPLX body skeleton)
        self.parent_indices = [
            -1,  # pelvis (root)
            0, 0, 0,  # left_hip, right_hip, spine1
            1, 2, 3,  # left_knee, right_knee, spine2
            4, 5, 6,  # left_ankle, right_ankle, spine3
            7, 8, 9,  # left_foot, right_foot, neck
            9, 9, 12,  # left_collar, right_collar, head
            13, 14,  # left_shoulder, right_shoulder
            16, 17,  # left_elbow, right_elbow
            18, 19   # left_wrist, right_wrist
        ]
    
    def load_smplx_data(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load SMPLX data from file.
        
        Args:
            filepath: Path to SMPLX data file (.npy or .npz)
            
        Returns:
            Dictionary containing SMPLX parameters
        """
        from npy_handler import NpyNpzHandler
        
        filepath = Path(filepath)
        handler = NpyNpzHandler()
        
        if filepath.suffix.lower() == '.npy':
            data = handler.load_npy(filepath)
            # If it's a dict inside npy
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = data.item()
        elif filepath.suffix.lower() == '.npz':
            data = handler.load_npz(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Loaded SMPLX data from {filepath}")
        return data
    
    def validate_smplx_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate SMPLX data structure.
        
        Args:
            data: SMPLX data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        # Check for essential SMPLX keys
        if 'root_orient' in data or 'pose_body' in data:
            logger.info("Valid SMPLX format detected")
            return True
        elif 'poses' in data:
            logger.info("SMPL format detected (can be converted to SMPLX)")
            return True
        else:
            logger.warning("Unknown format - missing required keys")
            return False
    
    def create_skeleton_edges(self) -> List[Tuple[int, int]]:
        """
        Create list of edges connecting joints based on parent indices.
        
        Returns:
            List of tuples (child, parent) for each edge
        """
        edges = []
        for i, parent in enumerate(self.parent_indices):
            if parent >= 0:
                edges.append((i, parent))
        return edges
    
    def plot_skeleton_3d(self, positions: np.ndarray, 
                        save_path: Optional[Union[str, Path]] = None,
                        title: str = "SMPLX Skeleton",
                        show_axes: bool = True) -> None:
        """
        Plot a 3D skeleton from joint positions.
        
        Args:
            positions: Joint positions, shape (num_joints, 3)
            save_path: Optional path to save the figure
            title: Title for the plot
            show_axes: Whether to show axis labels
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            logger.error("matplotlib is required for visualization. Install with: pip install matplotlib")
            return
        
        if positions.shape[0] < len(self.parent_indices):
            logger.warning(f"Expected at least {len(self.parent_indices)} joints, got {positions.shape[0]}")
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot joints
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='red', s=50, alpha=0.8, label='Joints')
        
        # Plot bones
        edges = self.create_skeleton_edges()
        for child, parent in edges:
            if child < len(positions) and parent < len(positions):
                xs = [positions[child, 0], positions[parent, 0]]
                ys = [positions[child, 1], positions[parent, 1]]
                zs = [positions[child, 2], positions[parent, 2]]
                ax.plot(xs, ys, zs, 'b-', linewidth=2, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        if not show_axes:
            ax.set_axis_off()
        
        # Set equal aspect ratio
        max_range = np.array([
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved skeleton plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def render_motion_sequence(self, positions: np.ndarray,
                              output_path: Union[str, Path],
                              fps: int = 30,
                              frame_skip: int = 1) -> None:
        """
        Render a motion sequence to video or GIF.
        
        Args:
            positions: Joint positions for each frame, shape (num_frames, num_joints, 3)
            output_path: Path to save the output video/gif
            fps: Frames per second for output
            frame_skip: Render every Nth frame to speed up processing
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import imageio
        except ImportError:
            logger.error("matplotlib and imageio are required. Install with: pip install matplotlib imageio")
            return
        
        output_path = Path(output_path)
        num_frames, num_joints, _ = positions.shape
        
        logger.info(f"Rendering {num_frames} frames to {output_path}")
        
        # Calculate bounds for all frames
        all_positions = positions.reshape(-1, 3)
        mins = all_positions.min(axis=0)
        maxs = all_positions.max(axis=0)
        max_range = (maxs - mins).max() / 2.0
        mid = (maxs + mins) / 2.0
        
        frames = []
        edges = self.create_skeleton_edges()
        
        for frame_idx in range(0, num_frames, frame_skip):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            pos = positions[frame_idx]
            
            # Plot joints
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], 
                      c='red', s=50, alpha=0.8)
            
            # Plot bones
            for child, parent in edges:
                if child < len(pos) and parent < len(pos):
                    xs = [pos[child, 0], pos[parent, 0]]
                    ys = [pos[child, 1], pos[parent, 1]]
                    zs = [pos[child, 2], pos[parent, 2]]
                    ax.plot(xs, ys, zs, 'b-', linewidth=2, alpha=0.6)
            
            # Set fixed view
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Frame {frame_idx + 1}/{num_frames}')
            
            # Convert plot to image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            
            plt.close(fig)
            
            if (frame_idx + 1) % 10 == 0:
                logger.info(f"Rendered {frame_idx + 1}/{num_frames} frames")
        
        # Save video/gif
        if output_path.suffix.lower() == '.gif':
            imageio.mimsave(output_path, frames, fps=fps, loop=0)
        else:
            # Default to mp4
            imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
        
        logger.info(f"Saved motion visualization to {output_path}")
    
    def export_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export summary information about SMPLX data.
        
        Args:
            data: SMPLX data dictionary
            
        Returns:
            Dictionary with summary information
        """
        info = {}
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Only compute stats for numeric types (check dtype first for performance)
                if np.issubdtype(value.dtype, np.number) and value.size > 0:
                    info[key] = {
                        'shape': value.shape,
                        'dtype': str(value.dtype),
                        'min': float(value.min()),
                        'max': float(value.max()),
                        'mean': float(value.mean())
                    }
                else:
                    info[key] = {
                        'shape': value.shape,
                        'dtype': str(value.dtype),
                        'size': value.size
                    }
            else:
                info[key] = {
                    'type': str(type(value)),
                    'value': str(value) if not isinstance(value, (dict, list)) else f"{type(value).__name__} (truncated)"
                }
        
        return info


# Convenience functions
def visualize_smplx_file(filepath: Union[str, Path], 
                        output_path: Optional[Union[str, Path]] = None,
                        title: str = "SMPLX Visualization") -> None:
    """
    Convenience function to visualize a single SMPLX file.
    
    Args:
        filepath: Path to SMPLX file
        output_path: Optional path to save visualization
        title: Title for the plot
    """
    visualizer = SMPLXVisualizer()
    data = visualizer.load_smplx_data(filepath)
    
    # Try to extract positions from common formats
    if 'poses' in data and 'trans' in data:
        # This is simplified - full SMPLX would need body model
        logger.info("Note: Full SMPLX visualization requires body model")
        logger.info("Showing data summary instead")
        info = visualizer.export_info(data)
        for key, value in info.items():
            print(f"{key}: {value}")
    else:
        logger.info("Data structure:")
        info = visualizer.export_info(data)
        for key, value in info.items():
            print(f"{key}: {value}")


def get_smplx_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about SMPLX file.
    
    Args:
        filepath: Path to SMPLX file
        
    Returns:
        Dictionary with file information
    """
    visualizer = SMPLXVisualizer()
    data = visualizer.load_smplx_data(filepath)
    return visualizer.export_info(data)
