"""
NPY and NPZ File Handler Module

This module provides robust utilities for reading, writing, and validating
.npy and .npz files with proper error handling.
"""

import numpy as np
import os
from pathlib import Path
from typing import Union, Dict, Any, Optional, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NpyNpzHandler:
    """
    A comprehensive handler for .npy and .npz files with error handling,
    validation, and utility methods.
    """
    
    def __init__(self, allow_pickle: bool = True):
        """
        Initialize the handler.
        
        Args:
            allow_pickle: Whether to allow loading pickled objects (default: True)
        """
        self.allow_pickle = allow_pickle
    
    def load_npy(self, filepath: Union[str, Path]) -> np.ndarray:
        """
        Load a .npy file with error handling.
        
        Args:
            filepath: Path to the .npy file
            
        Returns:
            Loaded numpy array
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid .npy file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.suffix.lower() != '.npy':
            raise ValueError(f"Expected .npy file, got: {filepath.suffix}")
        
        try:
            data = np.load(filepath, allow_pickle=self.allow_pickle)
            logger.info(f"Successfully loaded .npy file: {filepath}")
            logger.info(f"  Shape: {data.shape}, Dtype: {data.dtype}")
            return data
        except Exception as e:
            logger.error(f"Error loading .npy file {filepath}: {e}")
            raise
    
    def load_npz(self, filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Load a .npz file with error handling.
        
        Args:
            filepath: Path to the .npz file
            
        Returns:
            Dictionary of arrays from the .npz file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid .npz file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.suffix.lower() != '.npz':
            raise ValueError(f"Expected .npz file, got: {filepath.suffix}")
        
        try:
            # Load npz file
            data = np.load(filepath, allow_pickle=self.allow_pickle)
            
            # Check if it's already a dict (pickled dict) or NpzFile
            if isinstance(data, dict):
                result = data
            else:
                # It's an NpzFile, convert to dict
                result = {key: data[key] for key in data.files}
                # Close the NpzFile
                data.close()
            
            logger.info(f"Successfully loaded .npz file: {filepath}")
            logger.info(f"  Keys: {list(result.keys())}")
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    logger.info(f"    {key}: shape={value.shape}, dtype={value.dtype}")
            return result
        except Exception as e:
            logger.error(f"Error loading .npz file {filepath}: {e}")
            raise
    
    def save_npy(self, filepath: Union[str, Path], data: Union[np.ndarray, Dict[str, Any]], 
                 allow_overwrite: bool = False) -> None:
        """
        Save data to a .npy file.
        
        Args:
            filepath: Path where to save the file
            data: Numpy array or dict to save (dict will be converted to object array)
            allow_overwrite: Whether to allow overwriting existing files
            
        Raises:
            FileExistsError: If file exists and overwrite is not allowed
        """
        filepath = Path(filepath)
        
        # Convert dict to numpy object array if needed
        if isinstance(data, dict):
            data = np.array(data, dtype=object)
        
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array or dict, got {type(data)}")
        
        if filepath.exists() and not allow_overwrite:
            raise FileExistsError(
                f"File already exists: {filepath}. "
                "Set allow_overwrite=True to overwrite."
            )
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            np.save(filepath, data)
            logger.info(f"Successfully saved .npy file: {filepath}")
            if hasattr(data, 'shape'):
                logger.info(f"  Shape: {data.shape}, Dtype: {data.dtype}")
        except Exception as e:
            logger.error(f"Error saving .npy file {filepath}: {e}")
            raise
    
    def save_npz(self, filepath: Union[str, Path], data: Dict[str, np.ndarray],
                 compressed: bool = False, allow_overwrite: bool = False) -> None:
        """
        Save data to a .npz file.
        
        Args:
            filepath: Path where to save the file
            data: Dictionary of numpy arrays to save
            compressed: Whether to use compression
            allow_overwrite: Whether to allow overwriting existing files
            
        Raises:
            FileExistsError: If file exists and overwrite is not allowed
            TypeError: If data is not a dictionary of arrays
        """
        filepath = Path(filepath)
        
        if not isinstance(data, dict):
            raise TypeError(f"Expected dictionary, got {type(data)}")
        
        if filepath.exists() and not allow_overwrite:
            raise FileExistsError(
                f"File already exists: {filepath}. "
                "Set allow_overwrite=True to overwrite."
            )
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if compressed:
                np.savez_compressed(filepath, **data)
            else:
                np.savez(filepath, **data)
            logger.info(f"Successfully saved .npz file: {filepath}")
            logger.info(f"  Keys: {list(data.keys())}")
        except Exception as e:
            logger.error(f"Error saving .npz file {filepath}: {e}")
            raise
    
    def validate_motion_data(self, data: Union[np.ndarray, Dict[str, Any]], 
                            expected_keys: Optional[List[str]] = None) -> bool:
        """
        Validate motion data structure.
        
        Args:
            data: Motion data to validate (array or dict)
            expected_keys: List of expected keys if data is a dict
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if isinstance(data, np.ndarray):
                # Handle object dtype arrays (may contain dicts or complex structures)
                if data.dtype == object:
                    logger.info("Object array detected - skipping NaN check")
                    if data.size == 0:
                        logger.warning("Empty array")
                        return False
                    return True
                
                # Basic validation for numeric arrays
                if data.size == 0:
                    logger.warning("Empty array")
                    return False
                
                # Only check for NaN in numeric types
                if np.issubdtype(data.dtype, np.floating):
                    if np.isnan(data).any():
                        logger.warning("Array contains NaN values")
                        return False
                return True
            
            elif isinstance(data, dict):
                # Validate dictionary structure
                if expected_keys:
                    missing_keys = set(expected_keys) - set(data.keys())
                    if missing_keys:
                        logger.warning(f"Missing keys: {missing_keys}")
                        return False
                
                # Check each value in the dictionary
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        # Handle object dtype
                        if value.dtype == object:
                            continue
                        
                        if value.size == 0:
                            logger.warning(f"Empty array for key: {key}")
                            return False
                        
                        # Only check NaN for floating point types
                        if np.issubdtype(value.dtype, np.floating):
                            if np.isnan(value).any():
                                logger.warning(f"Array contains NaN for key: {key}")
                                return False
                
                return True
            
            else:
                logger.warning(f"Unexpected data type: {type(data)}")
                return False
                
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def get_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a .npy or .npz file without loading all data.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Dictionary with file information
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        info = {
            'filepath': str(filepath),
            'size_bytes': filepath.stat().st_size,
            'extension': filepath.suffix
        }
        
        try:
            if filepath.suffix.lower() == '.npy':
                # Load just to get shape and dtype
                data = np.load(filepath, allow_pickle=self.allow_pickle)
                info['shape'] = data.shape
                info['dtype'] = str(data.dtype)
                info['num_elements'] = data.size
                
            elif filepath.suffix.lower() == '.npz':
                data = np.load(filepath, allow_pickle=self.allow_pickle)
                
                if isinstance(data, dict):
                    info['keys'] = list(data.keys())
                    info['arrays'] = {}
                    for key, arr in data.items():
                        info['arrays'][key] = {
                            'shape': arr.shape if isinstance(arr, np.ndarray) else None,
                            'dtype': str(arr.dtype) if isinstance(arr, np.ndarray) else str(type(arr))
                        }
                else:
                    info['keys'] = list(data.files)
                    info['arrays'] = {}
                    for key in data.files:
                        arr = data[key]
                        info['arrays'][key] = {
                            'shape': arr.shape if isinstance(arr, np.ndarray) else None,
                            'dtype': str(arr.dtype) if isinstance(arr, np.ndarray) else str(type(arr))
                        }
                    data.close()
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            info['error'] = str(e)
            return info


# Convenience functions for backward compatibility
def load_npy(filepath: Union[str, Path], allow_pickle: bool = True) -> np.ndarray:
    """Convenience function to load .npy file."""
    handler = NpyNpzHandler(allow_pickle=allow_pickle)
    return handler.load_npy(filepath)


def load_npz(filepath: Union[str, Path], allow_pickle: bool = True) -> Dict[str, np.ndarray]:
    """Convenience function to load .npz file."""
    handler = NpyNpzHandler(allow_pickle=allow_pickle)
    return handler.load_npz(filepath)


def save_npy(filepath: Union[str, Path], data: Union[np.ndarray, Dict[str, Any]], 
             allow_overwrite: bool = False) -> None:
    """Convenience function to save .npy file."""
    handler = NpyNpzHandler()
    handler.save_npy(filepath, data, allow_overwrite=allow_overwrite)


def save_npz(filepath: Union[str, Path], data: Dict[str, np.ndarray],
             compressed: bool = False, allow_overwrite: bool = False) -> None:
    """Convenience function to save .npz file."""
    handler = NpyNpzHandler()
    handler.save_npz(filepath, data, compressed=compressed, allow_overwrite=allow_overwrite)
