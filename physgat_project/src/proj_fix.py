#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJ database warning fix module for handling geospatial library compatibility issues.
Configures environment variables and suppresses PROJ version mismatch warnings.

PROJ数据库警告修复模块，用于处理地理空间库兼容性问题。
配置环境变量并抑制PROJ版本不匹配警告。

Author: Wenhao Wang
"""

import os
import sys
import warnings
import logging
from pathlib import Path

def setup_proj_environment():
    """
    Set up the PROJ environment to use the correct data directory and suppress warnings.
    
    This function should be called before importing rasterio or any geospatial libraries.
    """
    try:
        # Try to import pyproj to get the correct PROJ data directory
        import pyproj
        
        # Get the correct PROJ data directory from pyproj
        correct_proj_dir = pyproj.datadir.get_data_dir()
        
        # Set environment variables to use the correct PROJ installation
        os.environ['PROJ_LIB'] = correct_proj_dir
        
        # Also set GDAL_DATA if it's not set
        if 'GDAL_DATA' not in os.environ:
            # Try to find GDAL data directory
            try:
                import rasterio
                gdal_data_path = os.path.join(os.path.dirname(rasterio.__file__), 'gdal_data')
                if os.path.exists(gdal_data_path):
                    os.environ['GDAL_DATA'] = gdal_data_path
            except ImportError:
                pass
        
        logging.info(f"PROJ environment configured: PROJ_LIB={correct_proj_dir}")
        return True
        
    except ImportError as e:
        logging.warning(f"Could not configure PROJ environment: {e}")
        return False

def suppress_proj_warnings():
    """
    Suppress PROJ-related warnings that don't affect functionality.
    
    This is a fallback solution when environment configuration doesn't work.
    """
    # Suppress specific PROJ warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')
    warnings.filterwarnings('ignore', message='.*PROJ.*DATABASE.LAYOUT.VERSION.*')
    
    # Also suppress at the logging level for rasterio
    logging.getLogger('rasterio._env').setLevel(logging.ERROR)
    logging.getLogger('rasterio.env').setLevel(logging.ERROR)
    
    logging.info("PROJ warnings suppressed")

def verify_proj_functionality():
    """
    Verify that PROJ and rasterio functionality works correctly after configuration.
    
    Returns:
        bool: True if functionality is working, False otherwise
    """
    try:
        import rasterio
        import pyproj
        
        # Test basic PROJ functionality
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = transformer.transform(103.0, 29.0)
        
        # Test that we can create a basic CRS
        crs = pyproj.CRS.from_epsg(4326)
        
        logging.info("PROJ functionality verification passed")
        return True
        
    except Exception as e:
        logging.error(f"PROJ functionality verification failed: {e}")
        return False

def apply_proj_fix():
    """
    Apply the complete PROJ fix solution.
    
    This function should be called at the beginning of the main script
    before any geospatial operations.
    
    Returns:
        bool: True if fix was applied successfully
    """
    logging.info("Applying PROJ database fix...")
    
    # Step 1: Try to configure environment
    env_configured = setup_proj_environment()
    
    # Step 2: Suppress warnings as fallback
    suppress_proj_warnings()
    
    # Step 3: Verify functionality
    functionality_ok = verify_proj_functionality()
    
    if functionality_ok:
        logging.info("PROJ fix applied successfully - geospatial functionality verified")
        return True
    else:
        logging.warning("PROJ fix applied but functionality verification failed")
        return False

# Context manager for temporary PROJ configuration
class ProjEnvironment:
    """
    Context manager for temporarily setting PROJ environment variables.
    
    Usage:
        with ProjEnvironment():
            # Your geospatial operations here
            import rasterio
            # ... do rasterio operations
    """
    
    def __init__(self):
        self.original_env = {}
        
    def __enter__(self):
        # Save original environment
        self.original_env = {
            'PROJ_LIB': os.environ.get('PROJ_LIB'),
            'GDAL_DATA': os.environ.get('GDAL_DATA')
        }
        
        # Apply PROJ fix
        apply_proj_fix()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original environment
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

# Decorator for functions that use geospatial operations
def with_proj_fix(func):
    """
    Decorator to apply PROJ fix before executing a function.
    
    Usage:
        @with_proj_fix
        def my_geospatial_function():
            import rasterio
            # ... geospatial operations
    """
    def wrapper(*args, **kwargs):
        with ProjEnvironment():
            return func(*args, **kwargs)
    return wrapper

if __name__ == "__main__":
    # Test the PROJ fix
    print("Testing PROJ fix...")
    success = apply_proj_fix()
    if success:
        print("PROJ fix test passed!")
    else:
        print("PROJ fix test failed!")
