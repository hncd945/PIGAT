#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data utilities module for loading, preprocessing, and constructing heterogeneous graphs.
Handles CSV data loading, DEM terrain features, PMF factorization, and physics-constrained edge building.

数据工具模块，用于加载、预处理和构建异构图。
处理CSV数据加载、DEM地形特征提取、PMF因子分解和物理约束边构建。

Author: Wenhao Wang
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import cosine_similarity
import logging

try:
    from .proj_fix import apply_proj_fix
    apply_proj_fix()
except ImportError:
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')
    warnings.filterwarnings('ignore', message='.*PROJ.*DATABASE.LAYOUT.VERSION.*')
    logging.getLogger('rasterio._env').setLevel(logging.ERROR)

import rasterio
from rasterio.sample import sample_gen

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    encodings = [kwargs.pop('encoding', 'utf-8'), 'utf-8-sig', 'gb18030', 'gbk']
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return pd.read_csv(path, **kwargs)

def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (data < lower_bound) | (data > upper_bound)

def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold

def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    df_processed = df.copy()

    if strategy == 'drop':
        df_processed = df_processed.dropna()
    elif strategy == 'median':
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
    elif strategy == 'mean':
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
    elif strategy == 'mode':
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown')

    return df_processed

def validate_data_quality(df: pd.DataFrame, metals: list) -> dict:

    report = {
        'total_samples': len(df),
        'missing_values': {},
        'outliers': {},
        'negative_values': {},
        'zero_values': {},
        'data_range': {}
    }

    for metal in metals:
        if metal in df.columns:
            col_data = df[metal]

            missing_count = col_data.isnull().sum()
            report['missing_values'][metal] = {
                'count': missing_count,
                'percentage': (missing_count / len(df)) * 100
            }

            if not col_data.isnull().all():
                outliers_iqr = detect_outliers_iqr(col_data.dropna())
                outliers_zscore = detect_outliers_zscore(col_data.dropna())
                report['outliers'][metal] = {
                    'iqr_method': outliers_iqr.sum(),
                    'zscore_method': outliers_zscore.sum()
                }

                report['negative_values'][metal] = (col_data < 0).sum()
                report['zero_values'][metal] = (col_data == 0).sum()

                report['data_range'][metal] = {
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'mean': col_data.mean(),
                    'std': col_data.std()
                }

    return report

def clean_and_validate_data(df: pd.DataFrame, metals: list,
                          outlier_method: str = 'iqr',
                          outlier_action: str = 'cap',
                          missing_strategy: str = 'median') -> tuple:

    logging.info("=" * 50)
    logging.info("DATA CLEANING AND VALIDATION")
    logging.info("=" * 50)

    original_report = validate_data_quality(df, metals)
    logging.info(f"Original data samples: {original_report['total_samples']}")

    df_cleaned = handle_missing_values(df, missing_strategy)
    logging.info(f"Samples after missing value handling: {len(df_cleaned)}")

    for metal in metals:
        if metal in df_cleaned.columns:
            col_data = df_cleaned[metal]

            if outlier_method == 'iqr':
                outliers = detect_outliers_iqr(col_data)
            else:
                outliers = detect_outliers_zscore(col_data)

            if outliers.any():
                outlier_count = outliers.sum()
                logging.info(f"{metal}: Detected {outlier_count} outliers")

                if outlier_action == 'cap':
                    # Use quantile capping for outliers
                    Q1 = col_data.quantile(0.05)
                    Q99 = col_data.quantile(0.95)
                    df_cleaned[metal] = col_data.clip(lower=Q1, upper=Q99)
                elif outlier_action == 'remove':
                    # Remove outlier rows
                    df_cleaned = df_cleaned[~outliers]

    final_report = validate_data_quality(df_cleaned, metals)
    logging.info(f"Final data sample count: {final_report['total_samples']}")

    quality_report = {
        'original': original_report,
        'final': final_report,
        'cleaning_summary': {
            'samples_removed': original_report['total_samples'] - final_report['total_samples'],
            'outlier_method': outlier_method,
            'outlier_action': outlier_action,
            'missing_strategy': missing_strategy
        }
    }

    logging.info("Data cleaning and validation completed")
    return df_cleaned, quality_report

def _add_elevation_features(df: pd.DataFrame, dem_path: str) -> pd.DataFrame:
    """Extracts elevation for each coordinate in the dataframe from a DEM file."""
    if not dem_path or not os.path.exists(dem_path):
        logging.warning(f"DEM file not found at '{dem_path}' or path not provided. Skipping elevation feature extraction.")
        df['elevation'] = 0  # Add a default value
        df['slope'] = 0
        df['aspect'] = 0
        df['terrain_roughness'] = 0
        return df

    logging.info(f"Extracting elevation data from {dem_path}...")
    coords = [(x, y) for x, y in zip(df['lon'], df['lat'])]

    try:
        with rasterio.open(dem_path) as src:
            # sample_gen returns a generator. We need to iterate to get the values.
            elevation_generator = sample_gen(src, coords)
            elevations = [item[0] for item in elevation_generator]

        df['elevation'] = elevations

        # Calculate additional terrain features
        df = _calculate_terrain_features(df, dem_path)

        logging.info(f"Elevation data extracted and added as a feature. Min: {np.min(elevations)}, Max: {np.max(elevations)}")
    except Exception as e:
        logging.error(f"Failed to process DEM file {dem_path}: {e}")
        df['elevation'] = 0  # Add a default value on error
        df['slope'] = 0
        df['aspect'] = 0
        df['terrain_roughness'] = 0
    return df

def _calculate_terrain_features(df: pd.DataFrame, dem_path: str) -> pd.DataFrame:
    """Calculate advanced terrain features including slope, aspect, and roughness (vectorized)."""
    try:
        from rasterio.transform import rowcol
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            transform = src.transform

            # Slope and aspect
            dy, dx = np.gradient(dem_data)
            slope = np.degrees(np.arctan(np.hypot(dx, dy)))
            aspect = (np.degrees(np.arctan2(-dx, dy)) + 360) % 360

            # Terrain roughness via local std
            from scipy.ndimage import uniform_filter
            f = dem_data.astype(float)
            kernel_size = 3
            local_mean = uniform_filter(f, size=kernel_size)
            local_var = uniform_filter(f**2, size=kernel_size) - local_mean**2
            roughness = np.sqrt(np.maximum(local_var, 0))

            # Vectorized sampling using row/col indices
            xs = df['lon'].to_numpy()
            ys = df['lat'].to_numpy()
            cols, rows = rowcol(transform, xs, ys, op=np.floor)
            rows = rows.astype(int); cols = cols.astype(int)
            h, w = slope.shape
            valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

            out_slope = np.zeros(len(df), dtype=float)
            out_aspect = np.zeros(len(df), dtype=float)
            out_rough = np.zeros(len(df), dtype=float)
            out_slope[valid] = slope[rows[valid], cols[valid]]
            out_aspect[valid] = aspect[rows[valid], cols[valid]]
            out_rough[valid] = roughness[rows[valid], cols[valid]]

            df['slope'] = out_slope
            df['aspect'] = out_aspect
            df['terrain_roughness'] = out_rough

    except Exception as e:
        logging.warning(f"Failed to calculate terrain features: {e}")
        df['slope'] = 0
        df['aspect'] = 0
        df['terrain_roughness'] = 0

    return df

def _calculate_cumulative_elevation_gain(source_coords, receptor_coords, dem_path: str) -> np.ndarray:
    """Calculate cumulative elevation gain from sources to receptors."""
    if not dem_path or not os.path.exists(dem_path):
        return np.zeros((len(source_coords), len(receptor_coords)))

    try:
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            transform = src.transform

            cumulative_gains = np.zeros((len(source_coords), len(receptor_coords)))

            for i, source_coord in enumerate(source_coords):
                for j, receptor_coord in enumerate(receptor_coords):
                    # Simple path elevation analysis (can be enhanced with actual path finding)
                    source_col, source_row = ~transform * (source_coord[0], source_coord[1])
                    receptor_col, receptor_row = ~transform * (receptor_coord[0], receptor_coord[1])

                    source_col, source_row = int(source_col), int(source_row)
                    receptor_col, receptor_row = int(receptor_col), int(receptor_row)

                    if (0 <= source_row < dem_data.shape[0] and 0 <= source_col < dem_data.shape[1] and
                        0 <= receptor_row < dem_data.shape[0] and 0 <= receptor_col < dem_data.shape[1]):

                        source_elev = dem_data[source_row, source_col]
                        receptor_elev = dem_data[receptor_row, receptor_col]

                        # Simple elevation difference (positive if uphill to receptor)
                        elev_gain = max(0, receptor_elev - source_elev)
                        cumulative_gains[i, j] = elev_gain

            return cumulative_gains

    except Exception as e:
        logging.warning(f"Failed to calculate cumulative elevation gain: {e}")
        return np.zeros((len(source_coords), len(receptor_coords)))

def _identify_hydrological_features(df: pd.DataFrame, dem_path: str) -> pd.DataFrame:
    """Identify hydrological features and flow directions (vectorized)."""
    if not dem_path or not os.path.exists(dem_path):
        df['flow_direction'] = 0
        df['flow_accumulation'] = 0
        df['distance_to_stream'] = 0
        return df

    try:
        from rasterio.transform import rowcol
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            transform = src.transform

            dy, dx = np.gradient(dem_data.astype(float))
            flow_direction = (np.degrees(np.arctan2(-dy, -dx)) + 360) % 360
            flow_accumulation = np.ones_like(dem_data, dtype=float)

            xs = df['lon'].to_numpy(); ys = df['lat'].to_numpy()
            cols, rows = rowcol(transform, xs, ys, op=np.floor)
            rows = rows.astype(int); cols = cols.astype(int)
            h, w = flow_direction.shape
            valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

            out_fd = np.zeros(len(df), dtype=float)
            out_fa = np.ones(len(df), dtype=float)
            out_dist = np.zeros(len(df), dtype=float)  # Placeholder for future stream network distance
            out_fd[valid] = flow_direction[rows[valid], cols[valid]]
            out_fa[valid] = flow_accumulation[rows[valid], cols[valid]]

            df['flow_direction'] = out_fd
            df['flow_accumulation'] = out_fa
            df['distance_to_stream'] = out_dist

    except Exception as e:
        logging.warning(f"Failed to calculate hydrological features: {e}")
        df['flow_direction'] = 0
        df['flow_accumulation'] = 0
        df['distance_to_stream'] = 0

    return df

def _calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points in kilometers using Haversine formula."""
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

def _find_highest_point_between(receptor_coord, source_coord, dem_path, num_samples=50):
    """
    Find the highest elevation point along the line between receptor and source.

    This function samples the DEM raster along the line connecting the receptor and source,
    and returns the coordinates of the point with the highest elevation (the apex).

    Args:
        receptor_coord: Receptor coordinates [lon, lat, elevation]
        source_coord: Source coordinates [lon, lat, elevation]
        dem_path: Path to DEM raster file
        num_samples: Number of sample points along the line (default: 50)

    Returns:
        Tuple of (apex_lon, apex_lat, apex_elevation) representing the highest point
    """
    if not dem_path or not os.path.exists(dem_path):
        # If DEM not available, return midpoint
        mid_lon = (receptor_coord[0] + source_coord[0]) / 2
        mid_lat = (receptor_coord[1] + source_coord[1]) / 2
        mid_elev = (receptor_coord[2] + source_coord[2]) / 2
        return (mid_lon, mid_lat, mid_elev)

    try:
        # Generate sample points along the line from source to receptor
        lons = np.linspace(source_coord[0], receptor_coord[0], num_samples)
        lats = np.linspace(source_coord[1], receptor_coord[1], num_samples)
        coords = list(zip(lons, lats))

        # Query DEM for elevations at sample points
        with rasterio.open(dem_path) as src:
            elevations = [item[0] for item in sample_gen(src, coords)]

        # Find the index of the highest elevation
        max_idx = np.argmax(elevations)
        apex_lon = lons[max_idx]
        apex_lat = lats[max_idx]
        apex_elev = elevations[max_idx]

        return (apex_lon, apex_lat, apex_elev)

    except Exception as e:
        logging.warning(f"Error finding highest point between coordinates: {e}, using midpoint")
        # Fallback to midpoint if DEM query fails
        mid_lon = (receptor_coord[0] + source_coord[0]) / 2
        mid_lat = (receptor_coord[1] + source_coord[1]) / 2
        mid_elev = (receptor_coord[2] + source_coord[2]) / 2
        return (mid_lon, mid_lat, mid_elev)

def _calculate_enhanced_distance_weights(source_coords, receptor_coords, source_types, source_indices,
                                       receptor_idx, dem_path=None, prevailing_wind_direction=225):

    distance_weights = {}
    receptor_coord = receptor_coords[receptor_idx]

    for source_idx in source_indices:
        source_coord = source_coords[source_idx]
        source_type = source_types[source_idx]

        euclidean_dist = _calculate_distance(receptor_coord[1], receptor_coord[0],
                                           source_coord[1], source_coord[0])

        if source_type == 'fertilizer' or source_type == 'manure':
            # Fertilizer and manure sources: use direct euclidean distance
            effective_distance = euclidean_dist

        elif source_type == 'atmosphere':
            effective_distance = _calculate_atmospheric_distance(
                receptor_coord, source_coord, euclidean_dist,
                prevailing_wind_direction, dem_path)

        elif source_type == 'irrigation':
            effective_distance = _calculate_irrigation_distance(
                receptor_coord, source_coord, euclidean_dist, dem_path)
        else:
            effective_distance = euclidean_dist

        distance_weight = np.exp(-effective_distance / 5.0)  
        distance_weights[source_idx] = max(distance_weight, 0.01)  

    return distance_weights

def _calculate_atmospheric_distance(receptor_coord, source_coord, euclidean_dist,
                                  wind_direction=225, dem_path=None, elevation_scale_km=1000.0):
    """Compute triangular path distance for atmospheric sources using DEM.

    IMPORTANT: This function is used for candidate filtering and distance constraints.
    It returns triangular path distance (source→apex→receptor) based on DEM elevation data.
    Wind direction is handled separately as an edge feature (see _calculate_wind_direction_feature).

    The triangular path represents the actual atmospheric deposition path:
    - Pollutants are emitted from the source
    - They rise to the highest point (apex) between source and receptor
    - They then descend to the receptor

    Args:
        receptor_coord: Receptor coordinates [lon, lat, elevation]
        source_coord: Source coordinates [lon, lat, elevation]
        euclidean_dist: Euclidean distance in km (horizontal distance)
        wind_direction: (UNUSED) Kept for backward compatibility
        dem_path: Path to DEM file
        elevation_scale_km: (UNUSED) Kept for backward compatibility

    Returns:
        Triangular path distance (source→apex→receptor) in km
    """
    # If DEM is not available, fall back to euclidean distance
    if not dem_path or not os.path.exists(dem_path):
        return euclidean_dist

    try:
        # Ensure coordinates have elevation data
        if len(source_coord) < 3 or len(receptor_coord) < 3:
            return euclidean_dist

        # Find the highest point (apex) between source and receptor
        apex_lon, apex_lat, apex_elev = _find_highest_point_between(
            receptor_coord, source_coord, dem_path)

        # Calculate distance from source to apex
        source_to_apex_dist = _calculate_distance(
            source_coord[1], source_coord[0],
            apex_lat, apex_lon)

        # Calculate distance from apex to receptor
        apex_to_receptor_dist = _calculate_distance(
            apex_lat, apex_lon,
            receptor_coord[1], receptor_coord[0])

        # Total triangular path distance
        triangular_distance = source_to_apex_dist + apex_to_receptor_dist

        return triangular_distance

    except Exception as e:
        logging.warning(f"Error calculating triangular atmospheric distance: {e}, falling back to euclidean")
        return euclidean_dist


def _calculate_wind_direction_feature(receptor_coord, source_coord, wind_direction=225):
    """Calculate wind direction alignment feature for atmospheric sources.

    This feature quantifies how well the source-receptor direction aligns with
    the prevailing wind direction. It is used as an edge feature for the graph
    neural network, NOT for distance-based filtering.

    Args:
        receptor_coord: Receptor coordinates [lon, lat, ...]
        source_coord: Source coordinates [lon, lat, ...]
        wind_direction: Prevailing wind direction in degrees (0=N, 90=E, 180=S, 270=W)

    Returns:
        wind_alignment_score: Float in [0, 1] where:
            - 1.0 = perfect alignment (source is directly upwind of receptor)
            - 0.5 = perpendicular to wind direction
            - 0.0 = opposite to wind direction (source is downwind)
    """
    from math import radians, cos, sin, atan2, degrees

    # Calculate bearing from source to receptor
    lat1, lon1 = radians(source_coord[1]), radians(source_coord[0])
    lat2, lon2 = radians(receptor_coord[1]), radians(receptor_coord[0])

    dlon = lon2 - lon1
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = degrees(atan2(y, x))
    bearing = (bearing + 360) % 360  # Normalize to [0, 360)

    # Calculate angle difference between wind direction and source-receptor bearing
    angle_diff = abs(bearing - wind_direction)
    angle_diff = min(angle_diff, 360 - angle_diff)  # Take smaller angle

    # Convert angle difference to alignment score
    # angle_diff = 0° (aligned) -> score = 1.0
    # angle_diff = 90° (perpendicular) -> score = 0.5
    # angle_diff = 180° (opposite) -> score = 0.0
    wind_alignment_score = 0.5 + 0.5 * cos(radians(angle_diff))

    return wind_alignment_score

def _calculate_irrigation_distance(receptor_coord, source_coord, euclidean_dist, dem_path=None,
                                 threshold_km=3.0, curve_multiplier=1.5):
    """Compute effective distance for irrigation sources (prefer river distance)."""
    if euclidean_dist > threshold_km:
        return euclidean_dist * curve_multiplier

    if dem_path and os.path.exists(dem_path):
        try:
            river_distance = _estimate_river_distance(receptor_coord, source_coord, dem_path)
            if river_distance > 0:
                return river_distance
        except Exception as e:
            logging.warning(f"River distance estimation failed, fallback to curve distance: {e}")

    return euclidean_dist * curve_multiplier

def _estimate_river_distance(receptor_coord, source_coord, dem_path):

    try:
        euclidean_dist = _calculate_distance(receptor_coord[1], receptor_coord[0],
                                           source_coord[1], source_coord[0])

        curvature_factor = 1.3  
        estimated_river_distance = euclidean_dist * curvature_factor

        return estimated_river_distance
    except Exception as e:
        logging.warning(f"Failed to estimate river distance, fallback to 0: {e}")
        return 0

def _calculate_wind_influence_range(source_coords, receptor_coords, source_types,
                                  prevailing_wind_direction=225, max_influence_distance=50):
    """
    Calculate wind influence range for atmospheric deposition sources.

    Args:
        source_coords: Array of source coordinates
        receptor_coords: Array of receptor coordinates
        source_types: Array of source types
        prevailing_wind_direction: Prevailing wind direction in degrees (default: 225° SW)
        max_influence_distance: Maximum influence distance in km

    Returns:
        influence_matrix: Boolean matrix indicating if source influences receptor
    """
    from math import radians, cos, sin, atan2, degrees

    n_sources = len(source_coords)
    n_receptors = len(receptor_coords)
    influence_matrix = np.zeros((n_sources, n_receptors), dtype=bool)

    for i, (source_coord, source_type) in enumerate(zip(source_coords, source_types)):
        # Only apply wind influence to atmospheric sources
        if source_type != 'atmosphere':
            continue

        source_lat, source_lon = source_coord[1], source_coord[0]

        for j, receptor_coord in enumerate(receptor_coords):
            receptor_lat, receptor_lon = receptor_coord[1], receptor_coord[0]

            # Calculate distance
            distance = _calculate_distance(source_lat, source_lon, receptor_lat, receptor_lon)

            if distance > max_influence_distance:
                continue

            # Calculate bearing from source to receptor
            lat1, lon1 = radians(source_lat), radians(source_lon)
            lat2, lon2 = radians(receptor_lat), radians(receptor_lon)

            dlon = lon2 - lon1
            y = sin(dlon) * cos(lat2)
            x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
            bearing = degrees(atan2(y, x))
            bearing = (bearing + 360) % 360

            # Wind influence is strongest in downwind direction
            angle_diff = abs(bearing - prevailing_wind_direction)
            angle_diff = min(angle_diff, 360 - angle_diff)  # Take smaller angle

            # Influence decreases with angle deviation and distance
            angle_factor = cos(radians(angle_diff)) if angle_diff <= 90 else 0
            distance_factor = max(0, 1 - distance / max_influence_distance)

            # Combined influence score (threshold can be adjusted)
            influence_score = angle_factor * distance_factor
            influence_matrix[i, j] = influence_score > 0.1  # Threshold for influence

    return influence_matrix

def _calculate_atmospheric_dispersion_weights(source_coords, receptor_coords, source_types,
                                            wind_speed=5.0, stability_class='D'):
    """
    Calculate atmospheric dispersion weights using simplified Gaussian plume model.

    Args:
        source_coords: Array of source coordinates
        receptor_coords: Array of receptor coordinates
        source_types: Array of source types
        wind_speed: Wind speed in m/s
        stability_class: Atmospheric stability class (A-F)

    Returns:
        dispersion_weights: Matrix of dispersion weights
    """
    n_sources = len(source_coords)
    n_receptors = len(receptor_coords)
    dispersion_weights = np.zeros((n_sources, n_receptors))

    # Simplified dispersion parameters based on stability class
    stability_params = {
        'A': {'sigma_y': 0.22, 'sigma_z': 0.20},  # Very unstable
        'B': {'sigma_y': 0.16, 'sigma_z': 0.12},  # Moderately unstable
        'C': {'sigma_y': 0.11, 'sigma_z': 0.08},  # Slightly unstable
        'D': {'sigma_y': 0.08, 'sigma_z': 0.06},  # Neutral
        'E': {'sigma_y': 0.06, 'sigma_z': 0.03},  # Slightly stable
        'F': {'sigma_y': 0.04, 'sigma_z': 0.016}  # Moderately stable
    }

    params = stability_params.get(stability_class, stability_params['D'])

    for i, (source_coord, source_type) in enumerate(zip(source_coords, source_types)):
        if source_type != 'atmosphere':
            continue

        for j, receptor_coord in enumerate(receptor_coords):
            distance = _calculate_distance(source_coord[1], source_coord[0],
                                         receptor_coord[1], receptor_coord[0]) * 1000  # Convert to meters

            if distance == 0:
                dispersion_weights[i, j] = 1.0
                continue

            # Simplified Gaussian dispersion calculation
            sigma_y = params['sigma_y'] * distance ** 0.894
            sigma_z = params['sigma_z'] * distance ** 0.894

            # Assume ground-level release and receptor
            h = 0  # Release height
            z = 0  # Receptor height

            # Gaussian plume formula (simplified)
            exp_term = np.exp(-0.5 * ((z - h) / sigma_z) ** 2)
            dispersion_factor = 1 / (2 * np.pi * sigma_y * sigma_z * wind_speed)

            concentration = dispersion_factor * exp_term
            dispersion_weights[i, j] = concentration

    # Normalize weights
    for i in range(n_sources):
        if source_types[i] == 'atmosphere':
            total_weight = np.sum(dispersion_weights[i, :])
            if total_weight > 0:
                dispersion_weights[i, :] /= total_weight

    return dispersion_weights

class PhysGATDataset(InMemoryDataset):
    """PyG Dataset class for loading, pre-processing, and building the graph."""
    def __init__(self, root: str, config: dict, transform=None, pre_transform=None):
        self.config = config
        self.metals = config.metals
        # Get DEM path from config, can be None
        self.dem_path = config.get('dem_path', None)
        super().__init__(root, transform, pre_transform)

        # Check if processed files exist, reprocess if not
        if all(os.path.exists(path) for path in self.processed_paths):
            try:
                self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
                self.receptors_df = pd.read_pickle(self.processed_paths[1])
                self.sources_df = pd.read_pickle(self.processed_paths[2])
                self.unified_backgrounds = torch.load(self.processed_paths[3], weights_only=False)
                self.pmf_diagnostics = torch.load(self.processed_paths[4], weights_only=False)
                self.unified_backgrounds_df = pd.read_pickle(self.processed_paths[5])
            except Exception as e:
                logging.warning(f"Failed to load processed data: {e}, will reprocess data")
                for path in self.processed_paths:
                    if os.path.exists(path):
                        os.remove(path)
                self.process()
                self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
                self.receptors_df = pd.read_pickle(self.processed_paths[1])
                self.sources_df = pd.read_pickle(self.processed_paths[2])
                self.unified_backgrounds = torch.load(self.processed_paths[3], weights_only=False)
                self.pmf_diagnostics = torch.load(self.processed_paths[4], weights_only=False)
                self.unified_backgrounds_df = pd.read_pickle(self.processed_paths[5])
        else:
            logging.info("Processed data files do not exist, will reprocess data")
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
            self.receptors_df = pd.read_pickle(self.processed_paths[1])
            self.sources_df = pd.read_pickle(self.processed_paths[2])
            self.unified_backgrounds = torch.load(self.processed_paths[3], weights_only=False)
            self.pmf_diagnostics = torch.load(self.processed_paths[4], weights_only=False)
            self.unified_backgrounds_df = pd.read_pickle(self.processed_paths[5])

    @property
    def raw_dir(self) -> str:
        # Raw data files are directly in the root directory (data/)
        return self.root

    @property
    def processed_dir(self) -> str:
        # Processed files are stored in data/processed/
        processed_path = os.path.join(self.root, 'processed')
        os.makedirs(processed_path, exist_ok=True)
        return processed_path

    @property
    def raw_file_names(self) -> list[str]:
        return ['soil.csv', 'background.csv', 'atmosphere.csv', 'irrigation.csv', 'fertilizer.csv', 'manure.csv']

    @property
    def processed_file_names(self) -> list[str]:
        return ['physgat_data.pt', 'receptors.pkl', 'sources.pkl', 'backgrounds.pt', 'pmf_diagnostics.pt', 'receptor_backgrounds_df.pkl']

    def download(self):
        for file_name in self.raw_file_names:
            if not os.path.exists(os.path.join(self.raw_dir, file_name)):
                raise FileNotFoundError(f"Raw data file '{file_name}' not found in '{self.raw_dir}'.")

    def process(self):
        logging.info("=== First time data processing and graph construction... ===")
        receptors_df, sources_df, unified_backgrounds_df = self._load_and_preprocess_data()
        pmf_w, pmf_diagnostics, _ = self._run_preliminary_pmf(receptors_df)
        data = self._build_graph(receptors_df, sources_df, unified_backgrounds_df, pmf_w)
        torch.save(self.collate([data]), self.processed_paths[0])
        receptors_df.to_pickle(self.processed_paths[1])
        sources_df.to_pickle(self.processed_paths[2])
        torch.save(torch.tensor(unified_backgrounds_df[self.metals].values, dtype=torch.float), self.processed_paths[3])
        torch.save(pmf_diagnostics, self.processed_paths[4])
        unified_backgrounds_df.to_pickle(self.processed_paths[5])
        logging.info("=== Data processing and saving complete. ===")

    def _load_and_preprocess_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logging.info("--- 1. Loading and pre-processing data ---")
        soil_df = _safe_read_csv(os.path.join(self.raw_dir, 'soil.csv'))
        background_df = _safe_read_csv(os.path.join(self.raw_dir, 'background.csv'))
        receptors_df = soil_df.dropna(subset=self.metals + ['lon', 'lat']).reset_index(drop=True)
        background_df = background_df.dropna(subset=self.metals + ['lon', 'lat']).reset_index(drop=True)

        # Calculate unified background values using standardized geochemical approach
        logging.info("=== Calculating unified background values ===")
        unified_background = self._calculate_unified_background(background_df)
        logging.info(f"Unified background values: {unified_background}")

        # Create unified background DataFrame for all receptors
        unified_backgrounds_df = pd.DataFrame([unified_background] * len(receptors_df))
        unified_backgrounds_df.index = receptors_df.index

        # Calculate net pollution using unified background
        receptor_metals = receptors_df[self.metals]
        background_metals = unified_backgrounds_df[self.metals]

        logging.info("=== Processing negative soil pollution values ===")
        net_pollution = receptor_metals - background_metals
        negative_count = (net_pollution < 0).sum().sum()
        if negative_count > 0:
            logging.info(f"Found {negative_count} negative pollution values, setting them to 0")
            for metal in self.metals:
                negative_mask = net_pollution[metal] < 0
                if negative_mask.any():
                    receptors_df.loc[negative_mask, metal] = unified_backgrounds_df.loc[negative_mask, metal]
                    logging.debug(f"{metal}: Processed {negative_mask.sum()} negative values")

        # Apply Nemerow index pre-filtering
        logging.info("=== Applying Nemerow index pre-filtering ===")
        receptors_df, unified_backgrounds_df = self._apply_nemerow_prefiltering(
            receptors_df, unified_backgrounds_df)

        source_files = ['atmosphere.csv', 'irrigation.csv', 'fertilizer.csv', 'manure.csv']
        source_dfs = []
        for file in source_files:
            df = _safe_read_csv(os.path.join(self.raw_dir, file))
            df['source_type'] = file.split('.')[0]
            source_dfs.append(df)
        sources_df = pd.concat(source_dfs, ignore_index=True).dropna(subset=self.metals + ['lon', 'lat']).reset_index(drop=True)
        logging.info("Source data loading complete.")

        # Add elevation and terrain features
        receptors_df = _add_elevation_features(receptors_df, self.dem_path)
        sources_df = _add_elevation_features(sources_df, self.dem_path)

        # Add hydrological features
        receptors_df = _identify_hydrological_features(receptors_df, self.dem_path)
        sources_df = _identify_hydrological_features(sources_df, self.dem_path)

        return receptors_df, sources_df, unified_backgrounds_df

    def _calculate_unified_background(self, background_df: pd.DataFrame) -> dict:
        """
        IMPROVED: Calculate unified background values with outlier detection.

        Key improvements:
        1. Individual outlier detection for each heavy metal element using IQR method
        2. Detailed logging of outlier detection process and results
        3. Robust background value calculation using median of clean data
        4. Statistical validation using multiple outlier detection methods
        """
        import numpy as np
        from scipy import stats

        unified_background = {}

        logging.info("=== IMPROVED BACKGROUND VALUE CALCULATION WITH OUTLIER DETECTION ===")

        for metal in self.metals:
            if metal in background_df.columns:
                # Get raw data for this metal element
                raw_data = background_df[metal].dropna()
                original_count = len(raw_data)

                logging.info(f"\n--- Processing {metal} ---")
                logging.info(f"Original data points: {original_count}")
                logging.info(f"Data range: {raw_data.min():.3f} - {raw_data.max():.3f} mg/kg")
                logging.info(f"Raw mean: {raw_data.mean():.3f}, Raw median: {raw_data.median():.3f}")

                # Outlier detection using IQR method (most conservative and robust)
                Q1 = raw_data.quantile(0.25)
                Q3 = raw_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Identify outliers based on IQR bounds
                outliers_mask = (raw_data < lower_bound) | (raw_data > upper_bound)
                outlier_values = raw_data[outliers_mask]
                clean_data = raw_data[~outliers_mask]

                # Log detailed outlier detection results
                if len(outlier_values) > 0:
                    logging.info(f"Outliers detected: {len(outlier_values)} values")
                    logging.info(f"Outlier values: {outlier_values.tolist()}")
                    logging.info(f"IQR bounds: [{lower_bound:.3f}, {upper_bound:.3f}] mg/kg")
                else:
                    logging.info(f"No outliers detected using IQR method")

                logging.info(f"Clean data points after outlier removal: {len(clean_data)}")

                # Calculate background value from clean data
                if len(clean_data) > 0:
                    # Use median as it's most robust for geochemical background determination
                    background_value = clean_data.median()
                    background_mean = clean_data.mean()
                    background_std = clean_data.std()

                    unified_background[metal] = background_value

                    logging.info(f"Final background value (median): {background_value:.3f} mg/kg")
                    logging.info(f"Clean data mean: {background_mean:.3f} mg/kg")
                    logging.info(f"Clean data std: {background_std:.3f} mg/kg")

                    # Additional validation using Z-score method for cross-verification
                    z_scores = np.abs(stats.zscore(raw_data))
                    zscore_outliers = raw_data[z_scores > 3]
                    if len(zscore_outliers) > 0:
                        logging.info(f"Cross-validation - Z-score outliers (|z|>3): {zscore_outliers.tolist()}")

                else:
                    logging.warning(f"No valid data remaining after outlier removal for {metal}")
                    unified_background[metal] = 0.0

            else:
                logging.warning(f"Metal {metal} not found in background data, using default value 0")
                unified_background[metal] = 0.0

        logging.info(f"\n=== FINAL IMPROVED BACKGROUND VALUES ===")
        for metal, value in unified_background.items():
            logging.info(f"{metal}: {value:.3f} mg/kg")

        return unified_background

    def _apply_nemerow_prefiltering(self, receptors_df: pd.DataFrame,
                                   backgrounds_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply Nemerow index pre-filtering to exclude clean soil receptors.
        Only receptors exceeding clean soil threshold proceed to source apportionment.
        """
        from .analysis import calculate_nemerow_index

        # Add background columns for Nemerow calculation
        receptors_with_bg = receptors_df.copy()
        for metal in self.metals:
            receptors_with_bg[f'bg_{metal}'] = backgrounds_df[metal]

        # Calculate Nemerow index
        receptors_with_nemerow = calculate_nemerow_index(receptors_with_bg, self.metals, self.config)

        # Apply filtering based on pollution level
        clean_threshold = 1.0  # Still Clean threshold - exclude Clean and Still Clean
        if hasattr(self.config, 'pollution_analysis') and hasattr(self.config.pollution_analysis, 'pollution_levels'):
            clean_threshold = self.config.pollution_analysis.pollution_levels.still_clean

        # Filter out clean soil receptors
        polluted_mask = receptors_with_nemerow['nemerow_index'] > clean_threshold
        filtered_receptors = receptors_df[polluted_mask].reset_index(drop=True)
        filtered_backgrounds = backgrounds_df[polluted_mask].reset_index(drop=True)

        clean_count = (~polluted_mask).sum()
        polluted_count = polluted_mask.sum()

        logging.info(f"Nemerow pre-filtering results:")
        logging.info(f"  Clean soil receptors (excluded): {clean_count}")
        logging.info(f"  Polluted receptors (included): {polluted_count}")
        logging.info(f"  Clean threshold: {clean_threshold}")

        if polluted_count == 0:
            logging.warning("No polluted receptors found after Nemerow filtering!")

        return filtered_receptors, filtered_backgrounds

    def _run_preliminary_pmf(self, receptors_df) -> tuple[torch.Tensor, dict, NMF]:
        logging.info("--- 2. Running preliminary PMF to generate features ---")
        scaler = MinMaxScaler()
        X_pmf = scaler.fit_transform(receptors_df[self.metals].values)
        pmf_diagnostics = self._run_pmf_diagnostics(X_pmf)
        optimal_factors = self._select_optimal_pmf_factors(pmf_diagnostics)
        logging.info(f"Automatically selected optimal PMF factors: {optimal_factors}")
        pmf = NMF(n_components=optimal_factors, init='nndsvd', random_state=self.config.seed, max_iter=self.config.pmf.max_iter, tol=self.config.pmf.tol, alpha_W=self.config.pmf.alpha_W, alpha_H=self.config.pmf.alpha_H)
        W_pmf = pmf.fit_transform(X_pmf)
        scaler_w = StandardScaler()
        W_pmf_scaled = scaler_w.fit_transform(W_pmf)
        logging.info(f"PMF contribution feature matrix shape: {W_pmf_scaled.shape}")
        return torch.tensor(W_pmf_scaled, dtype=torch.float), pmf_diagnostics, pmf

    def _run_pmf_diagnostics(self, X_pmf: np.ndarray) -> dict:
        """Runs PMF model diagnostics by testing different numbers of factors."""
        factor_range = range(2, min(10, len(self.metals)) + 1)
        diagnostics = {'n_factors': [], 'reconstruction_error': [], 'explained_variance': []}
        total_variance = np.var(X_pmf)
        for n_factors in factor_range:
            pmf_test = NMF(n_components=n_factors, init='nndsvd', random_state=self.config.seed, max_iter=500)
            W_test = pmf_test.fit_transform(X_pmf)
            reconstruction_error = np.mean((X_pmf - (W_test @ pmf_test.components_)) ** 2)
            diagnostics['n_factors'].append(n_factors)
            diagnostics['reconstruction_error'].append(reconstruction_error)
            diagnostics['explained_variance'].append(1 - (reconstruction_error / total_variance))
        return diagnostics

    def _select_optimal_pmf_factors(self, diagnostics: dict) -> int:
        """Selects the optimal number of PMF factors using the elbow method."""
        errors = diagnostics['reconstruction_error']
        if len(errors) < 3: return 2
        improvements = [(errors[i-1] - errors[i]) / errors[i-1] for i in range(1, len(errors))]
        if len(improvements) < 2: return 3
        max_drop_idx = np.argmax([improvements[i-1] - improvements[i] for i in range(1, len(improvements))])
        return diagnostics['n_factors'][min(max_drop_idx + 1, len(diagnostics['n_factors'])-1)]

    def _build_graph(self, receptors_df, sources_df, unified_backgrounds_df, preliminary_pmf_w) -> Data:
        """Builds the source-receptor graph."""
        logging.info("--- 3. Building the source-receptor graph (with enhanced terrain and hydrological features) ---")
        data = Data()

        available_features = self.metals

        # Receptor features
        receptor_features_orig = receptors_df[available_features].values
        data.x_receptor = torch.cat([torch.tensor(StandardScaler().fit_transform(receptor_features_orig), dtype=torch.float), preliminary_pmf_w], dim=1)

        # Source features
        source_features_orig = sources_df[available_features].values
        data.x_source = torch.tensor(StandardScaler().fit_transform(source_features_orig), dtype=torch.float)

        # Store original (unscaled) metal concentrations for loss calculation
        data.x_receptor_orig = torch.tensor(receptors_df[self.metals].values, dtype=torch.float)
        data.h_source_orig = torch.tensor(sources_df[self.metals].values, dtype=torch.float)

        data.unified_backgrounds = torch.tensor(unified_backgrounds_df[self.metals].values, dtype=torch.float)

        # Extract receptor coordinates with elevation (3D: lon, lat, elevation)
        receptor_coords = receptors_df[['lon', 'lat', 'elevation']].values
        # For k-neighbors graph, use only 2D coordinates (lon, lat)
        receptor_coords_2d = receptors_df[['lon', 'lat']].values
        rr_graph = kneighbors_graph(receptor_coords_2d, n_neighbors=min(self.config.graph.k_neighbors_r, len(receptor_coords_2d)-1))
        data.edge_index_rr = torch.tensor(rr_graph.nonzero(), dtype=torch.long)

        # Build receptor-source edges with balanced constraints
        # Extract source coordinates with elevation (3D: lon, lat, elevation)
        source_coords = sources_df[['lon', 'lat', 'elevation']].values
        source_types = sources_df['source_type'].values

        # Calculate wind influence matrix for atmospheric sources (for internal ranking only)
        env_cfg = self.config.get('environment', {}) if hasattr(self.config, 'get') else {}
        prevailing_wind = env_cfg.get('prevailing_wind_direction', 225)
        logging.info(f"Using prevailing_wind_direction={prevailing_wind}° for wind influence computation")
        wind_influence = _calculate_wind_influence_range(
            source_coords, receptor_coords, source_types,
            prevailing_wind_direction=prevailing_wind
        )

        rs_edge_index, edge_weights, dem_distances, wind_features, slope_angles = self._build_constrained_receptor_source_edges(
            receptors_df, sources_df, data.unified_backgrounds,
            receptor_coords, source_coords, wind_influence)
        data.edge_index_rs = rs_edge_index

        # Store edge weights (includes wind, hydro, and other constraint weights)
        if edge_weights is not None:
            data.edge_weights = torch.tensor(edge_weights, dtype=torch.float)

        # Store DEM distances for graph relationships and loss functions
        if dem_distances is not None:
            data.dem_distances = torch.tensor(dem_distances, dtype=torch.float)

        # Store wind direction alignment features for atmospheric sources
        if wind_features is not None:
            data.wind_features = torch.tensor(wind_features, dtype=torch.float)
            logging.info(f"Wind direction features stored: {len(wind_features)} edges")

        # Store slope angles for irrigation sources (for edge features)
        if slope_angles is not None:
            data.slope_angles = torch.tensor(slope_angles, dtype=torch.float)
            non_zero_count = (data.slope_angles != 0).sum().item()
            logging.info(f"Slope angles stored: {len(slope_angles)} edges, non-zero: {non_zero_count}")
            if non_zero_count > 0:
                logging.info(f"  Min slope angle: {data.slope_angles.min().item():.4f}°, Max: {data.slope_angles.max().item():.4f}°")

        # Log edge distribution for debugging
        self._log_edge_distribution(sources_df, rs_edge_index)

        logging.info(f"Graph construction complete: {data.x_receptor.shape[0]} receptors, {data.x_source.shape[0]} sources.")
        logging.info(f"Receptor feature dimension: {data.x_receptor.shape[1]}")
        logging.info(f"Source feature dimension: {data.x_source.shape[1]}")

        return data

    def _build_constrained_receptor_source_edges(self, receptors_df, sources_df, receptor_backgrounds,
                                                receptor_coords, source_coords, wind_influence=None):
        """
        Build constrained receptor-source edges using CandidateSourceSelector.

        This is a refactored version that uses the modular CandidateSourceSelector class
        instead of the original 400-line monolithic function.

        Args:
            receptors_df: DataFrame of receptors
            sources_df: DataFrame of sources
            receptor_backgrounds: Tensor of background concentrations
            receptor_coords: Array of receptor coordinates
            source_coords: Array of source coordinates
            wind_influence: Optional wind influence matrix

        Returns:
            Tuple of (edge_index, edge_weights, dem_distances, wind_features)
        """


        # Import the new CandidateSourceSelector class
        try:
            from .candidate_source_selector import CandidateSourceSelector
        except ImportError:
            from candidate_source_selector import CandidateSourceSelector

        # Create selector instance
        selector = CandidateSourceSelector(
            config=self.config,
            receptors_df=receptors_df,
            sources_df=sources_df,
            receptor_backgrounds=receptor_backgrounds,
            metals=self.metals
        )

        # Collect all edges, weights, and features
        all_edges = []
        all_edge_weights = []
        all_wind_features = []
        all_slope_angles = []  # Store slope angles for irrigation sources

        # Process each receptor
        for receptor_idx in range(len(receptor_coords)):
            receptor_coord = receptor_coords[receptor_idx]

            # Select candidates for this receptor
            source_type_edges = selector.select_candidates_for_receptor(
                receptor_idx=receptor_idx,
                receptor_coord=receptor_coord,
                source_coords=source_coords,
                wind_influence=wind_influence,
                distance_calculator=self._calculate_dem_distance_for_edge
            )

            # Calculate edge weights
            if source_type_edges:
                edges, weights, wind_feats = selector.calculate_edge_weights(
                    source_type_edges=source_type_edges,
                    receptor_idx=receptor_idx,
                    receptor_coords=receptor_coords,
                    source_coords=source_coords
                )

                all_edges.extend(edges)
                all_edge_weights.extend(weights)
                all_wind_features.extend(wind_feats)



        # Convert to tensors
        if all_edges:
            edge_tensor = torch.tensor(all_edges, dtype=torch.long).t()
            edge_weights_array = np.array(all_edge_weights)
            wind_features_array = np.array(all_wind_features)

            # Calculate DEM distances and slope angles for all edges
            dem_distances = []
            for receptor_idx, source_idx in all_edges:
                receptor_coord = receptor_coords[receptor_idx]
                source_coord = source_coords[source_idx]
                source_type = sources_df.iloc[source_idx]['source_type']

                # Calculate DEM-based distance using enhanced distance calculation
                dem_distance = self._calculate_dem_distance_for_edge(
                    receptor_coord, source_coord, source_type)
                dem_distances.append(dem_distance)

                # Calculate slope angle for irrigation sources
                if source_type == 'irrigation':
                    # Calculate horizontal distance (euclidean in 2D)
                    horizontal_dist = _calculate_distance(
                        receptor_coord[1], receptor_coord[0],
                        source_coord[1], source_coord[0])

                    # Calculate elevation difference
                    elevation_diff = receptor_coord[2] - source_coord[2]

                    # Calculate slope angle in degrees
                    if horizontal_dist > 0:
                        slope_angle = np.degrees(np.arctan(elevation_diff / (horizontal_dist * 1000)))
                    else:
                        slope_angle = 0.0

                    all_slope_angles.append(slope_angle)

                    # Debug: Log first few irrigation edges
                    if len(all_slope_angles) <= 5:
                        logging.debug(f"Irrigation edge {len(all_slope_angles)}: receptor_elev={receptor_coord[2]:.2f}, source_elev={source_coord[2]:.2f}, horiz_dist={horizontal_dist:.4f}km, slope_angle={slope_angle:.2f}°")
                else:
                    # For non-irrigation sources, set slope angle to 0
                    all_slope_angles.append(0.0)

            dem_distances_array = np.array(dem_distances)
            slope_angles_array = np.array(all_slope_angles)
        else:
            edge_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_weights_array = np.array([])
            wind_features_array = np.array([])
            dem_distances_array = None
            slope_angles_array = None

        return edge_tensor, edge_weights_array, dem_distances_array, wind_features_array, slope_angles_array

    def _build_constrained_receptor_source_edges_original(self, receptors_df, sources_df, receptor_backgrounds,
                                                receptor_coords, source_coords, wind_influence=None):
        """
        ORIGINAL IMPLEMENTATION (DEPRECATED - kept for reference and testing)

        Build constrained receptor-source edges with strict two-stage filtering:
        Stage 1: Distance-based pre-filtering
        Stage 2: Scoring and top-K selection
        """
        print(f"\n=== STARTING TWO-STAGE FILTERING (ORIGINAL) ===")
        print(f"Input: {len(receptor_coords)} receptors, {len(source_coords)} sources")

        all_edges = []
        edge_weights = []
        wind_features = []  # Store wind direction alignment features for atmospheric sources
        receptor_profiles_contrib_only = torch.relu(torch.tensor(receptors_df[self.metals].values) - receptor_backgrounds).numpy()
        source_chem_profiles_np = sources_df[self.metals].values

        # Get candidate selection configuration
        candidate_config = self.config.graph.get('candidate_selection', {})
        max_dist_config = self.config.graph.get('max_distance_km', {})
        default_max_dist = max_dist_config.get('default', 5)

        logging.info("Configuration loaded:")
        logging.info(f"  Distance thresholds: {max_dist_config}")
        logging.info(f"  Candidate config: {candidate_config}")

        # Source-type specific scoring weights for candidate ranking
        scoring_weights_config = candidate_config.get('scoring_weights', {})
        zero_candidate_config = candidate_config.get('zero_candidate_exception', {})

        def get_scoring_weights(source_type, is_exception=False):
            """Get scoring weights for a specific source type"""
            if is_exception and zero_candidate_config.get('enabled', True):
                # Use exception weights when no candidates found in distance filtering
                exception_key = f"{source_type}_exception"
                if exception_key in zero_candidate_config:
                    return zero_candidate_config[exception_key]
                else:
                    # Fallback: reduce distance weight and redistribute
                    normal_weights = scoring_weights_config.get(source_type, scoring_weights_config.get('default', {}))
                    distance_reduction = zero_candidate_config.get('distance_weight_reduction', 0.3)
                    exception_weights = normal_weights.copy()
                    if 'distance_score' in exception_weights:
                        old_distance = exception_weights['distance_score']
                        exception_weights['distance_score'] = distance_reduction
                        # Redistribute the reduced distance weight to other factors
                        redistribution = (old_distance - distance_reduction) / max(1, len(exception_weights) - 1)
                        for key in exception_weights:
                            if key != 'distance_score':
                                exception_weights[key] += redistribution
                    return exception_weights
            else:
                # Use normal weights
                return scoring_weights_config.get(source_type, scoring_weights_config.get('default', {
                    'distance_score': 0.5, 'chemical_score': 0.2,
                    'strength_score': 0.15, 'wind_score': 0.15
                }))
        # === Optimized candidate selection configuration ===
        max_candidates_per_type = candidate_config.get('max_candidates_per_type', 8)
        min_candidates_per_type = candidate_config.get('min_candidates_per_type', 4)  
        global_min_candidates = candidate_config.get('global_min_candidates', 16)     
        global_max_candidates = candidate_config.get('global_max_candidates', 12)     
        max_total_connections = candidate_config.get('max_total_connections_per_receptor', 32)
        min_total_connections = candidate_config.get('min_total_connections_per_receptor', 16)
        use_strict_filtering = candidate_config.get('use_strict_filtering', True)

        # Fix configuration conflicts
        if global_max_candidates < global_min_candidates:
            logging.warning(f"Configuration conflict: global_max_candidates ({global_max_candidates}) < global_min_candidates ({global_min_candidates})")
            logging.warning(f"Adjusting global_min_candidates to {global_max_candidates}")
            global_min_candidates = global_max_candidates

        if min_total_connections > global_max_candidates:
            logging.warning(f"Configuration conflict: min_total_connections ({min_total_connections}) > global_max_candidates ({global_max_candidates})")
            logging.warning(f"Adjusting min_total_connections to {global_max_candidates}")
            min_total_connections = global_max_candidates

        logging.info(f"=== Optimized candidate selection ===")
        logging.info(f"  Min candidates per type: {min_candidates_per_type}")
        logging.info(f"  Max candidates per type: {max_candidates_per_type}")
        logging.info(f"  Global min candidates: {global_min_candidates}")
        logging.info(f"  Global max candidates: {global_max_candidates}")
        logging.info(f"  Max total connections: {max_total_connections}")
        logging.info(f"  Min total connections: {min_total_connections}")

        for receptor_idx in range(len(receptor_coords)):
            source_type_edges = {}
            receptor_coord = receptor_coords[receptor_idx]

            for source_type in sources_df['source_type'].unique():
                source_indices_of_type = sources_df[sources_df['source_type'] == source_type].index
                if len(source_indices_of_type) == 0:
                    continue

                # === STAGE 1: Distance-based Pre-filtering ===
                max_dist = max_dist_config.get(source_type, default_max_dist)

                distance_filtered_indices = []
                distance_values = []

                # Apply strict distance filtering
                # IMPORTANT: Atmosphere sources use DEM climbing distance, others use Euclidean distance
                for s_idx in source_indices_of_type:
                    source_coord = source_coords[s_idx]
                    if source_type == 'atmosphere':
                        # Atmosphere sources: use DEM climbing distance
                        dist = self._calculate_dem_distance_for_edge(
                            receptor_coord, source_coord, source_type
                        )
                    else:
                        # Other sources: use Euclidean distance
                        dist = _calculate_distance(receptor_coord[1], receptor_coord[0],
                                                 source_coord[1], source_coord[0])

                    # Only include sources within distance threshold
                    if dist <= max_dist:
                        distance_filtered_indices.append(s_idx)
                        distance_values.append(dist)

                # Handle zero candidate exception with complex rules
                is_exception_case = False
                if not distance_filtered_indices:
                    # STRICT RULE: manure sources NEVER allow exception mechanism
                    if source_type == 'manure':
                        logging.info(f"Receptor {receptor_idx} {source_type} sources: No sources within {max_dist}km. STRICT POLICY: No exception allowed for manure sources, skipping.")
                        continue

                    # For other source types, apply exception mechanism if enabled
                    if zero_candidate_config.get('enabled', True):
                        # Apply complex exception mechanism based on source type
                        exception_result = self._apply_complex_exception_mechanism(
                            receptor_idx, receptor_coord, source_type, sources_df, source_coords, max_dist)

                        if exception_result:
                            distance_filtered_indices, distance_values, is_exception_case = exception_result
                            logging.info(f"Receptor {receptor_idx} {source_type} sources: Applied complex exception mechanism, found {len(distance_filtered_indices)} candidates")
                        else:
                            logging.debug(f"Receptor {receptor_idx} {source_type} sources: No sources found through exception mechanism")
                            continue
                    else:
                        logging.debug(f"Receptor {receptor_idx} {source_type} sources: No sources within {max_dist}km, skipping")
                        continue

                # === Stage 2: Multi-dimension scoring and ranking ===
                candidates_df = pd.DataFrame(index=distance_filtered_indices)

                # 1. Distance score (closer = higher score)
                max_dist_in_candidates = max(distance_values) if distance_values else 1.0
                candidates_df['distance_score'] = [(max_dist_in_candidates - d) / max_dist_in_candidates
                                                 for d in distance_values]

                # 2. Chemical similarity score (no hard filtering)
                # Compute similarity for all sources without filtering
                chem_scores = cosine_similarity([receptor_profiles_contrib_only[receptor_idx]],
                                              source_chem_profiles_np[distance_filtered_indices])[0]

                candidates_df['chemical_score'] = chem_scores

                # 3. Source strength score
                source_concentration_sum = sources_df.loc[distance_filtered_indices, self.metals].sum(axis=1)
                numerical_eps = getattr(self.config.graph, 'numerical_epsilon', 1e-8)
                max_conc = source_concentration_sum.max()
                candidates_df['strength_score'] = source_concentration_sum / (max_conc + numerical_eps)

                # 4. Wind direction score (atmosphere only)
                if source_type == 'atmosphere' and wind_influence is not None:
                    wind_scores = []
                    for idx in distance_filtered_indices:
                        wind_score = wind_influence[idx, receptor_idx]
                        min_wind_weight = getattr(self.config.graph, 'min_wind_weight', 0.1)
                        wind_scores.append(max(wind_score, min_wind_weight))
                    candidates_df['wind_score'] = wind_scores
                else:
                    # For non-atmosphere sources, set default wind score
                    candidates_df['wind_score'] = 1.0

                # 5. Hydrological score (irrigation only)
                if source_type == 'irrigation':
                    try:
                        hydro_scores = self._calculate_hydrological_influence(
                            distance_filtered_indices, receptor_idx, sources_df, receptors_df)
                        candidates_df['hydro_score'] = hydro_scores
                    except Exception as e:
                        logging.warning(f"Hydrological influence calculation failed, fallback to distance score: {e}")
                        candidates_df['hydro_score'] = candidates_df['distance_score']  # Fallback to distance score
                else:
                    candidates_df['hydro_score'] = candidates_df['distance_score']

                # === Stage 3: Weighted total score ===
                # Get source-type specific scoring weights
                scoring_weights = get_scoring_weights(source_type, is_exception_case)

                # Calculate weighted total score using source-type specific weights
                total_score = np.zeros(len(candidates_df))

                # Apply weights based on source type and available scores
                if 'distance_score' in scoring_weights:
                    total_score += candidates_df['distance_score'] * scoring_weights['distance_score']
                if 'chemical_score' in scoring_weights:
                    total_score += candidates_df['chemical_score'] * scoring_weights['chemical_score']
                if 'strength_score' in scoring_weights:
                    total_score += candidates_df['strength_score'] * scoring_weights['strength_score']
                if 'wind_score' in scoring_weights and source_type == 'atmosphere':
                    total_score += candidates_df['wind_score'] * scoring_weights['wind_score']
                if 'hydro_score' in scoring_weights and source_type == 'irrigation':
                    total_score += candidates_df['hydro_score'] * scoring_weights['hydro_score']

                candidates_df['total_score'] = total_score

                # Log scoring weights used
                if is_exception_case:
                    logging.info(f"Receptor {receptor_idx} {source_type} sources: Using exception weights: {scoring_weights}")
                else:
                    logging.info(f"Receptor {receptor_idx} {source_type} sources: Using normal weights: {scoring_weights}")

                # Sort by total score
                sorted_candidates = candidates_df.sort_values('total_score', ascending=False)

                # === Stage 2: Optimized Top-K selection ===
                # Ensure at least min_candidates_per_type candidates per type
                available_candidates = len(sorted_candidates)

                # New logic: if no candidates in range, handle based on source type
                if available_candidates == 0:
                    # STRICT RULE: manure sources NEVER allow out-of-range search
                    if source_type == 'manure':
                        logging.info(f"Receptor {receptor_idx} {source_type}: No candidates in range (max: {max_dist}km). STRICT POLICY: No out-of-range search allowed for manure sources.")
                        sorted_candidates = pd.DataFrame()
                        available_candidates = 0
                    else:
                        # For atmosphere, irrigation, fertilizer: allow finding 1 best candidate outside range
                        logging.info(f"Receptor {receptor_idx} {source_type}: No candidates in range (max: {max_dist}km), searching for 1 best candidate outside range...")
                        expanded_candidates = self._expand_candidate_search(
                            receptor_idx, receptor_coord, source_type, sources_df, source_coords,
                            max_dist, 1, scoring_weights)  # Only find 1 candidate

                        if expanded_candidates is not None:
                            sorted_candidates = expanded_candidates.head(1)  # Take only the best 1
                            available_candidates = len(sorted_candidates)
                            logging.info(f"Receptor {receptor_idx} {source_type}: Found {available_candidates} candidate outside range")
                        else:
                            # No candidates found even outside range
                            sorted_candidates = pd.DataFrame()
                            available_candidates = 0

                elif available_candidates < min_candidates_per_type:
                    # NEW RULE: Do NOT expand search just because we have fewer than min_candidates_per_type
                    # Only expand when we have ZERO candidates (handled above)
                    # If we have some candidates within range, use them even if fewer than minimum
                    if source_type == 'manure':
                        logging.info(f"Receptor {receptor_idx} {source_type}: Only {available_candidates} candidates in range. STRICT POLICY: No expansion allowed for manure sources.")
                        # Keep only the candidates found within range
                    else:
                        logging.info(f"Receptor {receptor_idx} {source_type}: Only {available_candidates} candidates in range (min: {min_candidates_per_type}). NEW POLICY: Using available candidates within range, no expansion.")
                        # Keep only the candidates found within range - do NOT expand

                # Select candidates: at most max_candidates_per_type (3)
                if use_strict_filtering:
                    target_count = min(max_candidates_per_type, available_candidates)
                    selected_candidates = sorted_candidates.head(target_count)
                else:
                    selected_candidates = sorted_candidates

                # Final validation: record selected candidates
                final_indices = selected_candidates.index.tolist()
                logging.info(f"Receptor {receptor_idx} {source_type} sources: {len(final_indices)} candidates selected (max: {max_candidates_per_type}, available: {available_candidates})")

                # Store candidate information
                source_type_edges[source_type] = {
                    'indices': final_indices,
                    'scores': selected_candidates,
                    'effective_k': len(selected_candidates),
                    'total_filtered': len(distance_filtered_indices),
                    'distance_filtered': len(distance_filtered_indices),
                    'final_selected': len(selected_candidates)
                }

            # === Candidate Validation and Connection Establishment ===
            if not source_type_edges:
                continue

            # Calculate total connections for this receptor
            total_connections_for_receptor = 0
            for source_type, edge_data in source_type_edges.items():
                total_connections_for_receptor += edge_data['final_selected']

            # === Check global minimum candidates ===
            if total_connections_for_receptor < global_min_candidates:
                logging.info(f"Receptor {receptor_idx}: Only {total_connections_for_receptor} candidates, need {global_min_candidates}. Adding more candidates...")
                source_type_edges = self._ensure_global_min_candidates(
                    receptor_idx, receptor_coord, source_type_edges, sources_df, source_coords,
                    global_min_candidates, total_connections_for_receptor)

                # Recompute total connections
                total_connections_for_receptor = 0
                for source_type, edge_data in source_type_edges.items():
                    total_connections_for_receptor += edge_data['final_selected']

            # === Check global maximum candidates ===
            if total_connections_for_receptor > global_max_candidates:
                logging.info(f"Receptor {receptor_idx}: Too many candidates ({total_connections_for_receptor}), reducing to {global_max_candidates}...")
                source_type_edges = self._reduce_to_global_max_candidates(
                    receptor_idx, source_type_edges, global_max_candidates)

                # Recompute total connections
                total_connections_for_receptor = 0
                for source_type, edge_data in source_type_edges.items():
                    total_connections_for_receptor += edge_data['final_selected']

            # Apply maximum connection constraint
            if use_strict_filtering and total_connections_for_receptor > max_total_connections:
                # Get distance limits from config
                max_dist_config = self.config.graph.get('max_distance_km', {})

                # Need to reduce connections - prioritize by total score across all source types
                # BUT ENFORCE DISTANCE LIMITS
                all_candidates = []
                for source_type, edge_data in source_type_edges.items():
                    # Get max distance for this source type
                    max_dist_for_type = max_dist_config.get(source_type, 5.0)

                    for idx in edge_data['indices']:
                        # Calculate distance to verify it's within limits
                        source_coord = source_coords[idx]
                        dist = self._calculate_dem_distance_for_edge(
                            receptor_coord, source_coord, source_type
                        )

                        # STRICT RULE: manure sources must be within 0.2km, no exceptions
                        if source_type == 'manure' and dist > 0.2:
                            logging.warning(f"Receptor {receptor_idx}: Skipping manure source {idx} at {dist:.2f}km (exceeds 0.2km limit)")
                            continue  # Skip manure sources beyond 0.2km

                        # For other source types, also respect their distance limits
                        if dist > max_dist_for_type:
                            logging.warning(f"Receptor {receptor_idx}: Skipping {source_type} source {idx} at {dist:.2f}km (exceeds {max_dist_for_type}km limit)")
                            continue  # Skip sources beyond their type-specific limits

                        score = edge_data['scores'].loc[idx, 'total_score']
                        all_candidates.append((score, source_type, idx))

                # Sort by score and take top max_total_connections
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                selected_candidates = all_candidates[:max_total_connections]

                # Rebuild source_type_edges with selected candidates
                new_source_type_edges = {}
                for score, source_type, idx in selected_candidates:
                    if source_type not in new_source_type_edges:
                        new_source_type_edges[source_type] = {
                            'indices': [],
                            'scores': source_type_edges[source_type]['scores'],
                            'final_selected': 0
                        }
                    new_source_type_edges[source_type]['indices'].append(idx)
                    new_source_type_edges[source_type]['final_selected'] += 1

                source_type_edges = new_source_type_edges
                total_connections_for_receptor = len(selected_candidates)

                logging.debug(f"Receptor {receptor_idx}: Reduced connections from {len(all_candidates)} to {total_connections_for_receptor} (max: {max_total_connections})")

            logging.debug(f"Receptor {receptor_idx} final connections: {total_connections_for_receptor}")

            # Record if no candidates pass filtering
            if total_connections_for_receptor == 0:
                logging.info(f"Receptor {receptor_idx}: No candidates passed filtering, no connections will be established")

            for source_type, edge_data in source_type_edges.items():
                # Get source-type specific weights for edge weight calculation
                edge_scoring_weights = get_scoring_weights(source_type, False)  # Use normal weights for edge weights

                for source_idx in edge_data['indices']:
                    all_edges.append((receptor_idx, source_idx))

                    scores = edge_data['scores']

                    distance_score = scores.loc[source_idx, 'distance_score']
                    chemical_score = scores.loc[source_idx, 'chemical_score']
                    strength_score = scores.loc[source_idx, 'strength_score']
                    wind_score = scores.loc[source_idx, 'wind_score'] if 'wind_score' in scores.columns else 0.0
                    hydro_score = scores.loc[source_idx, 'hydro_score'] if 'hydro_score' in scores.columns else distance_score

                    final_weight = 0.0
                    if 'distance_score' in edge_scoring_weights:
                        final_weight += distance_score * edge_scoring_weights['distance_score']
                    if 'chemical_score' in edge_scoring_weights:
                        final_weight += chemical_score * edge_scoring_weights['chemical_score']
                    if 'strength_score' in edge_scoring_weights:
                        final_weight += strength_score * edge_scoring_weights['strength_score']
                    if 'wind_score' in edge_scoring_weights and source_type == 'atmosphere':
                        final_weight += wind_score * edge_scoring_weights['wind_score']
                    if 'hydro_score' in edge_scoring_weights and source_type == 'irrigation':
                        final_weight += hydro_score * edge_scoring_weights['hydro_score']

                    edge_weights.append(max(final_weight, 0.01))

                    # Calculate wind direction alignment feature for atmospheric sources
                    if source_type == 'atmosphere':
                        receptor_coord = receptor_coords[receptor_idx]
                        source_coord = source_coords[source_idx]
                        env_cfg = self.config.get('environment', {}) if hasattr(self.config, 'get') else {}
                        prevailing_wind = env_cfg.get('prevailing_wind_direction', 225)
                        wind_alignment = _calculate_wind_direction_feature(
                            receptor_coord, source_coord, prevailing_wind)
                        wind_features.append(wind_alignment)
                    else:
                        # For non-atmospheric sources, set default value (0.5 = neutral)
                        wind_features.append(0.5)

        if not all_edges:
            logging.warning("No edges created after filtering - returning empty edge tensor")
            return torch.empty((2, 0), dtype=torch.long), None

        # Log filtering statistics
        total_sources = len(sources_df)
        total_connections = len(all_edges)
        avg_connections_per_receptor = total_connections / len(receptor_coords) if len(receptor_coords) > 0 else 0

        print(f"\n=== Two-Stage Filtering Results ===")
        print(f"Total sources available: {total_sources}")
        print(f"Total connections created: {total_connections}")
        print(f"Average connections per receptor: {avg_connections_per_receptor:.2f}")
        print(f"Distance thresholds: atmosphere={max_dist_config.get('atmosphere', default_max_dist)}km, "
              f"irrigation={max_dist_config.get('irrigation', default_max_dist)}km, "
              f"fertilizer={max_dist_config.get('fertilizer', default_max_dist)}km, "
              f"manure={max_dist_config.get('manure', default_max_dist)}km")
        print(f"Top-K per source type: {max_candidates_per_type}")
        print(f"Max connections per receptor: {max_total_connections}")
        print(f"Strict filtering enabled: {use_strict_filtering}")
        print("=" * 50)

        logging.info(f"=== Two-Stage Filtering Results ===")
        logging.info(f"Total sources available: {total_sources}")
        logging.info(f"Total connections created: {total_connections}")
        logging.info(f"Average connections per receptor: {avg_connections_per_receptor:.2f}")
        logging.info(f"Distance thresholds: atmosphere={max_dist_config.get('atmosphere', default_max_dist)}km, "
                    f"irrigation={max_dist_config.get('irrigation', default_max_dist)}km, "
                    f"fertilizer={max_dist_config.get('fertilizer', default_max_dist)}km, "
                    f"manure={max_dist_config.get('manure', default_max_dist)}km")
        logging.info(f"Top-K per source type: {max_candidates_per_type}")
        logging.info(f"Max connections per receptor: {max_total_connections}")

        # Calculate DEM distances for all edges (for graph relationships and loss functions)
        dem_distances = []
        for receptor_idx, source_idx in all_edges:
            receptor_coord = receptor_coords[receptor_idx]
            source_coord = source_coords[source_idx]
            source_type = sources_df.iloc[source_idx]['source_type']

            # Calculate DEM-based distance using enhanced distance calculation
            dem_distance = self._calculate_dem_distance_for_edge(
                receptor_coord, source_coord, source_type)
            dem_distances.append(dem_distance)

        edge_tensor = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_weights_array = np.array(edge_weights) if edge_weights else None
        dem_distances_array = np.array(dem_distances) if dem_distances else None
        wind_features_array = np.array(wind_features) if wind_features else None

        return edge_tensor, edge_weights_array, dem_distances_array, wind_features_array

    def _calculate_hydrological_influence(self, source_indices, receptor_idx, sources_df, receptors_df):

        hydro_scores = []
        receptor_coord = receptors_df.iloc[receptor_idx][['lon', 'lat']].values

        for source_idx in source_indices:
            source_coord = sources_df.iloc[source_idx][['lon', 'lat']].values

            source_flow_dir = sources_df.iloc[source_idx].get('flow_direction', 0)
            source_flow_acc = sources_df.iloc[source_idx].get('flow_accumulation', 0)
            source_stream_dist = sources_df.iloc[source_idx].get('distance_to_stream', 1000)

            direction_vector = receptor_coord - source_coord
            direction_angle = np.arctan2(direction_vector[1], direction_vector[0]) * 180 / np.pi

            flow_alignment = np.cos(np.radians(direction_angle - source_flow_dir))
            flow_score = max(flow_alignment, 0.1) 

            stream_score = np.exp(-source_stream_dist / 1000)  

            flow_acc_score = min(source_flow_acc / 1000, 1.0)  

            hydro_score = 0.4 * flow_score + 0.3 * stream_score + 0.3 * flow_acc_score
            hydro_scores.append(max(hydro_score, 0.1)) 

        return hydro_scores

    def _log_edge_distribution(self, sources_df, edge_index):
        if edge_index.shape[1] == 0:
            logging.warning("No edges created!")
            return

        edge_source_indices = edge_index[1].numpy()
        edge_source_types = sources_df.iloc[edge_source_indices]['source_type'].value_counts()

        logging.info("=== Edge distribution statistics ===")
        total_edges = len(edge_source_indices)
        for source_type, count in edge_source_types.items():
            percentage = count / total_edges * 100
            logging.info(f"{source_type}: {count} edges ({percentage:.2f}%)")

        # Calculate average connectivity for each source type
        source_type_counts = sources_df['source_type'].value_counts()
        logging.info("=== Average connectivity ===")
        for source_type in source_type_counts.index:
            type_sources = sources_df[sources_df['source_type'] == source_type].index
            type_edges = sum(1 for idx in edge_source_indices if idx in type_sources.values)
            avg_connections = type_edges / len(type_sources) if len(type_sources) > 0 else 0
            logging.info(f"{source_type}: {avg_connections:.2f} avg connections per source")

    def _apply_complex_exception_mechanism(self, receptor_idx, receptor_coord, source_type,
                                         sources_df, source_coords, original_max_dist):
        """
        Apply complex exception mechanism for zero candidate cases.

        STRICT RULES (Updated):
        1. For atmosphere: expand search range, find 1 best match (max 5km enforced)
        2. For irrigation: expand search range, find 1 best match (max 1km enforced)
        3. For fertilizer: expand search range, find 1 best match (max 1km enforced)
        4. For manure: STRICTLY FORBIDDEN - this function should never be called for manure sources
        """

        # STRICT RULE: manure sources should NEVER reach this function
        if source_type == 'manure':
            logging.error(f"Receptor {receptor_idx}: CRITICAL ERROR - Exception mechanism called for manure source type. This should never happen!")
            return None

        if source_type == 'atmosphere':
            # For atmosphere: allow finding 1 best source within reasonable range
            expanded_range = min(original_max_dist * 1.5, 10.0)  # Cap at 10km
            return self._find_best_source_in_range(
                receptor_coord, source_type, sources_df, source_coords, expanded_range)

        elif source_type == 'irrigation':
            # For irrigation: allow finding 1 best source within reasonable range
            expanded_range = min(original_max_dist * 1.5, 5.0)  # Cap at 5km
            return self._find_best_source_in_range(
                receptor_coord, source_type, sources_df, source_coords, expanded_range)

        elif source_type == 'fertilizer':
            # For fertilizer: allow finding 1 best source within reasonable range
            expanded_range = min(original_max_dist * 1.5, 5.0)  # Cap at 5km
            return self._find_best_source_in_range(
                receptor_coord, source_type, sources_df, source_coords, expanded_range)

        else:
            # Default fallback
            return self._find_best_source_in_range(
                receptor_coord, source_type, sources_df, source_coords, float('inf'))

    def _find_best_source_in_range(self, receptor_coord, source_type, sources_df, source_coords, max_range):
        """Find the best source of given type within specified range."""
        source_indices_of_type = sources_df[sources_df['source_type'] == source_type].index
        if len(source_indices_of_type) == 0:
            return None

        candidates = []
        for s_idx in source_indices_of_type:
            source_coord = source_coords[s_idx]
            dist = self._calculate_dem_distance_for_edge(
                receptor_coord, source_coord, source_type
            )
            if dist <= max_range:
                candidates.append((s_idx, dist))

        if not candidates:
            return None

        # Sort by distance and return the closest one
        candidates.sort(key=lambda x: x[1])
        best_source_idx, best_dist = candidates[0]

        return [best_source_idx], [best_dist], True

    def _apply_fertilizer_manure_exception(self, receptor_idx, receptor_coord, source_type,
                                        sources_df, source_coords):
        """
        Apply complex exception mechanism for fertilizer and manure sources.
        """
        # Check what sources exist in 0.1km range for both fertilizer and manure
        fertilizer_in_01km = self._find_sources_in_range(
            receptor_coord, 'fertilizer', sources_df, source_coords, 0.1)
        manure_in_01km = self._find_sources_in_range(
            receptor_coord, 'manure', sources_df, source_coords, 0.1)

        has_fertilizer_01km = len(fertilizer_in_01km) > 0
        has_manure_01km = len(manure_in_01km) > 0

        if source_type == 'fertilizer':
            # Modified Rule: Reduce mutual exclusion to allow more balanced contributions
            # Only skip fertilizer if manure is very close (within 0.05km) and dominant
            if has_manure_01km:
                # Check if manure sources are very close (< 0.05km)
                very_close_manure = self._find_sources_in_range(
                    receptor_coord, 'manure', sources_df, source_coords, 0.05)
                if len(very_close_manure) > 0:
                    logging.info(f"Receptor {receptor_idx}: Very close manure (<0.05km), skipping fertilizer search")
                    return None
                else:
                    # Manure exists but not very close, allow fertilizer search
                    logging.info(f"Receptor {receptor_idx}: Manure in 0.1km but not very close, allowing fertilizer search")

            # Search for fertilizer in expanding ranges regardless of distant manure presence
            if not has_fertilizer_01km:
                logging.info(f"Receptor {receptor_idx}: Searching fertilizer in expanding ranges")
                return self._search_fertilizer_expanding_ranges(receptor_coord, sources_df, source_coords)

            # If we reach here, should not happen in exception mechanism
            return None

        elif source_type == 'manure':
            # Modified Rule: Reduce mutual exclusion to allow more balanced contributions
            # Only skip manure if fertilizer is very close (within 0.05km) and dominant
            if has_fertilizer_01km:
                # Check if fertilizer sources are very close (< 0.05km)
                very_close_fertilizer = self._find_sources_in_range(
                    receptor_coord, 'fertilizer', sources_df, source_coords, 0.05)
                if len(very_close_fertilizer) > 0:
                    logging.info(f"Receptor {receptor_idx}: Very close fertilizer (<0.05km), skipping manure search")
                    return None
                else:
                    # Fertilizer exists but not very close, allow manure search
                    logging.info(f"Receptor {receptor_idx}: Fertilizer in 0.1km but not very close, allowing manure search")

            # Search for manure in 0.2km range regardless of distant fertilizer presence
            if not has_manure_01km:
                manure_in_02km = self._find_sources_in_range(
                    receptor_coord, 'manure', sources_df, source_coords, 0.2)
                if len(manure_in_02km) > 0:
                    # Found manure in 0.2km, select the best one
                    best_manure = self._select_best_source_by_score(
                        receptor_coord, manure_in_02km, sources_df, source_coords)
                    return [best_manure[0]], [best_manure[1]], True
                else:
                    logging.info(f"Receptor {receptor_idx}: No manure found in 0.2km, abandoning manure search")
                    return None

            # If we reach here, should not happen in exception mechanism
            return None

        return None

    def _find_sources_in_range(self, receptor_coord, source_type, sources_df, source_coords, max_dist):
        """Find all sources of given type within specified distance."""
        source_indices_of_type = sources_df[sources_df['source_type'] == source_type].index
        sources_in_range = []

        for s_idx in source_indices_of_type:
            source_coord = source_coords[s_idx]
            dist = self._calculate_dem_distance_for_edge(
                receptor_coord, source_coord, source_type
            )
            if dist <= max_dist:
                sources_in_range.append((s_idx, dist))

        return sources_in_range

    def _search_fertilizer_expanding_ranges(self, receptor_coord, sources_df, source_coords):
        """Search for fertilizer sources in expanding ranges: 2km, 3km, 4km, ..."""
        current_range = 2.0  # Start from 2km
        max_search_range = 50.0  # Reasonable upper limit

        while current_range <= max_search_range:
            fertilizer_sources = self._find_sources_in_range(
                receptor_coord, 'fertilizer', sources_df, source_coords, current_range)

            if len(fertilizer_sources) > 0:
                # Found fertilizer sources, select the best one
                best_fertilizer = self._select_best_source_by_score(
                    receptor_coord, fertilizer_sources, sources_df, source_coords)
                logging.info(f"Found fertilizer source at {current_range}km range")
                return [best_fertilizer[0]], [best_fertilizer[1]], True

            current_range += 1.0  # Expand by 1km each iteration

        logging.warning(f"No fertilizer sources found within {max_search_range}km")
        return None

    def _select_best_source_by_score(self, receptor_coord, source_candidates, sources_df, source_coords):
        """Select the best source from candidates using the same scoring mechanism."""
        if not source_candidates:
            return None

        # For now, simply return the closest source
        # In the future, this could use the full scoring mechanism
        source_candidates.sort(key=lambda x: x[1])  # Sort by distance
        return source_candidates[0]  # Return (source_idx, distance)

    def _calculate_dem_distance_for_edge(self, receptor_coord, source_coord, source_type):
        """
        Calculate DEM-based distance for a specific edge based on source type.
        This is used for graph relationships and loss functions.
        """
        # Basic euclidean distance
        euclidean_dist = _calculate_distance(receptor_coord[1], receptor_coord[0],
                                           source_coord[1], source_coord[0])

        if source_type == 'fertilizer' or source_type == 'manure':
            # Check if DEM-based distance is enabled for these source types
            terrain_cfg = self.config.graph.get('terrain', {}) if hasattr(self.config, 'graph') else {}

            if source_type == 'fertilizer' and terrain_cfg.get('enable_fertilizer_dem', False):
                # Apply terrain-based distance calculation for fertilizer sources
                slope_penalty = terrain_cfg.get('fertilizer_slope_penalty', 1.2)
                return self._calculate_terrain_distance(
                    receptor_coord, source_coord, euclidean_dist, slope_penalty)
            elif source_type == 'manure' and terrain_cfg.get('enable_manure_dem', False):
                # Apply terrain-based distance calculation for manure sources
                slope_penalty = terrain_cfg.get('manure_slope_penalty', 1.3)
                return self._calculate_terrain_distance(
                    receptor_coord, source_coord, euclidean_dist, slope_penalty)
            else:
                # Use direct euclidean distance if DEM is disabled
                return euclidean_dist

        elif source_type == 'atmosphere':
            # Atmospheric sources: consider wind and terrain effects using config
            env_cfg = self.config.get('environment', {}) if hasattr(self.config, 'get') else {}
            prevailing_wind = env_cfg.get('prevailing_wind_direction', 225)
            terrain_cfg = self.config.graph.get('terrain', {}) if hasattr(self.config, 'graph') else {}
            elev_scale = terrain_cfg.get('elevation_diff_scale_km', 1000.0)
            return _calculate_atmospheric_distance(
                receptor_coord, source_coord, euclidean_dist,
                wind_direction=prevailing_wind, dem_path=self.dem_path, elevation_scale_km=elev_scale)

        elif source_type == 'irrigation':
            # Irrigation sources: prefer river distance with configurable thresholds
            irrig_cfg = self.config.graph.get('irrigation', {}) if hasattr(self.config, 'graph') else {}
            threshold_km = irrig_cfg.get('direct_distance_threshold_km', 3.0)
            curve_multiplier = irrig_cfg.get('curve_multiplier', 1.5)
            return _calculate_irrigation_distance(
                receptor_coord, source_coord, euclidean_dist, dem_path=self.dem_path,
                threshold_km=threshold_km, curve_multiplier=curve_multiplier)

    def _calculate_terrain_distance(self, receptor_coord, source_coord, euclidean_dist, slope_penalty):
        """
        Calculate terrain-aware distance for fertilizer and manure sources.
        Applies slope-based penalty to account for terrain difficulty.
        """
        try:
            if not hasattr(self, 'dem_path') or not self.dem_path or not os.path.exists(self.dem_path):
                # Fallback to euclidean distance if DEM not available
                return euclidean_dist

            import rasterio

            with rasterio.open(self.dem_path) as src:
                # Get elevation at receptor and source locations
                receptor_row, receptor_col = src.index(receptor_coord[0], receptor_coord[1])
                source_row, source_col = src.index(source_coord[0], source_coord[1])

                # Read elevation values
                receptor_elev = src.read(1, window=((receptor_row, receptor_row+1), (receptor_col, receptor_col+1)))[0, 0]
                source_elev = src.read(1, window=((source_row, source_row+1), (source_col, source_col+1)))[0, 0]

                # Handle invalid elevation values
                if np.isnan(receptor_elev) or np.isnan(source_elev):
                    return euclidean_dist

                # Calculate elevation difference and slope
                elevation_diff = abs(receptor_elev - source_elev)
                slope_ratio = elevation_diff / (euclidean_dist * 1000)  # Convert km to meters

                # Apply slope penalty: steeper slopes increase effective distance
                terrain_multiplier = 1.0 + (slope_ratio * (slope_penalty - 1.0))
                terrain_distance = euclidean_dist * terrain_multiplier

                return terrain_distance

        except Exception as e:
            logging.warning(f"Error calculating terrain distance: {e}, falling back to euclidean")
            return euclidean_dist

    def _expand_candidate_search(self, receptor_idx, receptor_coord, source_type, sources_df, source_coords,
                               original_max_dist, min_required, scoring_weights):

        # STRICT RULE: This function should NEVER be called for manure sources
        if source_type == 'manure':
            logging.error(f"Receptor {receptor_idx}: CRITICAL ERROR - Expand candidate search called for manure source type. This should never happen!")
            return None

        source_indices_of_type = sources_df[sources_df['source_type'] == source_type].index
        if len(source_indices_of_type) == 0:
            return None

        # Define maximum allowed expansion distances per source type
        # These limits apply to the zero-candidate exception mechanism
        max_expansion_limits = {
            'atmosphere': 5.0,    # Atmosphere: strict 5km limit (DEM distance)
            'irrigation': 3.0,    # Irrigation: strict 3km limit (Euclidean distance)
            'fertilizer': 2.0,    # Fertilizer: strict 2km limit (Euclidean distance)
            'manure': 0.2        # Should never reach here, but keep strict limit
        }
        max_allowed_dist = max_expansion_limits.get(source_type, 5.0)

        all_distances = []
        for s_idx in source_indices_of_type:
            source_coord = source_coords[s_idx]
            # IMPORTANT: Atmosphere sources use DEM climbing distance, others use Euclidean distance
            if source_type == 'atmosphere':
                dist = self._calculate_dem_distance_for_edge(
                    receptor_coord, source_coord, source_type
                )
            else:
                dist = _calculate_distance(receptor_coord[1], receptor_coord[0],
                                         source_coord[1], source_coord[0])

            # Only include sources within the maximum allowed expansion distance
            if dist <= max_allowed_dist:
                all_distances.append((dist, s_idx))

        if not all_distances:
            logging.warning(f"Receptor {receptor_idx} {source_type}: No sources found within maximum expansion distance {max_allowed_dist}km")
            return None

        all_distances.sort(key=lambda x: x[0])
        expanded_indices = [idx for _, idx in all_distances[:min_required]]

        if len(expanded_indices) < min_required:
            expanded_indices = [idx for _, idx in all_distances]

        candidates_df = pd.DataFrame(index=expanded_indices)

        distances = [dist for dist, _ in all_distances[:len(expanded_indices)]]
        max_dist_in_candidates = max(distances) if distances else 1.0
        candidates_df['distance_score'] = [(max_dist_in_candidates - d) / max_dist_in_candidates for d in distances]

        receptor_profiles_contrib_only = torch.relu(torch.tensor(sources_df.iloc[receptor_idx:receptor_idx+1][self.metals].values)).numpy()
        if len(receptor_profiles_contrib_only) == 0:
            candidates_df['chemical_score'] = [0.5] * len(expanded_indices)
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            source_chem_profiles_np = sources_df.loc[expanded_indices, self.metals].values
            chem_scores = cosine_similarity(receptor_profiles_contrib_only, source_chem_profiles_np)[0]
            candidates_df['chemical_score'] = chem_scores

        source_concentration_sum = sources_df.loc[expanded_indices, self.metals].sum(axis=1)
        max_conc = source_concentration_sum.max() if len(source_concentration_sum) > 0 else 1.0
        candidates_df['strength_score'] = source_concentration_sum / (max_conc + 1e-8)

        candidates_df['wind_score'] = 1.0
        candidates_df['hydro_score'] = candidates_df['distance_score']

        total_score = np.zeros(len(candidates_df))
        for score_type, weight in scoring_weights.items():
            if score_type in candidates_df.columns:
                total_score += candidates_df[score_type] * weight

        candidates_df['total_score'] = total_score

        sorted_candidates = candidates_df.sort_values('total_score', ascending=False)

        logging.info(f"Expanded search for {source_type}: found {len(expanded_indices)} candidates within extended range")
        return sorted_candidates

    def _ensure_global_min_candidates(self, receptor_idx, receptor_coord, source_type_edges,
                                    sources_df, source_coords, global_min_candidates, current_total):

        needed_candidates = global_min_candidates - current_total
        if needed_candidates <= 0:
            return source_type_edges

        selected_indices = set()
        for source_type, edge_data in source_type_edges.items():
            selected_indices.update(edge_data['indices'])

        # Get distance limits from config
        max_dist_config = self.config.graph.get('max_distance_km', {})

        all_unselected = []
        for source_type in sources_df['source_type'].unique():
            source_indices_of_type = sources_df[sources_df['source_type'] == source_type].index

            # Get max distance for this source type
            max_dist_for_type = max_dist_config.get(source_type, 5.0)

            for s_idx in source_indices_of_type:
                if s_idx not in selected_indices:
                    source_coord = source_coords[s_idx]
                    dist = self._calculate_dem_distance_for_edge(
                        receptor_coord, source_coord, source_type
                    )

                    # STRICT RULE: manure sources must be within 0.2km, no exceptions
                    if source_type == 'manure' and dist > 0.2:
                        continue  # Skip manure sources beyond 0.2km

                    # For other source types, also respect their distance limits
                    if dist > max_dist_for_type:
                        continue  # Skip sources beyond their type-specific limits

                    score = 1.0 / (1.0 + dist) + np.random.random() * 0.1
                    all_unselected.append((score, source_type, s_idx, dist))

        all_unselected.sort(key=lambda x: x[0], reverse=True)
        additional_candidates = all_unselected[:needed_candidates]

        # Add additional candidates and ensure their score rows exist
        for score, source_type, s_idx, dist in additional_candidates:
            if source_type not in source_type_edges:
                source_type_edges[source_type] = {
                    'indices': [],
                    'scores': pd.DataFrame(columns=['distance_score','chemical_score','strength_score','wind_score','hydro_score']),
                    'effective_k': 0,
                    'total_filtered': 0,
                    'distance_filtered': 0,
                    'final_selected': 0
                }

            # Append index
            source_type_edges[source_type]['indices'].append(s_idx)
            source_type_edges[source_type]['final_selected'] += 1

            # Ensure the scores DataFrame has required columns
            scores_df = source_type_edges[source_type]['scores']
            for col in ['distance_score','chemical_score','strength_score','wind_score','hydro_score']:
                if col not in scores_df.columns:
                    scores_df[col] = np.nan

            # Compute simple, consistent default scores for the added candidate
            distance_score = 1.0 / (1.0 + dist)
            strength_series = sources_df[self.metals].sum(axis=1)
            max_conc = strength_series.max() if len(strength_series) > 0 else 1.0
            strength_score = float(strength_series.loc[s_idx] / (max_conc + 1e-8)) if s_idx in strength_series.index else 0.0
            wind_score = 1.0 if source_type == 'atmosphere' else 0.0
            hydro_score = distance_score  # fallback to distance score

            # Upsert the row into scores df
            scores_df.loc[s_idx, ['distance_score','chemical_score','strength_score','wind_score','hydro_score']] = [
                distance_score, 0.5, strength_score, wind_score, hydro_score
            ]

            # Save back
            source_type_edges[source_type]['scores'] = scores_df

        logging.info(f"Receptor {receptor_idx}: Added {len(additional_candidates)} additional candidates to reach global minimum")
        return source_type_edges

    def _reduce_to_global_max_candidates(self, receptor_idx, source_type_edges, global_max_candidates):
        """
        Reduce total candidates to global maximum by removing lowest-scoring candidates
        """
        # Collect all candidates with their scores
        all_candidates = []
        for source_type, edge_data in source_type_edges.items():
            scores_df = edge_data['scores']
            for idx, row in scores_df.iterrows():
                all_candidates.append({
                    'source_idx': idx,
                    'source_type': source_type,
                    'total_score': row.get('total_score', 0.0)
                })

        # Sort by total score (descending)
        all_candidates.sort(key=lambda x: x['total_score'], reverse=True)

        # Keep only top global_max_candidates
        selected_candidates = all_candidates[:global_max_candidates]

        # Rebuild source_type_edges with selected candidates
        new_source_type_edges = {}
        for source_type in source_type_edges.keys():
            new_source_type_edges[source_type] = {
                'indices': [],
                'scores': pd.DataFrame(),
                'effective_k': 0,
                'total_filtered': source_type_edges[source_type]['total_filtered'],
                'distance_filtered': source_type_edges[source_type]['distance_filtered'],
                'final_selected': 0
            }

        # Add selected candidates back
        for candidate in selected_candidates:
            source_type = candidate['source_type']
            source_idx = candidate['source_idx']

            # Add to indices
            new_source_type_edges[source_type]['indices'].append(source_idx)
            new_source_type_edges[source_type]['final_selected'] += 1

            # Add to scores DataFrame
            original_scores = source_type_edges[source_type]['scores']
            if source_idx in original_scores.index:
                candidate_row = original_scores.loc[[source_idx]]
                if new_source_type_edges[source_type]['scores'].empty:
                    new_source_type_edges[source_type]['scores'] = candidate_row.copy()
                else:
                    new_source_type_edges[source_type]['scores'] = pd.concat([
                        new_source_type_edges[source_type]['scores'],
                        candidate_row
                    ])

        # Update effective_k for each source type
        for source_type in new_source_type_edges.keys():
            new_source_type_edges[source_type]['effective_k'] = new_source_type_edges[source_type]['final_selected']

        logging.info(f"Receptor {receptor_idx}: Reduced candidates to {len(selected_candidates)} (max: {global_max_candidates})")

        return new_source_type_edges