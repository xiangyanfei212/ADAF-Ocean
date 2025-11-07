import os
import glob
import json
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pandas as pd
from icecream import ic


import numpy as np
import json

def z_score_normalize_2d(array, stats_path, channel_names=None):
    """
    Z-score normalization for a 2D array.
    Perform z-score normalization on a 2D array with shape [n_samples, n_features],
    using mean and standard deviation values for each feature provided in a JSON file.

    Args:
        array (np.ndarray): Input array of shape [n_samples, n_features].
        stats_path (str): Path to the JSON file containing mean and standard deviation for each feature.
        channel_names (list, optional): List of feature names to ensure the correct order.

    Returns:
        np.ndarray: The normalized array with the same shape as the input.
    """
    # Load mean and std values from the JSON file
    with open(stats_path, 'r') as file:
        stats_values = json.load(file)

    # Extract mean and std values into arrays based on feature order
    if not channel_names:
        channel_names = list(stats_values.keys())
    mean_vals = np.array([stats_values[feature]["mean"] for feature in channel_names])
    std_vals = np.array([stats_values[feature]["std"] for feature in channel_names])

    # Ensure the input array and mean/std values are compatible
    assert array.shape[1] == len(mean_vals), (
        f"Number of features in array ({array.shape[1]}) must match length of mean_vals ({len(mean_vals)})."
    )

    # Avoid division by zero by replacing zero std values with a very small value
    std_vals[std_vals == 0] = 1e-10  # Handle edge case where std == 0 for a feature

    # Normalize each feature independently
    normalized_array = (array - mean_vals) / std_vals

    # Replace NaN values with 0 (if any)
    if np.isnan(normalized_array).any():
        normalized_array = np.nan_to_num(normalized_array, nan=0.0)

    return normalized_array, channel_names  # Return feature names to verify order


def z_score_normalize(array, stats_path, channel_names=None):
    """
    Z-score normalization
    Perform z-score normalization on a 3D array with shape [n_blocks, n_features, channels],
    using mean and standard deviation values for each channel provided in a JSON file.

    Args:
        array (np.ndarray): Input array of shape [n_blocks, n_features, channels].
        stats_path (str): Path to the JSON file containing mean and standard deviation for each channel.
        channel_names (list, optional): List of channel names to ensure the correct order.

    Returns:
        np.ndarray: The normalized array with the same shape as the input.
    """
    # Load mean and std values from the JSON file
    with open(stats_path, 'r') as file:
        stats_values = json.load(file)

    # Extract mean and std values into arrays based on channel order
    if not channel_names:
        channel_names = list(stats_values.keys())
    mean_vals = np.array([stats_values[channel]["mean"] for channel in channel_names])
    std_vals = np.array([stats_values[channel]["std"] for channel in channel_names])

    # Ensure the input array and mean/std values are compatible
    if array.shape[2] > len(mean_vals):
        mean_vals = np.tile(mean_vals, int(array.shape[2]/len(mean_vals)))
        std_vals = np.tile(std_vals, int(array.shape[2]/len(std_vals)))

    assert array.shape[2] == len(mean_vals), "Number of channels in array must match length of mean_vals and std_vals."

    # Avoid division by zero by replacing zero std values with a very small value
    std_vals[std_vals == 0] = 1e-10  # Handle edge case where std == 0 for a channel

    # Normalize each channel independently
    normalized_array = (array - mean_vals) / std_vals

    # Replace NaN values with 0 (if any)
    if np.isnan(normalized_array).any():
        # print("Warning: NaN values detected in the normalized array. Replacing NaNs with 0.")
        normalized_array = np.nan_to_num(normalized_array, nan=0.0)

    return normalized_array, channel_names  # Return channel names to verify order


def z_score_denormalize(normalized_array, stats_path, channel_names=None):
    """
    Perform the inverse of z-score normalization on a 3D array with shape [n_blocks, n_features, channels],
    using mean and standard deviation values for each channel provided in a JSON file.

    Args:
        normalized_array (np.ndarray): Normalized array of shape [n_blocks, n_features, channels].
        stats_path (str): Path to the JSON file containing mean and standard deviation values for each channel.
        channel_names (list, optional): List of channel names specifying the order of channels.

    Returns:
        np.ndarray: The denormalized array with the same shape as the input.
    """
    # Load mean and std values from the JSON file
    with open(stats_path, 'r') as file:
        z_score_values = json.load(file)

    # Extract mean and std values into arrays based on channel order
    if not channel_names:
        channel_names = list(z_score_values.keys())
    mean_vals = np.array([z_score_values[channel]["mean"] for channel in channel_names])
    std_vals = np.array([z_score_values[channel]["std"] for channel in channel_names])

    # Ensure the input array and mean/std values are compatible
    if normalized_array.shape[2] > len(mean_vals):
        mean_vals = np.tile(mean_vals, int(normalized_array.shape[2]/len(mean_vals)))
        std_vals = np.tile(std_vals, int(normalized_array.shape[2]/len(std_vals)))

    assert normalized_array.shape[2] == len(mean_vals), "Number of channels in array must match length of mean_vals and std_vals."

    # Perform denormalization
    denormalized_array = normalized_array * std_vals + mean_vals

    return denormalized_array


def min_max_normalize_v1(array, stats_path, channel_names=None):
    """
    [0, 1]
    Perform min-max normalization on a 3D array with shape [n_blocks, n_features, channels],
    using min and max values for each channel provided in a JSON file.

    Args:
        array (np.ndarray): Input array of shape [n_blocks, n_features, channels].
        stats_path (str): Path to the JSON file containing min and max values for each channel.

    Returns:
        np.ndarray: The normalized array with the same shape as the input.
    """
    # ic(array.shape)
    # Load min and max values from the JSON file
    with open(stats_path, 'r') as file:
        min_max_values = json.load(file)

    # Extract min and max values into arrays based on channel order
    if not channel_names:
        channel_names = list(min_max_values.keys())
    min_vals = np.array([min_max_values[channel]["min"] for channel in channel_names])
    max_vals = np.array([min_max_values[channel]["max"] for channel in channel_names])
    # ic(len(min_vals), min_vals)

    # Ensure the input array and min/max values are compatible
    if array.shape[2] > len(min_vals):
        min_vals = np.tile(min_vals, int(array.shape[2]/len(min_vals)))
        max_vals = np.tile(max_vals, int(array.shape[2]/len(max_vals)))
    # ic(len(min_vals), min_vals)

    assert array.shape[2] == len(min_vals), "Number of channels in array must match length of min_vals and max_vals."

    # Avoid division by zero by replacing zero ranges with a very small value
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1e-10  # Handle edge case where max == min for a channel

    # Normalize each channel independently
    normalized_array = (array - min_vals) / ranges

    # Clip to ensure values are within [0, 1]
    # normalized_array = np.clip(normalized_array, 0, 1)

    # Replace NaN values with 0
    if np.isnan(normalized_array).any():
        # print("Warning: NaN values detected in the normalized array. Replacing NaNs with 0.")
        normalized_array = np.nan_to_num(normalized_array, nan=0.0)

    # print("channel_names:", channel_names)
    # print("Channel-wise min values after normalization:", normalized_array.min(axis=(0, 1)))
    # print("Channel-wise max values after normalization:", normalized_array.max(axis=(0, 1)))

    return normalized_array, channel_names  # Return channel names to verify order


def min_max_normalize_v2(array, stats_path, channel_names=None):
    """
    [-1, 1]
    Perform min-max normalization on a 3D array with shape [n_blocks, n_features, channels],
    using min and max values for each channel provided in a JSON file.

    Args:
        array (np.ndarray): Input array of shape [n_blocks, n_features, channels].
        stats_path (str): Path to the JSON file containing min and max values for each channel.

    Returns:
        np.ndarray: The normalized array with the same shape as the input.
    """
    array_dim = np.ndim(array)
    # ic(array_dim)

    # Load min and max values from the JSON file
    with open(stats_path, 'r') as file:
        min_max_values = json.load(file)

    # Extract min and max values into arrays based on channel order
    if not channel_names:
        channel_names = list(min_max_values.keys())
    min_vals = np.array([min_max_values[channel]["min"] for channel in channel_names])
    max_vals = np.array([min_max_values[channel]["max"] for channel in channel_names])

    # Ensure the number of channels matches the length of min_vals
    num_channels = array.shape[-1]
    if num_channels > len(min_vals):
        min_vals = np.tile(min_vals, int(num_channels / len(min_vals)))
        max_vals = np.tile(max_vals, int(num_channels / len(max_vals)))

    assert num_channels == len(min_vals), f"Number of channels ({num_channels}) in array must match length of min_vals and max_vals ({len(min_vals)})."

    # Avoid division by zero by replacing zero ranges with a very small value
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1e-10  # Handle edge case where max == min for a channel

    # Normalize each channel independently
    # Broadcasting min_vals and ranges to match the shape of the array
    if array_dim == 5:
        min_vals = min_vals.reshape(1, 1, 1, 1, -1)  # Shape: [1, 1, 1, 1, channels]
        ranges = ranges.reshape(1, 1, 1, 1, -1)      # Shape: [1, 1, 1, 1, channels]
    if array_dim == 4:
        min_vals = min_vals.reshape(1, 1, 1, -1)  # Shape: [1, 1, 1, 1, channels]
        ranges = ranges.reshape(1, 1, 1, -1)      # Shape: [1, 1, 1, 1, channels]
    normalized_array = (array - min_vals) / ranges * 2 - 1

    # Clip to ensure values are within [-1, 1]
    # normalized_array = np.clip(normalized_array, -1, 1)

    # Replace NaN values with 0
    if np.isnan(normalized_array).any():
        normalized_array = np.nan_to_num(normalized_array, nan=0.0)

    return normalized_array, channel_names  # Return channel names to verify order


def location_normalize(array, depth_max=643):
    """
    Normalize geographic information (latitude, longitude, depth) from the input array.
    
    The input array contains latitude, longitude, and depth for each point. 
    After normalization, the output array contains the following features for each point:
        1. Sine of latitude (sin(lat))
        2. Cosine of latitude (cos(lat))
        3. Sine of longitude (sin(lon))
        4. Cosine of longitude (cos(lon))
        5. Normalized depth (scaled to the range [-1, 1])

    Perform min-max normalization on a 3D array with shape [n_blocks, n_features, channels],
    using min and max values for each channel provided in a JSON file.

    Args:
        array (np.ndarray): Input 3D NumPy array of shape (N, M, 3), where:
                            - `array[:,:,0]` represents latitude (in degrees)
                            - `array[:,:,1]` represents longitude (in degrees)
                            - `array[:,:,2]` represents depth (assumed to have a maximum value of 643).
        depth_max (int): the maximum of depth

    Returns:
        np.ndarray: A 3D NumPy array of shape (N, M, 5), where each point contains the normalized features.
    """
    # ic(array.shape)

    lat = array[:,:,0]
    lon = array[:,:,1]

    lon_sin = np.sin(np.radians(lon))
    lon_cos = np.cos(np.radians(lon))
    lat_sin = np.sin(np.radians(lat))
    lat_cos = np.cos(np.radians(lat))
    
    normalized_depth = array[:,:,2] / depth_max * 2 - 1

    normalized_array = np.stack((lat_sin, lat_cos, lon_sin, lon_cos, normalized_depth), axis=2)

    # Replace NaN values with 0
    if np.isnan(normalized_array).any():
        # print("Warning: NaN values detected in the normalized array. Replacing NaNs with 0.")
        normalized_array = np.nan_to_num(normalized_array, nan=0.0)

    # print("channel_names:", channel_names)
    # print("Channel-wise min values after normalization:", normalized_array.min(axis=(0, 1)))
    # print("Channel-wise max values after normalization:", normalized_array.max(axis=(0, 1)))

    return normalized_array

def min_max_denormalize_v1(normalized_array, stats_path, channel_names=None):
    """
    Perform min-max denormalization on a 4D array with shape [lat, lon, depth, variables],
    using min and max values for each channel provided in a JSON file.

    Args:
        normalized_array (np.ndarray): Input normalized array of shape [lat, lon, depth, variables].
        stats_path (str): Path to the JSON file containing min and max values for each variable.
        channel_names (list, optional): List of channel names, to specify the order of variables.

    Returns:
        np.ndarray: The denormalized array with the same shape as the input.
    """
    # Load min and max values from the JSON file
    with open(stats_path, 'r') as file:
        min_max_values = json.load(file)

    # Extract min and max values into arrays based on channel order
    if channel_names is None:
        channel_names = list(min_max_values.keys())
    min_vals = np.array([min_max_values[channel]["min"] for channel in channel_names])
    max_vals = np.array([min_max_values[channel]["max"] for channel in channel_names])

    # Handle cases where input has more channels than min/max values
    if normalized_array.shape[2] > len(min_vals):
        min_vals = np.tile(min_vals, int(normalized_array.shape[3] / len(min_vals)))
        max_vals = np.tile(max_vals, int(normalized_array.shape[3] / len(max_vals)))

    # Ensure compatibility between array and min/max values
    assert normalized_array.shape[2] == len(min_vals), \
        "Number of variables in array must match the length of min_vals and max_vals."

    # Compute the original ranges
    ranges = max_vals - min_vals

    # Perform denormalization
    denormalized_array = normalized_array * ranges + min_vals

    return denormalized_array


def min_max_denormalize_v2(normalized_array, stats_path, channel_names=None):
    """
    Perform the inverse of min-max normalization on a 3D array with shape [n_blocks, n_features, channels],
    using min and max values for each channel provided in a JSON file.

    Args:
        normalized_array (np.ndarray): Normalized array of shape [n_blocks, n_features, channels].
        stats_path (str): Path to the JSON file containing min and max values for each channel.
        channel_names (list, optional): List of channel names specifying the order of channels.

    Returns:
        np.ndarray: The denormalized array with the same shape as the input.
    """
    # Load min and max values from the JSON file
    with open(stats_path, 'r') as file:
        min_max_values = json.load(file)

    # Extract min and max values into arrays based on channel order
    if not channel_names:
        channel_names = list(min_max_values.keys())
    min_vals = np.array([min_max_values[channel]["min"] for channel in channel_names])
    max_vals = np.array([min_max_values[channel]["max"] for channel in channel_names])

    # Ensure the input array and min/max values are compatible
    if normalized_array.shape[2] > len(min_vals):
        min_vals = np.tile(min_vals, int(normalized_array.shape[2]/len(min_vals)))
        max_vals = np.tile(max_vals, int(normalized_array.shape[2]/len(max_vals)))

    assert normalized_array.shape[2] == len(min_vals), "Number of channels in array must match length of min_vals and max_vals."

    # Calculate ranges and perform denormalization
    ranges = max_vals - min_vals
    denormalized_array = (normalized_array + 1) / 2 * ranges + min_vals

    return denormalized_array



def normalize(data, method, stats_file, vars):
    # print(f'Read stats_file from {stats_file}')
    if method == 'minmax_v1':
        data, _ = min_max_normalize_v1(data, stats_file, vars)
    elif method == 'minmax_v2':
        data, _ = min_max_normalize_v2(data, stats_file, vars)
    elif method == 'z_score':
        data, _ = z_score_normalize(data, stats_file, vars)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    return data 


def denormalize(data, method, stats_file, vars):
    # print(f'Read stats_file from {stats_file}')
    if method == 'minmax_v1':
        data = min_max_denormalize_v1(data, stats_file, vars)
    elif method == 'minmax_v2':
        data = min_max_denormalize_v2(data, stats_file, vars)
    elif method == 'z_score':
        data = z_score_denormalize(data, stats_file, vars)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    return data  


def normalize_coordinates(coords, method, max_depth=643, stats_file=None, variables=None):
    """
    Normalize coordinates based on the specified method.

    Args:
        coords (np.ndarray): Coordinates array to normalize.
        method (str): Normalization method, e.g., 'cycle', 'minmax_v1', 'minmax_v2'.
        stats_file (str, optional): Path to the stats file for min-max normalization.
        variables (list, optional): List of variable names to normalize.
        max_depth (int, optional): The maximum of depth

    Returns:
        np.ndarray: Normalized coordinates.
    """
    if method == 'cycle':
        return location_normalize(coords, max_depth)
    elif method == 'minmax_v1':
        coords, _ = min_max_normalize_v1(coords, stats_file, variables)
    elif method == 'minmax_v2':
        coords, _ = min_max_normalize_v2(coords, stats_file, variables)
    else:
        raise ValueError(f"Normalization method '{method}' not implemented!")
    
    return coords


if __name__ == '__main__':

    from read import *
    from block import *
    from plot import *

    # %% Test cycle-location normalization 
    latitudes = np.arange(-90, 91, 1)  # -90 to 90 inclusive, step 1
    longitudes = np.arange(0, 361, 1)  # 0 to 360 inclusive, step 1

    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing='ij')
    depth_grid = np.zeros_like(lat_grid)

    # Combine latitude, longitude, and depth into a single 3D array
    # Shape: (latitude_count, longitude_count, 3)
    test_array = np.stack((lat_grid, lon_grid, depth_grid), axis=2)

    # Call the location_normalize function
    normalized_array = location_normalize(test_array)
    ic(normalized_array.shape)

    
    plot_2d_map_multi_sub_figure(
        [normalized_array[:,:,0], normalized_array[:,:,1], normalized_array[:,:,2], normalized_array[:,:,3]],
        longitudes,
        latitudes,
        ['lat_sin', 'lat_cos', 'lon_sin', 'lon_cos'],
        'coolwarm',
        2,
        2,
        global_vmin=None,
        global_vmax=None,
        shared_colorbar=True,
        colorbar_label='',
        save_path='./imgs/location_cycle_norm.png'
    )

    exit()
    
    # lon = np.array([0, 90, 180, 270, 360])  # 经度范围: 0~360
    # lat = np.array([-90, -45, 0, 45, 90])   # 纬度范围: -90~90

    # 正弦和余弦编码
    # lon_sin = np.sin(np.radians(lon))  # 经度的正弦
    # lon_cos = np.cos(np.radians(lon))  # 经度的余弦

    # lat_sin = np.sin(np.radians(lat))  # 纬度的正弦
    # lat_cos = np.cos(np.radians(lat))  # 纬度的余弦

    # 输出结果
    # print("Longitude Sin:", np.round(lon_sin,3))
    # print("Longitude Cos:", np.round(lon_cos,3))

    # print("Latitude Sin:", np.round(lat_sin, 3))
    # print("Latitude Cos:", np.round(lat_cos, 3))

    # Check normalization
    # %% Read target (181, 360, 23, 5)
    target_file_path = '/groups/hxm_group/home/share/xiangyanfei/OCEAN_DATA/glorys1p_23layers/2020/20200101.h5' 
    target, lats, lons, depths, target_time = read_glorys_file(target_file_path)
    # ic(target_time, target_file_path)

    target_block_shape = [6, 6, 8]

    lat_step=3
    lon_step=3
    depth_step=3
    target_stats_file='/groups/hxm_group/home/share/xiangyanfei/02_DA_ocean/stats/glorys.json'
    target_vars=['T', 'S', 'U', 'V', 'SSH']

    target_blocks, sampled_coords, target_block_coords = slice_array(
        target,
        lats, lons, depths,
        target_block_shape[0],
        target_block_shape[1],
        target_block_shape[2],
        lat_step, 
        lon_step,
        depth_step)
    ic(np.nanmin(target_blocks), np.nanmax(target_blocks))

    n_blocks, lat_size, lon_size, depth_size, channel_num = target_blocks.shape
    target_blocks = target_blocks.reshape(n_blocks, lat_size * lon_size * depth_size, channel_num)
    target_blocks_norm, _ = min_max_normalize_v2(
        target_blocks, target_stats_file, target_vars)
    ic(np.min(target_blocks_norm), np.max(target_blocks_norm))

    target_blocks_unnorm = min_max_denormalize_v2(
        target_blocks_norm, target_stats_file, target_vars)
    ic(np.min(target_blocks_unnorm), np.max(target_blocks_unnorm))

    bias = target_blocks - target_blocks_unnorm
    ic(np.nanmin(bias), np.nanmax(bias))

