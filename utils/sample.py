import os
import h5py
import xarray as xr
import numpy as np
import pandas as pd
from icecream import ic
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def get_day_of_year(date_str):
    """
    Get the year and day of year in the format YYYYDDD from a date string.

    Parameters:
        date_str (str): Date string in the format 'YYYYMMDD'.

    Returns:
        str: A string in the format 'YYYYDDD', where DDD is the day of the year.
    """
    # Parse the date string into a datetime object
    date = datetime.strptime(date_str, "%Y%m%d")

    # Extract the year and calculate the day of the year
    year = date.year
    day_of_year = date.timetuple().tm_yday

    # Format the result as 'YYYYDDD'
    result = f"{year}{day_of_year:03d}"  # Pad day_of_year with zeros to make it 3 digits

    return result

def find_nearest_index(array, value):
    """
    Finds the nearest index in a 1D array for a given value.

    Args:
        array (np.ndarray): 1D array of sorted values (e.g., latitudes, longitudes, pressures).
        value (float): The value to find the nearest index for.

    Returns:
        int: The index of the nearest value in the array.
    """
    index = np.abs(array - value).argmin()
    return index

def sample_points_in_2D_area(shape, n, patch_shape):
    """
    Randomly samples `n` points from a 2D array with surrounding space constraints.

    Args:
        shape (tuple): The shape of the 2D array (latitude, longitude).
        n (int): The number of points to sample.
        patch_shape (tuple): The required patch_shape around each point (lat_grid, lon_grid).

    Returns:
        list: A list of tuples representing the sampled points (lat_idx, lon_idx).
    """

    lat_size, lon_size = shape
    lat_pad, lon_pad = patch_shape

    # Define the valid range for sampling (avoiding boundaries)
    lat_min, lat_max = lat_pad, lat_size - lat_pad
    lon_min, lon_max = lon_pad, lon_size - lon_pad

    # Ensure the valid range is large enough for sampling
    if (lat_max - lat_min) <= 0 or (lon_max - lon_min) <= 0:
        raise ValueError("patch_shape is too large for the array dimensions.")

    # Generate random indices within the valid range
    lat_indices = np.random.randint(lat_min, lat_max, size=n)
    lon_indices = np.random.randint(lon_min, lon_max, size=n)

    # Combine the indices into a list of points
    sampled_points_indices = list(zip(lat_indices, lon_indices))

    return sampled_points_indices

def sample_points_with_block_shape(shape, n, block_shape):
    """
    Randomly samples `n` points from a 3D array with surrounding space constraints.

    Args:
        shape (tuple): The shape of the 3D array (latitude, longitude, levels).
        n (int): The number of points to sample.
        block_shape (tuple): The required block_shape around each point (lat_grid, lon_grid, level_grid).

    Returns:
        list: A list of tuples representing the sampled points (lat_idx, lon_idx, depth_idx).
    """

    lat_size, lon_size, depth_size = shape
    lat_pad, lon_pad, depth_pad = block_shape

    # Define the valid range for sampling (avoiding boundaries)
    lat_min, lat_max = lat_pad, lat_size - lat_pad
    lon_min, lon_max = lon_pad, lon_size - lon_pad
    depth_min, depth_max = depth_pad, depth_size - depth_pad

    # Ensure the valid range is large enough for sampling
    if (lat_max - lat_min) <= 0 or (lon_max - lon_min) <= 0 or (depth_max - depth_min) <= 0:
        raise ValueError("block_shape is too large for the array dimensions.")

    # Generate random indices within the valid range
    lat_indices = np.random.randint(lat_min, lat_max, size=n)
    lon_indices = np.random.randint(lon_min, lon_max, size=n)
    depth_indices = np.random.randint(depth_min, depth_max, size=n)

    # Combine the indices into a list of points
    sampled_points_indices = list(zip(lat_indices, lon_indices, depth_indices))

    return sampled_points_indices


def map_indices_to_coordinates_2D(latitudes, longitudes, sampled_points):
    """
    Maps sampled indices to their corresponding latitude and longitude values.

    Args:
        latitudes (list or np.ndarray): List of latitude values.
        longitudes (list or np.ndarray): List of longitude values.
        sampled_points (list): List of sampled points as tuples (lat_idx, lon_idx).

    Returns:
        list: A list of tuples containing (latitude, longitude).
    """
    coordinates = []
    for lat_idx, lon_idx in sampled_points:
        lat = latitudes[lat_idx]  # Map latitude index to actual latitude value
        lon = longitudes[lon_idx]  # Map longitude index to actual longitude value
        coordinates.append((lat, lon))  # Add the depth index
    return coordinates


def map_indices_to_coordinates_3D(latitudes, longitudes, depths, sampled_points):
    """
    Maps sampled indices to their corresponding latitude and longitude values.

    Args:
        latitudes (list or np.ndarray): List of latitude values.
        longitudes (list or np.ndarray): List of longitude values.
        depths (list or np.ndarray): List of pressure values. 
        sampled_points (list): List of sampled points as tuples (lat_idx, lon_idx, depth_idx).

    Returns:
        list: A list of tuples containing (latitude, longitude, pressure_index).
    """
    coordinates = []
    for lat_idx, lon_idx, depth_idx in sampled_points:
        lat = latitudes[lat_idx]  # Map latitude index to actual latitude value
        lon = longitudes[lon_idx]  # Map longitude index to actual longitude value
        depth = depths[depth_idx]
        coordinates.append((lat, lon, depth))  # Add the depth index
    return coordinates

def filter_in_situ_by_coords_and_ranges_v1(
        file_path,
        sampled_coords,
        ranges,
        pad = 256,
        variables = ['temp', 'sal'],
    ):
    
    # Check if file_path is empty
    if not file_path:
        # Shape: (N_points, pad, len(variables)) for values
        values_shape = (len(coords), pad, len(variables))
        # Shape: (N_points, pad, 3) for coords
        coords_shape = (len(coords), pad, 3)
        return np.zeros(values_shape), np.zeros(coords_shape)
   
    # Load the CSV file into a DataFrame
    # print(f'Read in situ obs from {file_path}')
    data = pd.read_csv(file_path) # , index_col=0)
    # print(data)
    # data['lon'] = (data['lon'] + 360) % 360 # convert lon
    # print(f'lon: {data['lon']}')

    # Extract the grid sizes
    lat_range, lon_range, depth_range = ranges
    # print(f'lat_range: {lat_range}, lon_range: {lon_range}, depth_range: {depth_range}')

    # Initialize a dictionary to store results for each point
    values = []
    coords = []

    # Iterate over the coordinate points
    num_value_in_block = []
    for lat, lon, depth in sampled_coords:
        # ic(lat, lon, depth)
        # print(f'lat={lat}, lon={lon}, depth={depth}')
        # Define the range for each coordinate point
        lat_min, lat_max = lat - lat_range / 2, lat + lat_range / 2
        lon_min, lon_max = lon - lon_range / 2, lon + lon_range / 2
        depth_min, depth_max = depth - depth_range / 2, depth + depth_range / 2
        # print(f'lat: {lat_min}~{lat_max}')
        # print(f'lon: {lon_min}~{lon_max}')
        # print(f'depth: {depth_min}~{depth_max}')

        # Filter the data for the current point
        current_filtered = data[
            (data["lat"] >= lat_min) & (data["lat"] <= lat_max) &
            (data["lon"] >= lon_min) & (data["lon"] <= lon_max) &
            (data["depth"] >= depth_min) & (data["depth"] <= depth_max)
        ]

        # QC
        current_filtered = current_filtered[
            (current_filtered['temp'] >= -3) & (current_filtered['temp'] <= 36)
        ]
        current_filtered = current_filtered[
            (current_filtered['sal'] >= 0) & (current_filtered['sal'] <= 40)
        ]

        current_value = np.array(current_filtered[variables].values)
        num_value = current_value.shape[0]
        num_value_in_block.append(num_value)
        # ic(current_value.shape)
        # Pad the current_value array to have pad rows and 4 columns
        if current_value.shape[0] < pad:  # If the number of rows is less than pad
            rows_to_pad = pad - current_value.shape[0]
            current_value = np.pad(
                current_value, ((0, rows_to_pad), (0, 0)), mode="constant", constant_values=0)
        elif current_value.shape[0] > pad:  # If there are more than pad rows, truncate it
            current_value = current_value[:pad, :]
        values.append(current_value)  # Append the padded array to the results list
        num_value_pad = current_value.shape[0]
        # if num_value > 0:
        #     print(f'pad value: {num_value}->{num_value_pad}')
        # else:
        #     print('There is no observations')

        # Extract the corresponding coordinates: lat, lon, pressure
        current_coords = np.array(current_filtered[["lat", "lon", "depth"]].values)
        # Pad the current_coords array to have pad rows and 3 columns
        if current_coords.shape[0] < pad:  # If the number of rows is less than pad
            rows_to_pad = pad - current_coords.shape[0]
            current_coords = np.pad(
                current_coords, ((0, rows_to_pad), (0, 0)), mode="constant", constant_values=0)
        elif current_coords.shape[0] > pad:  # If there are more than pad rows, truncate it
            current_coords = current_coords[:pad, :]
        coords.append(current_coords)  # Append the padded array to the results list
        # ic(current_coords.shape)

    # return values, coords
    return np.stack(values, axis=0), np.stack(coords, axis=0), num_value_in_block


def filter_in_situ_by_coords_and_ranges_v2(
        file_path,
        sampled_coords,
        ranges, # units: degree
        pad = 256,
        variables = ['temp', 'sal'],
        output_file=None
    ):
    
   
    # Check if file_path is empty
    if not file_path:
        # Shape: (N_points, pad, len(variables)) for values
        values_shape = (len(coords), pad, len(variables))
        # Shape: (N_points, pad, 3) for coords
        coords_shape = (len(coords), pad, 3)
        return np.zeros(values_shape), np.zeros(coords_shape)
   
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path, index_col=0)
    data['lon'] = (data['lon'] + 360) % 360
    # ic(data)

    lat_range, lon_range = ranges
    # ic(lat_range, lon_range, depth_range)

    # Initialize a dictionary to store results for each point
    values = []
    coords = []

    # Iterate over the coordinate points
    for lat, lon in sampled_coords:
        # ic(lat, lon)
        # Define the range for each coordinate point
        lat_min, lat_max = lat - lat_range / 2, lat + lat_range / 2
        lon_min, lon_max = lon - lon_range / 2, lon + lon_range / 2

        # Filter the data for the current point
        current_filtered = data[
            (data["lat"] >= lat_min) & (data["lat"] <= lat_max) &
            (data["lon"] >= lon_min) & (data["lon"] <= lon_max) &
            (data["depth"] <= 700)
        ]

        # QC
        current_filtered = current_filtered[
            (current_filtered['temp'] >= -3) & (current_filtered['temp'] <= 36)
        ]
        current_filtered = current_filtered[
            (current_filtered['sal'] >= 0) & (current_filtered['sal'] <= 40)
        ]

        current_value = np.array(current_filtered[variables].values)
        # ic(current_value.shape)
        # Pad the current_value array to have pad rows and 4 columns
        if current_value.shape[0] < pad:  # If the number of rows is less than pad
            rows_to_pad = pad - current_value.shape[0]
            current_value = np.pad(current_value, ((0, rows_to_pad), (0, 0)), mode="constant", constant_values=0)
        elif current_value.shape[0] > pad:  # If there are more than pad rows, truncate it
            current_value = current_value[:pad, :]
        values.append(current_value)  # Append the padded array to the results list
        # ic(current_value.shape)

        # Extract the corresponding coordinates: lat, lon, pressure
        current_coords = np.array(current_filtered[["lat", "lon", "depth"]].values)
        # ic(current_coords.shape)  # Debugging: Print the shape of the filtered coordinates array

        # Pad the current_coords array to have pad rows and 3 columns
        if current_coords.shape[0] < pad:  # If the number of rows is less than pad
            rows_to_pad = pad - current_coords.shape[0]
            current_coords = np.pad(current_coords, ((0, rows_to_pad), (0, 0)), mode="constant", constant_values=0)
        elif current_coords.shape[0] > pad:  # If there are more than pad rows, truncate it
            current_coords = current_coords[:pad, :]
        coords.append(current_coords)  # Append the padded array to the results list
        # ic(current_coords.shape)

    # return values, coords
    return np.stack(values, axis=0), np.stack(coords, axis=0)

def filter_in_situ_by_coords_and_ranges_v3(
        file_path,
        sampled_coords,
        ranges,
        pad = 256,
        variables = ['temp', 'sal'],
        std_variables = ['temp_clima_std', 'sal_clima_std'],
    ):
    
    # Check if file_path is empty
    if not file_path:
        # Shape: (N_points, pad, len(variables)) for values
        values_shape = (len(coords), pad, len(variables))
        # Shape: (N_points, pad, 3) for coords
        coords_shape = (len(coords), pad, 3)
        return np.zeros(values_shape), np.zeros(coords_shape)
   
    # Load the CSV file into a DataFrame
    # print(f'Read in situ obs from {file_path}')
    data = pd.read_csv(file_path) # , index_col=0)
    # print(data)
    # data['lon'] = (data['lon'] + 360) % 360 # convert lon
    # print(f'lon: {data['lon']}')

    # Extract the grid sizes
    lat_range, lon_range, depth_range = ranges
    # print(f'lat_range: {lat_range}, lon_range: {lon_range}, depth_range: {depth_range}')

    # Initialize a dictionary to store results for each point
    values = []
    coords = []
    clima_std = []

    # Iterate over the coordinate points
    num_value_in_block = []
    for lat, lon, depth in sampled_coords:
        # ic(lat, lon, depth)
        # print(f'lat={lat}, lon={lon}, depth={depth}')
        # Define the range for each coordinate point
        lat_min, lat_max = lat - lat_range / 2, lat + lat_range / 2
        lon_min, lon_max = lon - lon_range / 2, lon + lon_range / 2
        depth_min, depth_max = depth - depth_range / 2, depth + depth_range / 2
        # print(f'lat: {lat_min}~{lat_max}')
        # print(f'lon: {lon_min}~{lon_max}')
        # print(f'depth: {depth_min}~{depth_max}')

        # Filter the data for the current point
        current_filtered = data[
            (data["lat"] >= lat_min) & (data["lat"] <= lat_max) &
            (data["lon"] >= lon_min) & (data["lon"] <= lon_max) &
            (data["depth"] >= depth_min) & (data["depth"] <= depth_max)
        ]

        # QC
        current_filtered = current_filtered[
            (current_filtered['temp'] >= -3) & (current_filtered['temp'] <= 36)
        ]
        current_filtered = current_filtered[
            (current_filtered['sal'] >= 0) & (current_filtered['sal'] <= 40)
        ]

        current_value = np.array(current_filtered[variables].values)
        num_value = current_value.shape[0]
        num_value_in_block.append(num_value)
        # ic(current_value.shape)
        # Pad the current_value array to have pad rows and 4 columns
        if current_value.shape[0] < pad:  # If the number of rows is less than pad
            rows_to_pad = pad - current_value.shape[0]
            current_value = np.pad(
                current_value, ((0, rows_to_pad), (0, 0)), mode="constant", constant_values=0)
        elif current_value.shape[0] > pad:  # If there are more than pad rows, truncate it
            current_value = current_value[:pad, :]
        values.append(current_value)  # Append the padded array to the results list
        num_value_pad = current_value.shape[0]
        # if num_value > 0:
        #     print(f'pad value: {num_value}->{num_value_pad}')
        # else:
        #     print('There is no observations')

        # Extract the corresponding coordinates: lat, lon, pressure
        current_coords = np.array(current_filtered[["lat", "lon", "depth"]].values)
        # Pad the current_coords array to have pad rows and 3 columns
        if current_coords.shape[0] < pad:  # If the number of rows is less than pad
            rows_to_pad = pad - current_coords.shape[0]
            current_coords = np.pad(
                current_coords, ((0, rows_to_pad), (0, 0)), mode="constant", constant_values=0)
        elif current_coords.shape[0] > pad:  # If there are more than pad rows, truncate it
            current_coords = current_coords[:pad, :]
        coords.append(current_coords)  # Append the padded array to the results list
        # ic(current_coords.shape)

        # Extract the clima std: 
        current_std = np.array(current_filtered[std_variables].values)
        if current_std.shape[0] < pad:
            rows_to_pad = pad - current_std.shape[0]
            current_std = np.pad(
                current_std, ((0, rows_to_pad), (0, 0)), mode="constant", constant_values=0)
        elif current_std.shape[0] > pad:
            current_std = current_std[:pad, :]
        clima_std.append(current_std)

    # return values, coords
    return np.stack(values, axis=0), np.stack(coords, axis=0), np.stack(clima_std, axis=0), num_value_in_block


def extract_blocks_from_coordinates_v1(
        array, 
        latitudes,
        longitudes,
        depths, 
        sampled_coords,
        block_shape):
    """
    Extracts blocks of a given shape (units: grid) from a 4D array based on geographic coordinates.

    Args:
        array (np.ndarray): The 4D array of shape [lat, lon, depth, channel].
        latitudes (np.ndarray): 1D array of latitude values 
        longitudes (np.ndarray): 1D array of longitude values
        depths (np.ndarray): 1D array of depth values
        sampled_coords (list): List of sampled coordinates as tuples (lat, lon, press).
        block_shape (tuple): The shape of the block to extract (block_lat, block_lon). (units: grid)
    """

    (lat_num, lon_num, depth_num, channel_num) = array.shape
    # ic(channel_num, press_num, lat_num, lon_num)

    # Validate inputs
    if len(array.shape) != 4:
        raise ValueError("Input array must have 4 dimensions.")

    # Get block shape and calculate half sizes
    block_lat, block_lon, block_depth = block_shape
    lat_half, lon_half = block_lat // 2, block_lon // 2

    # Prepare output list for blocks
    blocks = []
    block_coordinates = []

    # Iterate through sampled coordinates
    for lat, lon, depth in sampled_coords:
        # Find the nearest indices for the given coordinate
        lat_idx = find_nearest_index(latitudes, lat)
        lon_idx = find_nearest_index(longitudes, lon)
        # ic(lat_idx, lon_idx, press_idx)

        # Calculate the slicing ranges
        lat_start, lat_end = lat_idx - lat_half, lat_idx + lat_half
        lon_start, lon_end = lon_idx - lon_half, lon_idx + lon_half
        # ic(lat_start, lat_end, lon_start, lon_end, press_start, press_end)

        # Check if the block is within the valid range of the array
        if (lat_start >= 0 and lat_end <= lat_num and
            lon_start >= 0 and lon_end <= lon_num):
            
            # Extract the block
            block = array[lat_start:lat_end, lon_start:lon_end, :, :]
            # ic(block.shape)
            blocks.append(block)

            # Generate the full coordinate array for the block
            block_latitudes = latitudes[lat_start:lat_end]
            block_longitudes = longitudes[lon_start:lon_end]

            # Create a 3D meshgrid of coordinates for the block
            lat_grid, lon_grid, depth_grid = np.meshgrid(block_latitudes, block_longitudes, depths, indexing="ij")
            block_coords = np.stack((lat_grid, lon_grid, depth_grid), axis=-1)
            # ic(block_coords.shape)
            block_coordinates.append(block_coords)

    # Stack all blocks into a single NumPy array
    return np.stack(blocks, axis=0), np.stack(block_coordinates, axis=0)


def extract_blocks_from_coordinates_v2(
        array,
        latitudes,
        longitudes,
        depths,
        sampled_coords,
        block_shape):
    """
    Extracts blocks of a given shape (units: degree) from a 4D array based on geographic coordinates.

    Args:
        array (np.ndarray): The 4D array of shape [lat, lon, depth, channel].
        latitudes (np.ndarray): 1D array of latitude values 
        longitudes (np.ndarray): 1D array of longitude values
        depths (np.ndarray): 1D array of depth values
        sampled_coords (list): List of sampled coordinates as tuples (lat, lon, press).
        block_shape (tuple): The shape of the block to extract (block_lat, block_lon). (units: degree)
    """

    (lat_num, lon_num, depth_num, channel_num) = array.shape
    # ic(channel_num, press_num, lat_num, lon_num)

    # Validate inputs
    if len(array.shape) != 4:
        raise ValueError("Input array must have 4 dimensions.")

    # Get block shape and calculate half sizes
    block_lat, block_lon, block_depth = block_shape
    lat_half, lon_half, depth_half = block_lat // 2, block_lon // 2, block_depth // 2

    # Prepare output list for blocks
    blocks = []
    block_coordinates = []

    # Iterate through sampled coordinates
    for lat, lon, depth in sampled_coords:
        # print(f'lat={lat}, lon={lon}, depth={depth}')
        
        # Calculate the slicing ranges
        lat_start, lat_end = lat - lat_half, lat + lat_half
        lon_start, lon_end = lon - lon_half, lon + lon_half
        lat_start_idx = find_nearest_index(latitudes, lat_start)
        lat_end_idx = find_nearest_index(latitudes, lat_end)
        lon_start_idx = find_nearest_index(longitudes, lon_start)
        lon_end_idx = find_nearest_index(longitudes, lon_end)
        # print(f'lat_range: {lat_start}~{lat_end}')
        # print(f'lat_idx_range: {lat_start_idx}~{lat_end_idx}')
        # print(f'lon_range: {lon_start}~{lon_end}')
        # print(f'lon_idx_range: {lon_start_idx}~{lon_end_idx}')

        depth_idx = find_nearest_index(depths, depth)
        depth_start_idx, depth_end_idx = depth_idx - depth_half, depth_idx + depth_half
        
        if lat_start_idx < 0: # shift to north
            # print(f'update lat range from {lat_start_idx}~{lat_end_idx}')
            lat_end_idx = lat_end_idx - lat_start_idx
            lat_start_idx = 0
            # print(f'to {lat_start_idx}~{lat_end_idx}')
        if lat_end_idx == lat_num-1:
            # print(f'update lat range from {lat_start_idx}~{lat_end_idx}')
            lat_end_idx = lat_num
            # print(f'to {lat_start_idx}~{lat_end_idx}')
        if lon_end_idx == lon_num-1:
            # print(f'update lon range from {lon_start_idx}~{lon_end_idx}')
            lon_end_idx = lon_num
            # print(f'to {lon_start_idx}~{lon_end_idx}')

        if lon_start_idx < 0: # cycle 
            # print(f'cycle longitude from {lon_start_idx}~{lon_end_idx}')
            block1 = array[lat_start_idx:lat_end_idx, lon_start_idx:, :, :]
            block2 = array[lat_start_idx:lat_end_idx, 0:lon_end_idx, :, :]
            block = np.concatenate((block1, block2), axis=0)
            block_latitudes = latitudes[lat_start_idx:lat_end_idx]
            block_longitudes1 = longitudes[lon_start_idx:]
            block_longitudes2 = longitudes[0:lon_end_idx]
            block_longitudes = np.concatenate((block_longitudes1, block_longitudes2), axis=0)

        if (lat_start_idx >= 0 and lat_end_idx <= lat_num and
            lon_start_idx >= 0 and lon_end_idx <= lon_num):

            # extract the block

            if depth_num == 1:
                block = array[
                    lat_start_idx:lat_end_idx, 
                    lon_start_idx:lon_end_idx, 
                    :, :]
                block_depths = depths
            else:
                block = array[
                    lat_start_idx:lat_end_idx, 
                    lon_start_idx:lon_end_idx, 
                    depth_start_idx:depth_end_idx,
                    :]
                block_depths = depths[depth_start_idx:depth_end_idx]
            # ic(block.shape)
            
            # Generate the full coordinate array for the block
            block_latitudes = latitudes[lat_start_idx:lat_end_idx]
            block_longitudes = longitudes[lon_start_idx:lon_end_idx]

        # if list(block.shape[:-1]) != block_shape:
        # if block.shape[:-1] != block_shape:
        #     print(f'extract block: {block.shape}/{block_shape}')
        #     print(f'lat={lat}, lon={lon}, depth={depth}')
        #     print(f'lat_start_idx: {lat_start_idx}')
        #     print(f'lat_end_idx: {lat_end_idx}')
        #     print(f'lon_start_idx: {lon_start_idx}')
        #     print(f'lon_end_idx: {lon_end_idx}')
        #     print(f'depth_start_idx: {depth_start_idx}')
        #     print(f'depth_end_idx: {depth_end_idx}')
        #     print(f'lat_num: {lat_num}')
        #     print(f'lon_num: {lon_num}')
        #     print(f'depth_num: {depth_num}')
        #     exit()
            
        blocks.append(block)

        # Create a 3D meshgrid of coordinates for the block
        lat_grid, lon_grid, depth_grid = np.meshgrid(block_latitudes, block_longitudes, block_depths, indexing="ij")
        block_coords = np.stack((lat_grid, lon_grid, depth_grid), axis=-1)
        # ic(block_coords.shape)
        block_coordinates.append(block_coords)

    # Stack all blocks into a single NumPy array
    return np.stack(blocks, axis=0), np.stack(block_coordinates, axis=0)


def extract_blocks_from_coordinates_v3(
        array, 
        latitudes,
        longitudes,
        depths, 
        sampled_coords,
        block_shape):
    """
    Extracts blocks of a given shape (units: degree) from a 4D array based on geographic coordinates.

    Args:
        array (np.ndarray): The 4D array of shape [lat, lon, depth, channel].
        latitudes (np.ndarray): 1D array of latitude values 
        longitudes (np.ndarray): 1D array of longitude values
        depths (np.ndarray): 1D array of depth values
        sampled_coords (list): List of sampled coordinates as tuples (lat, lon, press).
        block_shape (tuple): The shape of the block to extract (block_lat, block_lon). (units: degree)
    """

    (lat_num, lon_num, depth_num, channel_num) = array.shape
    # ic(channel_num, press_num, lat_num, lon_num)

    # Validate inputs
    if len(array.shape) != 4:
        raise ValueError("Input array must have 4 dimensions.")

    # Get block shape and calculate half sizes
    block_lat, block_lon, block_depth = block_shape
    lat_half, lon_half, depth_half = block_lat // 2, block_lon // 2, block_depth // 2

    # Prepare output list for blocks
    blocks = []
    block_coordinates = []

    # Iterate through sampled coordinates
    for lat, lon, depth in sampled_coords:
        # print(f'lat={lat}, lon={lon}, depth={depth}')
        
        # Calculate the slicing ranges
        lat_start, lat_end = lat - lat_half, lat + lat_half
        # print(f'lat_range: {lat_start}~{lat_end}')
        if lat_start < -90:
            lat_end = lat_end - lat_start - 90
            lat_start = -90
            # print(f'update lat_range: {lat_start}~{lat_end}')
        lat_start_idx = find_nearest_index(latitudes, lat_start)
        lat_end_idx = find_nearest_index(latitudes, lat_end)
        if lat_start_idx == 170:
            lat_end_idx = 180
        if lat_start_idx >= 170:
            lat_end_idx = 180
            lat_start_idx = 170
        # print(f'lat_idx_range: {lat_start_idx}~{lat_end_idx}')

        lon_start, lon_end = lon - lon_half, lon + lon_half
        # print(f'lon_range: {lon_start}~{lon_end}')
        if lon_start < 0: # TODO: cycle
            lon_end = lon_end - lon_start
            lon_start = 0
            # print(f'update lon_range: {lon_start}~{lon_end}')
        if lon_end > 360:
            lon_start = lon_start - (lon_end - 360)
            lon_end = 360

        lon_start_idx = find_nearest_index(longitudes, lon_start)
        lon_end_idx = find_nearest_index(longitudes, lon_end)
        if lon_start_idx == 340:
            lon_end_idx = 360
        # print(f'lon_idx_range: {lon_start_idx}~{lon_end_idx}')

        depth_idx = find_nearest_index(depths, depth)
        depth_start_idx, depth_end_idx = depth_idx - depth_half, depth_idx + depth_half
        
        if (lat_start_idx >= 0 and lat_end_idx <= lat_num and
            lon_start_idx >= 0 and lon_end_idx <= lon_num):

            if depth_num == 1:
                block = array[
                    lat_start_idx:lat_end_idx, 
                    lon_start_idx:lon_end_idx, 
                    :, :]
                block_depths = depths
            else:
                block = array[
                    lat_start_idx:lat_end_idx, 
                    lon_start_idx:lon_end_idx, 
                    depth_start_idx:depth_end_idx,
                    :]
                block_depths = depths[depth_start_idx:depth_end_idx]
            # ic(block.shape)
            
            # Generate the full coordinate array for the block
            block_latitudes = latitudes[lat_start_idx:lat_end_idx]
            block_longitudes = longitudes[lon_start_idx:lon_end_idx]

        if list(block.shape[:-1]) != block_shape:
        # if block.shape[:-1] != block_shape:
            print(f'extract block: {block.shape}/{block_shape}')
            print(f'lat={lat}, lon={lon}, depth={depth}')
            print(f'lat_start_idx: {lat_start_idx}')
            print(f'lat_end_idx: {lat_end_idx}')
            print(f'lon_start_idx: {lon_start_idx}')
            print(f'lon_end_idx: {lon_end_idx}')
            print(f'depth_start_idx: {depth_start_idx}')
            print(f'depth_end_idx: {depth_end_idx}')
            print(f'lat_num: {lat_num}')
            print(f'lon_num: {lon_num}')
            print(f'depth_num: {depth_num}')
            exit()
            
        blocks.append(block)

        # Create a 3D meshgrid of coordinates for the block
        lat_grid, lon_grid, depth_grid = np.meshgrid(block_latitudes, block_longitudes, block_depths, indexing="ij")
        block_coords = np.stack((lat_grid, lon_grid, depth_grid), axis=-1)
        # ic(block_coords.shape)
        block_coordinates.append(block_coords)

    # Stack all blocks into a single NumPy array
    return np.stack(blocks, axis=0), np.stack(block_coordinates, axis=0)


def extract_patch_from_coordinates_v1(array, latitudes, longitudes, sampled_coords, patch_shape, source=''):
    """
    Extracts blocks of a given shape (units: grid) from a 3D array based on geographic coordinates.

    Args:
        array (np.ndarray): The 2D array of shape [lat, lon].
        latitudes (np.ndarray): 1D array of latitude values (length = array.shape[1]).
        longitudes (np.ndarray): 1D array of longitude values (length = array.shape[2]).
        sampled_coords (list): List of sampled coordinates as tuples (lat, lon, press).
        patch_shape (tuple): The shape of the patch to extract (block_lat, block_lon). # units: grid

    Returns:
        - patches (np.ndarray): A 3D array of extracted patches, shape [n_patches, patch_lat, patch_lon].
        - patch_coordinates (list): A list of 2D coordinate arrays for each patch.
            Each element is an array of shape [patch_lat, patch_lon, 2] containing (lat, lon).
    """
    (channel_num, lat_num, lon_num) = array.shape

    # Get patch shape and calculate half sizes
    patch_lat, patch_lon = patch_shape
    lat_half, lon_half  = patch_lat // 2, patch_lon // 2

    # Prepare output list for blocks
    patches = []
    patch_coordinates = []

    # Iterate through sampled coordinates
    for lat, lon, _ in sampled_coords:
        # print(f'\nlat={lat}, lon={lon}')

        # Find the nearest indices for the given coordinate
        lat_idx = find_nearest_index(latitudes, lat)
        lon_idx = find_nearest_index(longitudes, lon)
        # print(f'lat_idx={lat_idx}, lon_idx={lon_idx}')

        # Calculate the slicing ranges
        lat_start_idx, lat_end_idx = lat_idx - lat_half, lat_idx + lat_half
        lon_start_idx, lon_end_idx = lon_idx - lon_half, lon_idx + lon_half
        # print(f'lat_idx_range: {lat_start_idx}~{lat_end_idx}')
        # print(f'lon_idx_range: {lon_start_idx}~{lon_end_idx}')

        if lat_start_idx < 0:
            lat_end_idx = lat_end_idx - lat_start_idx
            lat_start_idx = 0
            # print(f'update lat_idx range to {lat_start_idx}~{lat_end_idx}')
        
        if lat_end_idx > lat_num:
            lat_start_idx = lat_start_idx - (lat_end_idx - lat_num)
            lat_end_idx = lat_num
            # print(f'update lat_idx range to {lat_start_idx}~{lat_end_idx}')

        if lon_start_idx < 0: # cycle
            # print(f'lon cycle')
            patch1 = array[:, lat_start_idx:lat_end_idx, lon_start_idx:]
            patch2 = array[:, lat_start_idx:lat_end_idx, 0:lon_end_idx]
            patch = np.concatenate((patch1, patch2), axis=2)
            patch_latitudes = latitudes[lat_start_idx:lat_end_idx]
            patch_longitudes1 = longitudes[lon_start_idx:]
            patch_longitudes2 = longitudes[0:lon_end_idx]
            patch_longitudes = np.concatenate((patch_longitudes1, patch_longitudes2), axis=0)

        if lon_end_idx > lon_num: # cycle
            # print(f'lon cycle')
            patch1 = array[:, lat_start_idx:lat_end_idx, lon_start_idx:]
            patch2 = array[:, lat_start_idx:lat_end_idx, 0:lon_end_idx-lon_num]
            patch = np.concatenate((patch1, patch2), axis=2)
            patch_latitudes = latitudes[lat_start_idx:lat_end_idx]
            patch_longitudes1 = longitudes[lon_start_idx:]
            patch_longitudes2 = longitudes[0:lon_end_idx-lon_num]
            patch_longitudes = np.concatenate((patch_longitudes1, patch_longitudes2), axis=0)

        if (lat_start_idx >= 0 and lat_end_idx <= lat_num and
            lon_start_idx >= 0 and lon_end_idx <= lon_num):
            
            patch = array[:, lat_start_idx:lat_end_idx, lon_start_idx:lon_end_idx]
            patch_latitudes = latitudes[lat_start_idx:lat_end_idx]
            patch_longitudes = longitudes[lon_start_idx:lon_end_idx]

        # ic(patch.shape)

        if list(patch.shape[1:]) != patch_shape:
        # if patch.shape[1:] != patch_shape:
            ic(lat_start_idx, lat_end_idx, lon_start_idx, lon_end_idx)
            ic(patch.shape)
            exit()
        patches.append(patch)

        # Create a 2D coordinate grid for the patch
        lat_grid, lon_grid, depth_grid = np.meshgrid(patch_latitudes, patch_longitudes, [0], indexing="ij") # Shape: [patch_lat, patch_lon, 3]
        patch_coords = np.stack((lat_grid, lon_grid, depth_grid), axis=-1)
        # patch_coords = np.stack((lat_grid, lon_grid, np.zeros(lat_grid.shape)), axis=-1)
        # print(f'patch_coords: {patch_coords.shape}')
        patch_coordinates.append(patch_coords)

    # Stack all blocks into a single NumPy array
    return np.stack(patches, axis=0), np.stack(patch_coordinates, axis=0)


def extract_patch_from_coordinates_v2(array, latitudes, longitudes, sampled_coords, patch_shape, source=''):
    """
    Extracts blocks of a given shape (units:degree) from a 3D array based on geographic coordinates.

    Args:
        array (np.ndarray): The 2D array of shape [lat, lon].
        latitudes (np.ndarray): 1D array of latitude values (length = array.shape[1]).
        longitudes (np.ndarray): 1D array of longitude values (length = array.shape[2]).
        sampled_coords (list): List of sampled coordinates as tuples (lat, lon, press).
        patch_shape (tuple): The shape of the patch to extract (block_lat, block_lon). units: degree

    Returns:
        - patches (np.ndarray): A 3D array of extracted patches, shape [n_patches, patch_lat, patch_lon].
        - patch_coordinates (list): A list of 2D coordinate arrays for each patch.
            Each element is an array of shape [patch_lat, patch_lon, 2] containing (lat, lon).
    """
    (channel_num, lat_num, lon_num) = array.shape

    # Get patch shape and calculate half sizes
    patch_lat, patch_lon = patch_shape
    lat_half, lon_half  = patch_lat // 2, patch_lon // 2

    # Prepare output list for blocks
    patches = []
    patch_coordinates = []

    # Iterate through sampled coordinates
    for lat, lon, _ in sampled_coords:

        # Calculate the slicing ranges
        lat_start, lat_end = lat - lat_half, lat + lat_half
        lon_start, lon_end = lon - lon_half, lon + lon_half
        lat_start_idx = find_nearest_index(latitudes, lat_start)
        lat_end_idx = find_nearest_index(latitudes, lat_end)
        lon_start_idx = find_nearest_index(longitudes, lon_start)
        lon_end_idx = find_nearest_index(longitudes, lon_end)

        if lat_start_idx < 0:
            # print(f'update lat range from {lat_start}~{lat_end}')
            lat_end_idx = lat_end_idx - lat_start_idx
            lat_start = 0
            # print(f'to {lat_start}~{lat_end}')
        if lat_end_idx > lat_num:
            # print(f'update lat range from {lat_start}~{lat_end}')
            lat_start_idx = lat_start_idx - (lat_end_idx - lat_num)
            lat_end_idx = lat_num
            # print(f'to {lat_start}~{lat_end}')
        if lon_start_idx < 0:
            # print(f'update lon range from {lon_start}~{lon_end}')
            lon_end_idx = lon_end_idx - lon_start_idx
            lon_start_idx = 0
            # print(f'to {lon_start}~{lon_end}')
        if lon_end_idx > lon_num:
            # print(f'update lon range from {lon_start}~{lon_end}')
            lon_start_idx = lon_start_idx - (lon_end_idx - lon_num)
            lon_end_idx = lon_num        
            # print(f'to {lon_start}~{lon_end}')  
            
        # Extract the block
        patch = array[:, lat_start_idx:lat_end_idx, lon_start_idx:lon_end_idx]
        # ic(patch.shape)
        # if patch.shape != patch_shape:
        #     ic(patch.shape)
        #     ic(lat_start_idx, lat_end_idx, lon_start_idx, lon_end_idx)
        #     exit()
        patches.append(patch)

        # Extract the latitude and longitude ranges for the patch
        patch_latitudes = latitudes[lat_start_idx:lat_end_idx]
        patch_longitudes = longitudes[lon_start_idx:lon_end_idx]

        # Create a 2D coordinate grid for the patch
        lon_grid, lat_grid = np.meshgrid(patch_latitudes, patch_longitudes)
        patch_coords = np.stack((lat_grid, lon_grid, np.zeros(lat_grid.shape)), axis=-1)  # Shape: [patch_lat, patch_lon, 3]
        # ic(patch_coords.shape)
        patch_coordinates.append(patch_coords)

    # Stack all blocks into a single NumPy array
    return np.stack(patches, axis=0), np.stack(patch_coordinates, axis=0)

def extract_obs_in_blocks(file_path, coords, block_shape):
    # block_shape: tuple, units: (degree, degree, m)

    # print(f"Reading in situ observations from: {file_path}")
    ds = xr.open_dataset(file_path)

    lat_size, lon_size, depth_size = block_shape

    # Initialize a dictionary to store results for each point
    sal_values = []
    temp_values = [] 
    # coords = []

    # Iterate over the coordinate points
    for lat, lon, depth in coords:
        ic(lat, lon, depth)

        # Define the range for each coordinate point
        lat_min, lat_max = lat - lat_size / 2, lat + lat_size / 2
        lon_min, lon_max = lon - lon_size / 2, lon + lon_size / 2
        depth_min, depth_max = depth - depth_size / 2, depth + depth_size / 2

        profiles_in_range = ds.where(
            (ds["lon"] >= lon_min) & (ds["lon"] <= lon_max) &
            (ds["lat"] >= lat_min) & (ds["lat"] <= lat_max),
            drop=True
        )
        # ic(profiles_in_range)
        # ic(profiles_in_range.coords['depth'])
        obs_in_range = profiles_in_range.where(
            (profiles_in_range["depth"] >= depth_min) &
            (profiles_in_range["depth"] <= depth_max),
            drop=True
        )
        ic(obs_in_range)

        temp = obs_in_range["temp"].values
        sal = obs_in_range["sal"].values
        lat = obs_in_range["lat"].values
        lon = obs_in_range["lon"].values
        depth = obs_in_range["depth"].values

        ic(temp.shape, sal.shape, lat.shape, lon.shape, depth.shape)

    return temp, sal, lat, lon, depth


if __name__ == '__main__':

    date_str = '20100115'
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    ic(date_obj)

    latitudes=np.arange(-90, 90, 1)
    longitudes=np.arange(-180, 180, 1)
    depths=[0, 10, 25, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000]
    ic(len(latitudes), len(longitudes), len(depths))

    sampled_points = sample_points_in_2D_area(
        shape=(180, 360), n=10, patch_shape=(10, 10))
    ic(sampled_points)

    sampled_coords = map_indices_to_coordinates_2D(
        latitudes=latitudes,
        longitudes=longitudes,
        sampled_points=sampled_points)
    ic(sampled_coords)




    





