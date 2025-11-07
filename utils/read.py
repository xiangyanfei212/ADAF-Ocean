import os
import h5py
import json
import numpy as np
import xarray as xr
import pandas as pd
from icecream import ic
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime, timedelta
from functools import reduce
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def read_foracet_file(
        file_path,
        lead_time,
        fill_land=True,
        interp=False, # 1 -> 0.25 via interpolation
        variables=['T', 'S', 'U', 'V', 'SSH']
    ):

    file_time_str = file_path.split('/')[-1].split('.')[0]
    ic_time = datetime.strptime(file_time_str, "%Y%m%d")
    valid_time = ic_time + timedelta(days=lead_time)

    with xr.open_dataset(file_path) as ds:
        pred = ds["seq_pred"].values[lead_time] # [variables, lats, lons]
        # variables = ds.coords['variables'].values
        lats = ds.coords['lat'].values
        lons = ds.coords['lon'].values

        data = []
        for vi, var in enumerate(variables):
            var_data = pred[vi]
            # ic(var, var_data.shape)
            if fill_land:
                var_data = fill_land_with_lat_mean_2d(var_data) # (lat, lon)
            if interp:
                # print(f'interp data from {var_data.shape} to (721, 1440)')
                interpolator = RegularGridInterpolator(
                    (lats, lons), var_data, method='linear', bounds_error=False, fill_value=None)
                lat_high = np.linspace(-90, 90, 720, endpoint=False)
                lon_high = np.linspace(0, 360, 1440, endpoint=False)
                lat_grid, lon_grid = np.meshgrid(lat_high, lon_high, indexing='ij')  # shape (721, 1440)
                var_data = interpolator((lat_grid, lon_grid))
                # lats = lat_high
                # lons = lon_high
                # ic(var_data.shape)
            data.append(var_data)
        
        data = np.stack(data, axis=0) # (vars, lat, lon)
        data = data[np.newaxis,:,:,:] # (depth, vars, lat, lon)
        data = np.transpose(data, (2, 3, 0, 1)) # (lat, lon, depth, vars)

    depths = [0]

    if interp:
        return data, lat_high, lon_high, depths, valid_time
    else:
        return data, lats, lons, depths, valid_time

def in_situ_obs_qc(
    data,
    stats_file,
    channel_names
):

    # Load mean and std values from the JSON file
    with open(stats_file, 'r') as file:
        stats_values = json.load(file)

    # Extract mean and std values into arrays based on channel order
    if not channel_names:
        channel_names = list(stats_values.keys())
    max_vals = np.array([stats_values[channel]["max"] for channel in channel_names])
    min_vals = np.array([stats_values[channel]["min"] for channel in channel_names])

    # 广播 min_vals 和 max_vals 到 data 的形状
    min_vals = min_vals[np.newaxis, np.newaxis, :]  # 形状变为 [1, 1, channels]
    max_vals = max_vals[np.newaxis, np.newaxis, :]  # 形状变为 [1, 1, channels]

    # 超出范围的值为 0
    data = np.where((data >= min_vals) & (data <= max_vals), data, np.nan)

    return data


def sate_sss_qc(
    data,
    stats_file,
    channel_names
):

    # Load mean and std values from the JSON file
    with open(stats_file, 'r') as file:
        stats_values = json.load(file)

    # Extract mean and std values into arrays based on channel order
    if not channel_names:
        channel_names = list(stats_values.keys())
    max_vals = np.array([stats_values[channel]["max"] for channel in channel_names])
    min_vals = np.array([stats_values[channel]["min"] for channel in channel_names])

    # 广播 min_vals 和 max_vals 到 data 的形状
    min_vals = min_vals[np.newaxis, np.newaxis, :]  # 形状变为 [1, 1, channels]
    max_vals = max_vals[np.newaxis, np.newaxis, :]  # 形状变为 [1, 1, channels]

    # 超出范围的值为 0
    data = np.where((data >= min_vals) & (data <= max_vals), data, np.nan)

    return data




def get_all_file_paths(folder_path, suffix):
    """
    Traverse the given folder and return a list of all file paths.

    Args:
        folder_path (str): The path to the folder to traverse.

    Returns:
        list: A list containing the full paths of all files in the folder.
    """
    file_paths = []
    
    # Use os.walk to traverse the folder
    # for root, dirs, files in os.walk(folder_path):
    #     for file in files:
    #         if file.endswith(suffix):
    #             # Construct the full file path
    #             full_path = os.path.join(root, file)
    #             file_paths.append(full_path)

    for file in os.listdir(folder_path):
        # 构造文件的完整路径
        full_path = os.path.join(folder_path, file)
        # 检查是否是文件并且后缀匹配
        if os.path.isfile(full_path) and file.endswith(suffix):
            file_paths.append(full_path)
    
    return file_paths

def multiply_list_elements(lst):
    """
    Multiplies all elements in a list together.

    Args:
        lst (list): A list of numbers, e.g. [150, 150, 4].

    Returns:
        int: The result of multiplying all the elements in the list.
    """
    return reduce(lambda x, y: x * y, lst)


def filter_AVHRR_pathfinder_files_by_time(directory, start_time, end_time):
    """
    Filter files in a directory based on a specified time range.

    Parameters:
        directory (str): Path to the directory containing the files.
        start_time (str): Start time in the format 'YYYYMMDDHHMMSS'.
        end_time (str): End time in the format 'YYYYMMDDHHMMSS'.

    Returns:
        list: A list of file paths that fall within the specified time range.
    """
    # Parse start and end times into datetime objects
    # start_dt = datetime.strptime(start_time, "%Y%m%d%H%M%S")
    # end_dt = datetime.strptime(end_time, "%Y%m%d%H%M%S")

    directory = os.path.join(directory, str(start_time.year))
    # print(f'Find AVHRR pathfinde files in {directory}')
    # print(f'time_range: {start_time}~{end_time}')

    # Initialize a list to store filtered files
    filtered_files = []

    # Iterate through files in the directory
    for file in os.listdir(directory):
        # Extract the timestamp from the filename (assuming it's in the format 'YYYYMMDDHHMMSS')
        try:
            timestamp_str = file.split('-')[0]  # Extract the first part of the filename
            file_dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")  # Parse to datetime
        except (ValueError, IndexError):
            # Skip files that don't match the expected format
            continue

        # Check if the file's timestamp is within the specified range
        if start_time <= file_dt <= end_time:
            filtered_files.append(os.path.join(directory, file))  # Add full file path

    return filtered_files

def read_and_convert_lon_to_360(file_path, lon_name='longitude'):
    """
    将 NetCDF 文件的经度从 -180~180 转换为 0~360。

    Parameters:
        file_path (str): 输入 NetCDF 文件路径。
    """
    ds = xr.open_dataset(file_path)
    lon = ds.coords[lon_name].values
    # ic(lon)
    
    # 如果经度范围是 -180~180，则转换为 0~360
    if np.any(lon < 0):
        # 1. 调整经度到 0~360 范围
        ds['longitude_adjusted'] = xr.where(
            ds[lon_name] < 0, ds[lon_name] % 360, ds[lon_name]
        )

        # 2. 排序经度，并替换维度
        ds = (
            ds.swap_dims({lon_name: 'longitude_adjusted'}) # 替换维度
            .sortby('longitude_adjusted')                  # 按新经度排序
            .drop(lon_name)                                # 删除旧经度变量
        )

        # 3. 重命名经度回原始名称
        ds = ds.rename({'longitude_adjusted': lon_name})
        
        # print("Converting longitude from -180~180 to 0~360...")
        
        # ds['longitude_adjusted'] = xr.where(
        #     ds[lon_name] < 0, ds[lon_name]%360, ds[lon_name])
        # ds = (ds.swap_dims(
        #     {lon_name: 'longitude_adjusted'}).sel(
        #         **{'longitude_adjusted': sorted(
        #             ds.longitude_adjusted)}).drop(lon_name))
        # ds = ds.rename({'longitude_adjusted': lon_name})
        
    return ds

def fill_land_with_lat_mean(data):

    # 查看原始数据的 NaN 比例
    # print(f"NaN ratio: {np.isnan(data).mean():.2%}")

    # Step 1: 沿经度方向 (axis=2) 计算纬向平均，忽略 NaN
    lat_mean = np.nanmean(data, axis=2)  # 纬向平均，结果形状为 (23, 181)
    # ic(lat_mean.shape, np.min(lat_mean))

    # Step 2: 扩展纬向平均值的形状，方便替换
    lat_mean_expanded = np.expand_dims(lat_mean, axis=2)  # 扩展维度到 (23, 181, 1)
    lat_mean_expanded = np.repeat(lat_mean_expanded, 360, axis=2)  # 重复到 (23, 181, 360)
    # ic(lat_mean_expanded.shape)

    # Step 3: 填充 NaN 值
    # 找到 NaN 的位置
    nan_mask = np.isnan(data)
    # 用纬向平均值填充 NaN
    data_filled = np.where(nan_mask, lat_mean_expanded, data)

    # 验证填充结果
    # print(f"NaN ratio after filling: {np.isnan(data_filled).mean():.2%}")

    return data_filled

def fill_land_with_lat_mean_2d(data_2d):

    # 查看原始数据的 NaN 比例
    # print(f"NaN ratio: {np.isnan(data_2d).mean():.2%}")
    lat_num, lon_num = data_2d.shape

    # Step 1: 沿经度方向计算纬向平均，忽略 NaN
    lat_mean = np.nanmean(data_2d, axis=1)  # 纬向平均，结果形状为 (720)

    # Step 2: 扩展纬向平均值的形状，方便替换
    lat_mean_expanded = np.expand_dims(lat_mean, axis=1)  # 扩展维度到 (720, 1)
    lat_mean_expanded = np.repeat(lat_mean_expanded, lon_num, axis=1)  # 重复到 (720, 1440)
    # ic(lat_mean_expanded.shape)

    # Step 3: 填充 NaN 值
    # 找到 NaN 的位置
    nan_mask = np.isnan(data_2d)
    # 用纬向平均值填充 NaN
    data_filled = np.where(nan_mask, lat_mean_expanded, data_2d)

    # 验证填充结果
    # print(f"NaN ratio after filling: {np.isnan(data_filled).mean():.2%}")

    return data_filled


def read_glorys_file(file_path, variables):

    file_time_str = file_path.split('/')[-1].split('.')[0]
    # ic(file_time_str)
    file_time_obj = datetime.strptime(file_time_str, "%Y%m%d")

    data = []
    with h5py.File(file_path, 'r') as h5_file:
        for var in variables:
            var_data = h5_file[var][:]
            if var == 'SSH':
                var_data = np.tile(var_data, (23, 1, 1))
            data.append(var_data)
            
    data = np.stack(data, axis=0)
    data = np.transpose(data, (2, 3, 1, 0))
    lats = np.arange(-90, 90)
    lons = np.arange(0, 360)
    depths = [0, 2, 5, 7, 11, 15, 21, 29, 40, 55, 77, 92, 109, 130, 155, 186, 222, 266, 318, 380, 453, 541, 643]

    return data[:180], lats, lons, depths, file_time_obj


def read_glorys_file_v2(file_path, variables, fill_land=False, only_surface=True):

    # file_time_str = file_path.split('/')[-1].split('.')[0]
    # file_time_obj = datetime.strptime(file_time_str, "%Y%m%d")
    file_time_str = file_path.split('/')[-1].split('.')[0]
    try:
        file_time_obj = datetime.strptime(file_time_str, "%Y%m%d")
    except ValueError:
        file_time_obj = None

    data = []
    # print(f'Read glorys from {file_path}')
    with h5py.File(file_path, 'r') as h5_file:
        for var in variables:
            var_data = h5_file[var][:]
            # ic(var, var_data.shape)
            if fill_land:
                var_data = fill_land_with_lat_mean(var_data)
            if var == 'SSH':
                var_data = np.tile(var_data, (23, 1, 1))
            # ic(var, var_data.shape)
            data.append(var_data)
    data = np.stack(data, axis=0)
    data = np.transpose(data, (2, 3, 1, 0)) # lat,lon,depth,vars
    # ic(data.shape)

    lats = np.linspace(-90, 90, data.shape[0])
    lons = np.linspace(0, 360, data.shape[1], endpoint=False)
    # lats = np.arange(-90, 90)
    # lons = np.arange(0, 360)

    if only_surface:
        data = np.expand_dims(data[:,:,0,:], axis=2)
        depths = [0]
    else:
        depths = [0, 2, 5, 7, 11, 15, 21, 29, 40, 55, 77, 92, 109, 130, 155, 186, 222, 266, 318, 380, 453, 541, 643]

    return data[:data.shape[0]-1], lats, lons, depths, file_time_obj


def read_hycom_file(
    file_path:str,
    variables:list,
):

    file_time_str = file_path.split('/')[-1].split('.')[0]
    # print(f'file_time: {file_time_str}')
    try:
        file_time_obj = datetime.strptime(file_time_str[:8], "%Y%m%d")
    except ValueError:
        file_time_obj = None

    data = []
    ds = xr.open_dataset(file_path)
    for var in variables:
        var_data = ds[var].values[np.newaxis,:,:] # (1, lat, lon, 1)
        data.append(var_data)
    data = np.stack(data, axis=0) # (var, 1, lat, lon)
    data = np.transpose(data, (2, 3, 1, 0)) # lat,lon,depth,vars

    lats = ds.coords['lat'].values
    lons = ds.coords['lon'].values
    depths = [0]
    
    return data, lats, lons, depths, file_time_obj


def read_multi_nc_files(file_files, variable_name):
    """
    Load and combine a specific variable (e.g., 'sst') from multiple NetCDF files in a directory.

    Parameters:
        file_files (list): Path of files.
        variable_name (str): The name of the variable to extract (e.g., 'sst').

    Returns:
        xarray.Dataset: A Dataset containing only the specified variable.
    """
    try:
        ds = xr.open_mfdataset(file_files, combine="by_coords")

        lon_name = 'lon'
        lon = ds.coords[lon_name].values
        
        # 如果经度范围是 -180~180，则转换为 0~360
        if np.any(lon < 0):
            # 1. 调整经度到 0~360 范围
            ds['longitude_adjusted'] = xr.where(
                ds[lon_name] < 0, ds[lon_name] % 360, ds[lon_name]
            )
            # 2. 排序经度，并替换维度
            ds = (
                ds.swap_dims({lon_name: 'longitude_adjusted'}) # 替换维度
                .sortby('longitude_adjusted')                  # 按新经度排序
                .drop(lon_name)                                # 删除旧经度变量
            )
            # 3. 重命名经度回原始名称
            ds = ds.rename({'longitude_adjusted': lon_name})
            
        
        data = ds[variable_name].values
        lat = ds.coords['lat'].values
        lon = ds.coords['lon'].values

        return data, lat, lon
    except Exception as e: 
        print(f"Error loading file: {file_files}.")
        return None, None, None


if __name__ == '__main__':
    
    from plot import *
    from block import *
    from sample import *
    from normalization import *

    target, lats, lons, depths, time_obj = read_glorys_file_v2(
        '/groups/hxm_group/home/share/xiangyanfei/OCEAN_DATA/glorys1p_23layers/2010/20100304.h5',
        fill_land=True)
    # show_map(extent=[0, 360, -90, 90], data2d=target[:, :, 0, 0], title_str='Glorys_SSS', save_path='./imgs/glorys_sss.png')
    exit()
    target_blocks, target_center_coords, target_block_coords = slice_array(
            target,
            lats, lons, depths,
            lat_block=6, lon_block=6, depth_block=8, 
            lat_step=3, lon_step=3, depth_step=3)
    n_blocks, lat_size, lon_size, depth_size, channel_num = target_blocks.shape
    target_blocks = target_blocks.reshape(n_blocks, lat_size * lon_size * depth_size, channel_num)
    n_blocks, lat_size, lon_size, depth_size, channel_num = target_block_coords.shape
    target_block_coords = target_block_coords.reshape(n_blocks, lat_size * lon_size * depth_size, channel_num)
    print(f'target_blocks: {target_blocks.shape}')
    print(f'target_block_coords: {target_block_coords.shape}')
    print(f'target_center_coords: {target_center_coords.shape}')

    target_stats_file = '/groups/hxm_group/home/share/xiangyanfei/02_DA_ocean/stats/glorys.json'
    # location_stats_file = '/groups/hxm_group/home/share/xiangyanfei/02_DA_ocean/stats/location.json'
    target_vars = ['T', 'S', 'U', 'V', 'SSH']
    target_blocks, _ = min_max_normalize_v2(target_blocks, target_stats_file, target_vars)
    target_block_coords = location_normalize(target_block_coords)
    print(f'target_center_coords after loc norm: {target_center_coords.shape}')
    ic(np.min(target_block_coords[:,:,0]), np.max(target_block_coords[:,:,0]))
    ic(np.min(target_block_coords[:,:,1]), np.max(target_block_coords[:,:,1]))
    ic(np.min(target_block_coords[:,:,2]), np.max(target_block_coords[:,:,2]))
    ic(np.min(target_block_coords[:,:,3]), np.max(target_block_coords[:,:,3]))
    ic(np.min(target_block_coords[:,:,4]), np.max(target_block_coords[:,:,4]))
    exit()
    
    # block_idx = np.random.randint(0, n_blocks, size=n_blocks)
    # target_blocks = target_blocks[block_idx]
    # target_block_coords = target_block_coords[block_idx]
    # target_center_coords = target_center_coords[block_idx]
    # print('select subset of target blocks')
    # print(f'target_blocks: {target_blocks.shape}')
    # print(f'target_block_coords: {target_block_coords.shape}')
    # print(f'target_center_coords: {target_center_coords.shape}')

    # %% Background
    # bg_dir='/groups/hxm_group/home/share/xiangyanfei/OCEAN_DATA/glorys1p_23layers' 
    # bg_time = time_obj - timedelta(days = 1)
    # ic(bg_time)
    # bg_file_path = os.path.join(bg_dir, str(bg_time.year), bg_time.strftime("%Y%m%d")+'.h5')
    # bg, lats, lons, depths, _ = read_glorys_file(bg_file_path)
    # bg_blocks, bg_block_coords = extract_blocks_from_coordinates_v2(
    #     bg,
    #     lats, lons, depths,
    #     target_center_coords,
    #     (6, 6, 8))
    # n_blocks, lat_size, lon_size, depth_size, channel_num = bg_blocks.shape
    # bg_blocks = bg_blocks.reshape(n_blocks, lat_size * lon_size * depth_size, channel_num)
    # bg_block_coords = bg_block_coords.reshape(n_blocks, lat_size * lon_size * depth_size, 3)
    # print(f'bg_blocks: {bg_blocks.shape}')
    # print(f'bg_block_coords: {bg_block_coords.shape}')

    # bg_stats_file = '/groups/hxm_group/home/share/xiangyanfei/02_DA_ocean/stats/glorys.json'
    # bg_vars = ['T', 'S', 'U', 'V', 'SSH']
    # bg_blocks, _ = min_max_normalize_v2(bg_blocks, bg_stats_file, bg_vars)
    # bg_block_coords = location_normalize(bg_block_coords)
    
    # %% Satellite SSW
    # sate_ssw_asc_dir='/groups/hxm_group/home/share/xiangyanfei/OCEAN_DATA/00_raw_data/SSW/WIND_GLO_PHY_L3_MY_012_005/cmems_obs-wind_glo_phy_my_l3-metopa-ascat-asc-0.25deg_P1D-i'
    # sate_ssw_des_dir='/groups/hxm_group/home/share/xiangyanfei/OCEAN_DATA/00_raw_data/SSW/WIND_GLO_PHY_L3_MY_012_005/cmems_obs-wind_glo_phy_my_l3-metopa-ascat-des-0.25deg_P1D-i'
    sate_date_obj = time_obj - timedelta(days = 1)
    print(f'satellite date: {sate_date_obj}')
    sate_date_str = sate_date_obj.strftime('%Y%m%d')
    # sate_ssw_asc_file = os.path.join(sate_ssw_asc_dir, str(sate_date_obj.year), f'{sate_date_str}.nc')
    # ds_asc = read_and_convert_lon_to_360(sate_ssw_asc_file)
    # sate_asc_ssw_u = ds_asc['eastward_wind'].values 
    # sate_asc_ssw_v = ds_asc['northward_wind'].values
    # lats = ds_asc.coords['latitude'].values
    # lons = ds_asc.coords['longitude'].values
    # print('lats: ', lats)
    # print('lons: ', lons)
    # ds_asc.close()
    # sate_ssw_des_file = os.path.join(sate_ssw_des_dir, str(sate_date_obj.year), f'{sate_date_str}.nc')
    # ds_dec = read_and_convert_lon_to_360(sate_ssw_des_file)
    # sate_des_ssw_u = ds_dec['eastward_wind'].values 
    # sate_des_ssw_v = ds_dec['northward_wind'].values
    # ds_asc.close()
    # sate_ssw = np.stack([sate_asc_ssw_u, sate_asc_ssw_v, sate_des_ssw_u, sate_des_ssw_v], axis=1)
    # sate_ssw, sate_ssw_coords = extract_patch_from_coordinates_v1(
    #     np.squeeze(sate_ssw),
    #     lats, lons,
    #     target_center_coords,
    #     (24, 24),
    # )
    # sate_ssw = np.transpose(sate_ssw, (0, 2, 3, 1))
    # n_blocks, lat_size, lon_size, channel_num = sate_ssw.shape
    # sate_ssw = sate_ssw.reshape(n_blocks, lat_size * lon_size, channel_num)
    # sate_ssw_coords = sate_ssw_coords.reshape(n_blocks, lat_size * lon_size, 3)
    # print(f'sate_ssw: {sate_ssw.shape}')
    # print(f'sate_ssw_coords: {sate_ssw_coords.shape}')

    # %% Satellite SSS
    # sate_sss_asc_dir='/groups/hxm_group/home/share/xiangyanfei/OCEAN_DATA/00_raw_data/SSS/SMOS_CATDS_L2Q/cmems_obs-mob_glo_phy-sss_mynrt_smos-asc_P1D'
    # sate_sss_des_dir='/groups/hxm_group/home/share/xiangyanfei/OCEAN_DATA/00_raw_data/SSS/SMOS_CATDS_L2Q/cmems_obs-mob_glo_phy-sss_mynrt_smos-des_P1D'
    # sss_asc_file = os.path.join(sate_sss_asc_dir, str(sate_date_obj.year), f'{sate_date_str}.nc')
    # sss_des_file = os.path.join(sate_sss_des_dir, str(sate_date_obj.year), f'{sate_date_str}.nc')
    # ds_asc = read_and_convert_lon_to_360(sss_asc_file)
    # sss_asc = ds_asc['Sea_Surface_Salinity'].values
    # lats = ds_asc.coords['latitude'].values
    # lons = ds_asc.coords['longitude'].values
    # print('lats: ', lats)
    # print('lons: ', lons)
    # ds_asc.close()
    # ds_des = read_and_convert_lon_to_360(sss_des_file)
    # sss_des = ds_des['Sea_Surface_Salinity'].values
    # ds_des.close()
    # sss = np.squeeze(np.stack([sss_asc, sss_des], axis=0))
    # print(f'sss_asc: {sss_asc.shape}, sss_des: {sss_des.shape}, sss: {sss.shape}')
    # sate_sss, sate_sss_coords = extract_patch_from_coordinates_v1(
    #     sss,
    #     lats, lons,
    #     target_center_coords,
    #     (30, 30),
    # )
    # sate_sss = np.transpose(sate_sss, (0, 2, 3, 1))
    # n_blocks, lat_size, lon_size, channel_num = sate_sss.shape
    # sate_sss = sate_sss.reshape(n_blocks, lat_size * lon_size, channel_num)
    # sate_sss_coords = sate_sss_coords.reshape(n_blocks, lat_size * lon_size, 3)
    # print(f'sate_sss: {sate_sss.shape}, sate_sss_coords: {sate_sss_coords.shape}')

    # %% Satellite SLA
    # sate_sla_dir = '/groups/hxm_group/home/share/xiangyanfei/OCEAN_DATA/01_process_data/SSH/SEALEVEL_GLO_PHY_L3_MY_008_062/cmems_obs-sl_glo_phy-ssh_my_j2_j3-l3-duacs_PT1S_202411'
    # sla_file = os.path.join(sate_sla_dir, str(sate_date_obj.year), f'{sate_date_str}.nc')
    # ds = read_and_convert_lon_to_360(sla_file)
    # lats = ds.coords['latitude'].values
    # lons = ds.coords['longitude'].values
    # print('lats: ', lats)
    # print('lons: ', lons)
    # sate_sla, sate_sla_coords = extract_patch_from_coordinates_v1(
    #     np.expand_dims(ds['sla_unfiltered'].values, axis=0),
    #     lats, lons,
    #     target_center_coords,
    #     (24, 24),
    # )
    # sate_sla = np.transpose(sate_sla, (0, 2, 3, 1))
    # n_blocks, lat_size, lon_size, channel_num = sate_sla.shape
    # sate_sla = sate_sla.reshape(n_blocks, lat_size * lon_size, channel_num)
    # sate_sla_coords = sate_sla_coords.reshape(n_blocks, lat_size * lon_size, 3)
    # print(f'sate_sla: {sate_sla.shape}, sate_sla_coords: {sate_sla_coords.shape}')

    # %% Satellite SST
    # sate_sst_dir='/groups/hxm_group/home/share/xiangyanfei/OCEAN_DATA/00_raw_data/SST/AVHRR_Pathfinder/'
    # sst_files = filter_AVHRR_pathfinder_files_by_time(sate_sst_dir, sate_date_obj, time_obj)
    # print(f'sst_files: {sst_files}')
    # sate_sst, lats, lons = read_multi_nc_files(sst_files, 'sea_surface_temperature')
    # print(f'lats: {lats}')
    # print(f'lons: {lons}')
    # print(f'sate_sst: {sate_sst.shape}')
    # sate_sst, sate_sst_coords = extract_patch_from_coordinates_v1(
    #     sate_sst,
    #     lats, lons,
    #     target_center_coords,
    #     (144, 144),
    # )
    # sate_sst = np.transpose(sate_sst, (0, 2, 3, 1))
    # n_blocks, lat_size, lon_size, channel_num = sate_sst.shape
    # sate_sst = sate_sst.reshape(n_blocks, lat_size * lon_size, channel_num)
    # sate_sst_coords = sate_sst_coords.reshape(n_blocks, lat_size * lon_size, 3)
    # print(f'sate_sst: {sate_sst.shape}, sate_sst_coords: {sate_sst_coords.shape}')

    # %% Read sea ice concentration 
    # sate_sic_dir = '/groups/hxm_group/home/share/xiangyanfei/OCEAN_DATA/00_raw_data/SIC/G02202_V4'
    # sea_ice_file = os.path.join(sate_sic_dir, str(sate_date_obj.year), f'{sate_date_str}.nc')
    # ds = read_and_convert_lon_to_360(sea_ice_file, 'lon')
    # lats = ds.coords['lat'].values,
    # lons = ds.coords['lon'].values,
    # print(f'lats: {lats}')
    # print(f'lons: {lons}')
    # sate_sic, sate_sic_coords = extract_patch_from_coordinates_v1(
    #     np.expand_dims(ds['data'].values, axis=0),
    #     ds.coords['lat'].values,
    #     ds.coords['lon'].values,
    #     target_center_coords,
    #     (6, 6),
    # )
    # sate_sic = np.transpose(sate_sic, (0, 2, 3, 1))
    # n_blocks, lat_size, lon_size, channel_num = sate_sic.shape
    # sate_sic = sate_sic.reshape(n_blocks, lat_size * lon_size, channel_num)
    # sate_sic_coords = sate_sic_coords.reshape(n_blocks, lat_size * lon_size, 3)
    # print(f'sate_sic: {sate_sic.shape}, sate_sic_coords: {sate_sic_coords.shape}')

    # In-situ observations
    in_situ_obs_dir = '/groups/hxm_group/home/share/xiangyanfei/OCEAN_DATA/00_raw_data/hadiod1200'
    in_situ_obs_file = os.path.join(in_situ_obs_dir, str(sate_date_obj.year), f'{sate_date_str}.csv')
    in_situ_obs, in_situ_obs_coords, num_obs_in_blocks = filter_in_situ_by_coords_and_ranges_v1(
        in_situ_obs_file,
        target_center_coords,
        (6, 6, 500),
        1000,
    )
    print(f'in_situ_obs: {in_situ_obs.shape}, in_situ_obs_coords: {in_situ_obs_coords.shape}')
    print(f'num_obs_in_blocks: {num_obs_in_blocks}')

    # visualize the number of observations in blocks
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(num_obs_in_blocks, bins=10, color='blue', alpha=0.7, edgecolor='black', label='Histogram')
    ax.set_title("Number of observations in blocks (6, 6, 500)", fontsize=16)
    ax.set_xlabel("Value", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.legend()
    fig.savefig("./imgs/number_of_observations_in_blocks.png", dpi=300, bbox_inches='tight')







