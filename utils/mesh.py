import numpy as np
from icecream import ic

def generate_coordinates(min_lon, max_lon, lon_res, min_lat, max_lat, lat_res, depths):
    """
    生成经纬度和深度的三维坐标网格。

    参数:
    - min_lon, max_lon: 经度范围
    - lon_res: 经度分辨率
    - min_lat, max_lat: 纬度范围
    - lat_res: 纬度分辨率
    - depths: 深度列表
    
    返回:
    - coords: 形状为 [num_points, 3] 的数组，每行是 (lon, lat, depth)
    """
    # 生成经度和纬度范围
    lons = np.arange(min_lon, max_lon + lon_res, lon_res)  # 经度数组
    lats = np.arange(min_lat, max_lat + lat_res, lat_res)  # 纬度数组
    ic(lons.shape, lats.shape)

    # 使用 meshgrid 生成经纬度网格
    lat_grid, lon_grid, depth_grid = np.meshgrid(lats, lons, depths, indexing='ij')  # [len(lats), len(lons)] 的网格
    ic(lon_grid.shape)

    coords = np.stack((lat_grid, lon_grid, depth_grid), axis=-1)

    return coords, lats, lons


def slice_coordinates(
    min_lon,
    max_lon,
    min_lat,
    max_lat,
    depths,
    lon_res,
    lat_res,
    lat_block,
    lon_block,
    depth_block,
    lat_step,
    lon_step,
    depth_step):

    # 生成坐标
    coords, lats, lons = generate_coordinates(min_lon, max_lon, lon_res, min_lat, max_lat, lat_res, depths)
    ic(coords.shape)
    print(f'coords: {np.min(coords[:,:,:,0])}~{np.max(coords[:,:,:,0])}')
    print(f'coords: {np.min(coords[:,:,:,1])}~{np.max(coords[:,:,:,1])}')
    print(f'coords: {np.min(coords[:,:,:,2])}~{np.max(coords[:,:,:,2])}')

    lat_size, lon_size, depth_size, var_size = coords.shape

    # Sliding window slicing
    center_coords = []
    block_coords = []
    for lat_start in range(0, lat_size - lat_block + 1, lat_step):
        for lon_start in range(0, lon_size - lon_block + 1, lon_step):
            for depth_start in range(0, depth_size - depth_block + 1, depth_step):

                cs = (
                    lats[lat_start + lat_block // 2],
                    lons[lon_start + lon_block // 2 ],
                    depths[depth_start + depth_block // 2])
                center_coords.append(cs)

                chunk = coords[
                    lat_start:lat_start + lat_block,       # Latitude slice
                    lon_start:lon_start + lon_block,       # Longitude slice
                    depth_start:depth_start + depth_block, # Depth slice
                    :                                      # Variable dimension (unchanged)
                ]

                # Generate all coordinates for this block
                # lat_coords = lats[lat_start:lat_start + lat_block]
                # lon_coords = lons[lon_start:lon_start + lon_block]
                # if len(depths) == 1:
                #     depth_coords = depths
                # else:
                #     depth_coords = depths[depth_start: depth_start + depth_block]
                
                # # Create a grid of coordinates
                # grid_lat, grid_lon, grid_depth = np.meshgrid(
                #     lat_coords, lon_coords, depth_coords, indexing='ij')
                # blocks_coords_ = np.stack(
                #     (grid_lat, grid_lon, grid_depth),
                #     axis=-1)  # Shape: [lat_block, lon_block, depth_block, 3]
                block_coords.append(chunk)
            
    block_coords = np.array(block_coords)
    center_coords = np.array(center_coords)

    # return coords, center_coords, block_coords, lat_size, lon_size, depth_size
    return coords, center_coords, block_coords, lats, lons, depths




if __name__ == '__main__':

    from block import reconstruct_array

    lon_res = 0.5                  # 经度分辨率
    lat_res = 0.5                  # 纬度分辨率
    min_lon, max_lon = 0, 360-lon_res       # 经度范围
    min_lat, max_lat = -90, 90-lat_res      # 纬度范围
    depths = [0]                    # 深度列表
    lat_block = 10
    lon_block = 20
    depth_block = 1 
    lat_step = 5
    lon_step = 5
    depth_step = 5

    coords, center_coords, block_coords, lats, lons, depths = slice_coordinates(
        min_lon, max_lon,
        min_lat, max_lat,
        depths,
        lon_res, lat_res,
        lat_block, lon_block, depth_block,
        lat_step, lon_step, depth_step)
    lat_size, lon_size, depth_size, var_size = coords.shape
    ic(coords.shape, center_coords.shape, block_coords.shape, lat_size, lon_size, depth_size)
    print(f'block_coords: {np.min(block_coords[:,:,:,:,0])}~{np.max(block_coords[:,:,:,:,0])}')
    print(f'block_coords: {np.min(block_coords[:,:,:,:,1])}~{np.max(block_coords[:,:,:,:,1])}')
    print(f'block_coords: {np.min(block_coords[:,:,:,:,2])}~{np.max(block_coords[:,:,:,:,2])}')
    
    # reconstructed_array: (lat_size, lon_size, depth_size, var_size)
    reconstructed_coords = reconstruct_array(
        block_coords,
        lat_size, lon_size, depth_size, var_size, 
        lat_block, lon_block, depth_block, 
        lat_step, lon_step, depth_step)
    ic(reconstructed_coords.shape)
    ic(np.mean(reconstructed_coords))
    
    bias = coords - reconstructed_coords
    ic(np.min(bias), np.max(bias))
    exit()

    for i in np.arange(coords.shape[0]): # lat
        for j in np.arange(coords.shape[1]): # lon
            for k in np.arange(coords.shape[2]): # depth
                for n in np.arange(coords.shape[3]): # var
                    if coords[i,j,k,n] != reconstructed_coords[i,j,k,n]:
                        # ic(i,j,k,n, array[i,j,k,n], reconstructed_array[i,j,k,n])
                        print(f'{i},{j},{k},{n}: {coords[i,j,k,n]}, {reconstructed_coords[i,j,k,n]}')


    
