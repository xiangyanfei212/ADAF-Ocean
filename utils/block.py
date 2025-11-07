import numpy as np
from icecream import ic

def slice_array(
        array, 
        lats, lons, depths, 
        lat_block, lon_block, depth_block, 
        lat_step, lon_step, depth_step):
    """
    Perform sliding window slicing on a 4D array.

    Parameters:
    - array: np.ndarray
        The 4D array to slice, with shape [lat_size, lon_size, depth_size, var_size].
    - lat_block: int
        Block size in the latitude direction.
    - lon_block: int
        Block size in the longitude direction.
    - depth_block: int
        Block size in the depth dimension.
    - lat_step: int
        Step size for the latitude direction.
    - lon_step: int
        Step size for the longitude direction.
    - depth_step: int
        Step size for the depth dimension.

    Returns:
    - np.ndarray
        The sliced array, with each block shaped as [lat_block, lon_block, depth_block, var_size].
    """
    # Get the shape of the input array
    lat_size, lon_size, depth_size, var_size = array.shape

    # Initialize an empty list to store the blocks
    chunks = []
    coords = []
    center_coords = []

    # Sliding window slicing
    for lat_start in range(0, lat_size - lat_block + 1, lat_step):
        for lon_start in range(0, lon_size - lon_block + 1, lon_step):
            for depth_start in range(0, depth_size - depth_block + 1, depth_step):
                # ic(lat_start, lon_start, depth_start)  # Debug: print start indices

                cs = (
                    lats[lat_start + lat_block // 2],
                    lons[lon_start + lon_block // 2 ],
                    depths[depth_start + depth_block // 2])

                # Extract the current block
                chunk = array[
                    lat_start:lat_start + lat_block,                # Latitude slice
                    lon_start:lon_start + lon_block,                # Longitude slice
                    depth_start:depth_start + depth_block, # Pressure slice
                    :                                               # Variable dimension (unchanged)
                ]
                # ic(chunk.shape)  # Debug: print the shape of the current block

                # Validate the shape of the sliced block
                assert chunk.shape == (lat_block, lon_block, depth_block, var_size), "Wrong shape"

                # Generate all coordinates for this block
                lat_coords = lats[lat_start:lat_start + lat_block]
                lon_coords = lons[lon_start:lon_start + lon_block]
                depth_coords = depths[depth_start: depth_start + depth_block]
                
                # Create a grid of coordinates
                grid_lat, grid_lon, grid_depth = np.meshgrid(lat_coords, lon_coords, depth_coords, indexing='ij')
                chunck_coords = np.stack((grid_lat, grid_lon, grid_depth), axis=-1)  # Shape: [lat_block, lon_block, depth_block, 3]

                chunks.append(chunk)
                center_coords.append(cs)
                coords.append(chunck_coords)

    # Convert the list of blocks to a NumPy array and return
    return np.array(chunks), np.array(center_coords), np.array(coords)


def reconstruct_array(chunks,
                      lat_size, lon_size, depth_size, var_size, 
                      lat_block, lon_block, depth_block, 
                      lat_step, lon_step, depth_step):
    """
    Reconstruct the original array from sliced chunks using sliding window aggregation.

    Parameters:
    - chunks: np.ndarray
        The sliced chunks, with shape [num_chunks, lat_block, lon_block, depth_block, var_size].
    - lat_size: int
        Original size in the latitude dimension.
    - lon_size: int
        Original size in the longitude dimension.
    - depth_size: int
        Original size in the depth dimension.
    - var_size: int
        Number of variables in the array (unchanged during reconstruction).
    - lat_block: int
        Block size in the latitude direction.
    - lon_block: int
        Block size in the longitude direction.
    - depth_block: int
        Block size in the depth dimension.
    - lat_step: int
        Step size in the latitude direction.
    - lon_step: int
        Step size in the longitude direction.
    - depth_step: int
        Step size in the depth dimension.

    Returns:
    - reconstructed_array: np.ndarray
        The reconstructed original array with shape [lat_size, lon_size, depth_size, var_size].
    """
    # Initialize the reconstructed array and weight array
    reconstructed_array = np.zeros((lat_size, lon_size, depth_size, var_size))
    print(f'reconstructed_array: {reconstructed_array.shape}')
    weight_array = np.zeros((lat_size, lon_size, depth_size, var_size))  # For handling overlaps

    # Reconstruct the original array using sliding window
    chunk_index = 0
    for lat_start in range(0, lat_size - lat_block + 1, lat_step):
        for lon_start in range(0, lon_size - lon_block + 1, lon_step):
            for depth_start in range(0, depth_size - depth_block + 1, depth_step):
                # Add the current chunk to the reconstructed array
                reconstructed_array[
                    lat_start:lat_start + lat_block,
                    lon_start:lon_start + lon_block,
                    depth_start:depth_start + depth_block,
                    :
                ] += chunks[chunk_index]

                # Update the weight array
                weight_array[
                    lat_start:lat_start + lat_block,
                    lon_start:lon_start + lon_block,
                    depth_start:depth_start + depth_block,
                    :
                ] += 1

                chunk_index += 1

    # Handle overlapping regions by taking the mean
    with np.errstate(divide='ignore', invalid='ignore'):
        reconstructed_array = np.divide(reconstructed_array, weight_array, where=(weight_array > 0))
        reconstructed_array[weight_array == 0] = 0  # Fill uncovered regions with 0

    # Debug: Check for uncovered regions
    # ic(np.where(weight_array == 0))

    return reconstructed_array


if __name__ == '__main__':

    # %% 2D
    array = np.random.rand(720, 1440, 1, 5)
    lat_size, lon_size, depth_size, var_size = array.shape 

    latitudes = np.arange(-90, 90, 0.25)
    longitudes = np.arange(0, 360, 0.25)
    depths = [0]

    lat_block = 10   # 纬度方向每块大小
    lon_block = 20   # 经度方向每块大小
    depth_block = 1  # 深度方向每块大小

    lat_step = 5
    lon_step = 5
    depth_step = 1

    depth_list = np.arange(0, 1 - depth_block + 1, depth_step)
    lat_list = np.arange(0, lat_size - lat_block + 1, lat_step)
    lon_list = np.arange(0, lon_size - lon_block + 1, lat_step)
    ic(depth_list, lat_list, lon_list)

    chunks, center_coords, coords = slice_array(
        array,
        latitudes, longitudes, depths,
        lat_block, lon_block, depth_block, 
        lat_step, lon_step, depth_step)
    ic(chunks.shape, center_coords.shape, coords.shape)

    reconstructed_array = reconstruct_array(
        chunks,
        lat_size, lon_size, depth_size, var_size, 
        lat_block, lon_block, depth_block, 
        lat_step, lon_step, depth_step)
    ic(reconstructed_array.shape)

    bias = array - reconstructed_array
    ic(np.min(bias), np.max(bias), np.mean(bias))
    exit()
    
    for i in np.arange(array.shape[0]):
        for j in np.arange(array.shape[1]):
            for k in np.arange(array.shape[2]):
                for n in np.arange(array.shape[3]):
                    if array[i,j,k,n] != reconstructed_array[i,j,k,n]:
                        # ic(i,j,k,n, array[i,j,k,n], reconstructed_array[i,j,k,n])
                        print(f'{i},{j},{k},{n}: {array[i,j,k,n]}, {reconstructed_array[i,j,k,n]}')
                        # assert array[i,j,k,:].any() == reconstructed_array[i,j,k,:].any(), f"i={i},j={j},k={k}, Not match"
    
    exit()

    # %% 2D
    array = np.random.rand(180, 360, 1, 5)
    latitudes = np.arange(-90, 90, 1)
    longitudes = np.arange(0, 360, 1)
    depths = [0]

    lat_block = 40   # 纬度方向每块大小
    lon_block = 60   # 经度方向每块大小
    depth_block = 1  # 深度方向每块大小

    lat_step = 20
    lon_step = 50
    depth_step = 1

    depth_list = np.arange(0, 1 - depth_block + 1, depth_step)
    lat_list = np.arange(0, 180 - lat_block + 1, lat_step)
    lon_list = np.arange(0, 360 - lon_block + 1, lat_step)
    ic(depth_list, lat_list, lon_list)

    chunks, center_coords, coords = slice_array(
        array,
        latitudes, longitudes, depths,
        lat_block, lon_block, depth_block, 
        lat_step, lon_step, depth_step)
    ic(chunks.shape, center_coords.shape, coords.shape)

    lat_size, lon_size, depth_size, var_size = 180, 360, 1, 5
    reconstructed_array = reconstruct_array(
        chunks,
        lat_size, lon_size, depth_size, var_size, 
        lat_block, lon_block, depth_block, 
        lat_step, lon_step, depth_step)
    ic(reconstructed_array.shape)

    bias = array - reconstructed_array
    ic(np.min(bias), np.max(bias), np.mean(bias))
    
    for i in np.arange(array.shape[0]):
        for j in np.arange(array.shape[1]):
            for k in np.arange(array.shape[2]):
                for n in np.arange(array.shape[3]):
                    if array[i,j,k,n] != reconstructed_array[i,j,k,n]:
                        # ic(i,j,k,n, array[i,j,k,n], reconstructed_array[i,j,k,n])
                        print(f'{i},{j},{k},{n}: {array[i,j,k,n]}, {reconstructed_array[i,j,k,n]}')
                        # assert array[i,j,k,:].any() == reconstructed_array[i,j,k,:].any(), f"i={i},j={j},k={k}, Not match"
    
    exit()
    
    
    
    


    # %% 3D
    array = np.random.rand(180, 360, 23, 5)

    latitudes = np.arange(-90, 90, 1)
    longitudes = np.arange(0, 360, 1)
    depths = [0, 2, 5, 7, 11, 15, 21, 29, 40, 55, 77, 92, 109, 130, 155, 186, 222, 266, 318, 380, 453, 541, 643]
    ic(len(latitudes), len(longitudes), len(depths))

    # 定义切块大小和重叠大小
    lat_block = 6   # 纬度方向每块大小
    lon_block = 6   # 经度方向每块大小
    depth_block = 8  # 深度方向每块大小

    lat_step = 3
    lon_step = 3
    depth_step = 3

    depth_list = np.arange(0, 23 - depth_block + 1, depth_step)
    lat_list = np.arange(0, 180 - lat_block + 1, lat_step)
    lon_list = np.arange(0, 360 - lon_block + 1, lat_step)
    ic(depth_list, lat_list, lon_list)

    # chunks: (num_blocks, lat_block, lon_block, depth_block, num_variables)
    # coords: (num_blocks, lat_block, lon_block, depth_block, 3)
    chunks, center_coords, coords = slice_array(
        array,
        latitudes, longitudes, depths,
        lat_block, lon_block, depth_block, 
        lat_step, lon_step, depth_step)
    ic(chunks.shape, coords.shape)

    lat_size, lon_size, depth_size, var_size = 180, 360, 23, 5
    # reconstructed_array: (lat_size, lon_size, depth_size, var_size)
    reconstructed_array = reconstruct_array(
        chunks,
        lat_size, lon_size, depth_size, var_size, 
        lat_block, lon_block, depth_block, 
        lat_step, lon_step, depth_step)
    ic(reconstructed_array.shape)

    bias = array - reconstructed_array
    ic(np.min(bias), np.max(bias))
    ic(np.where(bias!=0)[0])
    ic(np.where(bias!=0)[1])
    ic(np.min(np.where(bias!=0)[2]), np.max(np.where(bias!=0)[2]))
    exit()

    for i in np.arange(array.shape[0]):
        for j in np.arange(array.shape[1]):
            for k in np.arange(array.shape[2]):
                for n in np.arange(array.shape[3]):
                    if array[i,j,k,n] != reconstructed_array[i,j,k,n]:
                        # ic(i,j,k,n, array[i,j,k,n], reconstructed_array[i,j,k,n])
                        print(f'{i},{j},{k},{n}: {array[i,j,k,n]}, {reconstructed_array[i,j,k,n]}')
                        # assert array[i,j,k,:].any() == reconstructed_array[i,j,k,:].any(), f"i={i},j={j},k={k}, Not match"