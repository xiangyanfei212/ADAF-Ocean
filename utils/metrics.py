import numpy as np

def calculate_rmse_and_correlation(array1, array2):
    """
    计算两个数组的 RMSE 和相关系数。

    参数:
    - array1: 第一个数组，形状 (180, 360, 1)
    - array2: 第二个数组，形状 (180, 360, 1)

    返回:
    - rmse: 均方根误差
    - correlation: 相关系数
    """
    # 检查两个数组的形状是否一致
    if array1.shape != array2.shape:
        raise ValueError("两个数组的形状必须一致！")

    # 展平数组以方便计算
    array1_flat = array1.ravel()  # 将数组展平为 1D
    array2_flat = array2.ravel()

    # 创建有效数据的掩码，仅选择非 NaN 的位置
    valid_mask = ~np.isnan(array1_flat) & ~np.isnan(array2_flat)
    valid_array1 = array1_flat[valid_mask]
    valid_array2 = array2_flat[valid_mask]

    # 计算 RMSE
    mse = np.mean((valid_array1 - valid_array2) ** 2)  # 均方误差
    rmse = np.sqrt(mse)  # 均方根误差

    # 计算相关系数
    correlation = np.corrcoef(valid_array1, valid_array2)[0, 1]  # 计算相关系数矩阵并取 [0,1] 元素

    return rmse, correlation

def calculate_corr(
        pred: np.ndarray,
        target: np.ndarray
    ):

    # 检查两个数组的形状是否一致
    if pred.shape != target.shape:
        raise ValueError("两个数组的形状必须一致！")

    pred = pred.ravel() 
    target = target.ravel()

    # 创建有效数据的掩码，仅选择非 NaN 的位置
    valid_mask = ~np.isnan(pred) & ~np.isnan(target)
    pred = pred[valid_mask]
    target = target[valid_mask]

    correlation = np.corrcoef(pred, target)[0, 1]  # 计算相关系数矩阵并取 [0,1] 元素

    return correlation


def lat_np(j, num_lat):
    return 90. - j * 180./(num_lat-1)


def latitude_weighting_factor(j, num_lat, s):
    return num_lat * np.cos(3.1416/180. * lat_np(j, num_lat))/s


def calculate_rmse(
        pred: np.ndarray,
        target: np.ndarray,
        latitude_weighted: bool = False):
    """
    计算纬度加权的 RMSE。
    输入为三维数组，形状为 (lat, lon, 1)。
    返回纬度加权的 RMSE。
    """
    # 确保输入形状为 (lat, lon, 1)
    assert pred.ndim == 3 and target.ndim == 3, "Input arrays must be 3-dimensional."
    assert pred.shape == target.shape, "Prediction and array2 arrays must have the same shape."
    assert pred.shape[2] == 1, "The last dimension must be of size 1."

    if latitude_weighted:
        num_lat = pred.shape[0]  # 获取纬度维度的大小
        lat_t = np.arange(0, num_lat)  # 纬度索引
        s = np.sum(np.cos(np.pi / 180. * lat_np(lat_t, num_lat))) # 计算纬度加权因子
        weight = np.reshape(latitude_weighting_factor(lat_t, num_lat, s), (-1, 1, 1))
    else:
        weight = 1

    result = np.sqrt(np.nanmean(weight * (pred - target) ** 2., axis=(0, 1)))
    
    return result


def calculate_acc(
        pred: np.ndarray,
        target: np.ndarray,
        clima: np.ndarray,
        latitude_weighted: bool = False) -> np.ndarray:
    """
    计算 ACC (Anomaly Correlation Coefficient)，支持加权和未加权计算。
    
    参数:
        pred (np.ndarray): 预测值，形状为 (lat, lon, 1)
        target (np.ndarray): 目标值，形状为 (lat, lon, 1)
        clima (np.ndarray): 气候基准场，形状与 pred 和 target 相同，仅在加权时使用。
        weighting (bool): 是否进行纬度加权，如果为 False，则计算未加权的 ACC。

    返回:
        np.ndarray: ACC 值，形状为与输入的批量或通道维度对应。
    """

    # 确保输入形状为 (lat, lon, 1)
    assert pred.ndim == 3 and target.ndim == 3, "Input arrays must be 3-dimensional."
    assert pred.shape == target.shape, "Prediction and array2 arrays must have the same shape."
    assert pred.shape[2] == 1, "The last dimension must be of size 1."

    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    clima = np.nan_to_num(clima, nan=0.0, posinf=0.0, neginf=0.0)
    
    if latitude_weighted:
        # 确保 clima 存在
        if clima is None:
            raise ValueError("纬度加权计算需要提供 clima 参数。")

        # 去除气候基准场
        pred -= clima
        target -= clima

        # print('nan in pred and target?')
        # print(np.isnan(pred).any(), np.isnan(target).any())  # 检查 NaN
        # print(np.isinf(pred).any(), np.isinf(target).any())  # 检查 Inf
        
        # 获取纬度维度大小
        num_lat = pred.shape[0]  # 纬度维度 (lat)
        lat_t = np.arange(0, num_lat)  # 纬度索引
        
        # 计算纬度加权因子
        s = np.sum(np.cos(np.pi / 180.0 * lat_np(lat_t, num_lat)))
        weight = np.reshape(latitude_weighting_factor(lat_t, num_lat, s), (-1, 1, 1))
        print(f'weight: {weight.shape}')
        
        # 计算加权 ACC
        numerator = np.nansum(weight * pred * target, axis=(0, 1))
        denominator = np.sqrt(
            np.nansum(weight * pred * pred, axis=(0, 1)) * np.nansum(weight * target * target, axis=(0, 1))
        )
    else:
        # print(f'Is there NaN in denominator?')
        # print(np.sum(pred * pred, axis=(-1, -2)))
        # print(np.sum(target * target, axis=(-1, -2)))

        # 未加权 ACC 计算
        numerator = np.sum(pred * target, axis=(0, 1))
        denominator = np.sqrt(
            np.sum(pred * pred, axis=(0, 1)) * np.sum(target * target, axis=(0, 1))
        )
        print(f'numerator: {numerator.shape} {numerator}')
        print(f'denominator: {denominator.shape} {denominator}')

    # 避免除以 0
    result = numerator / np.where(denominator == 0, np.nan, denominator)
    print(f'result: {result}')
    return result
