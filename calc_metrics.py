import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
from icecream import ic
from datetime import datetime, timedelta
from utils.read import read_glorys_file_v2, read_foracet_file
from utils.plot import plot_2d_map_multi_sub_figure
from utils.metrics import calculate_corr, calculate_rmse, calculate_acc


def read_filenames(file_path):
    with open(file_path, 'r') as file:
        filenames = [line.strip() for line in file if line.strip()]
    return filenames


def read_mask(mask_file):

    mask_ds = xr.open_dataset(mask_file)
    land_sea_mask = mask_ds['land_sea_mask'].values[0, :180]  # surface
    return np.expand_dims(land_sea_mask, axis=(2, 3))  # Expand dimensions

def read_data(file_time_obj, target_dir, bg_dir, clima_dir, land_sea_mask, bg_lead_time, variables):

    # Target Data
    target_file = os.path.join(target_dir, str(file_time_obj.year), f"{file_time_obj.strftime('%Y%m%d')}.h5")
    target_data, lats, lons, depths, _ = read_glorys_file_v2(
        target_file, variables, fill_land=False, only_surface=True)
    target_data[np.broadcast_to(land_sea_mask, target_data.shape) == 1] = np.nan

    # Background Data
    bg_time = file_time_obj - timedelta(days=bg_lead_time)
    bg_file_path = os.path.join(bg_dir, bg_time.strftime("%Y%m%d") + '.nc')
    bg_data, _, _, _, _ = read_foracet_file(bg_file_path, bg_lead_time, fill_land=True, variables=variables)
    bg_data[np.broadcast_to(land_sea_mask, bg_data.shape) == 1] = np.nan

    # Climatology Data
    day_of_year = file_time_obj.timetuple().tm_yday
    clima_file = os.path.join(clima_dir, f"day_{day_of_year}.h5")
    clima_data, _, _, _, _ = read_glorys_file_v2(clima_file, variables, fill_land=False, only_surface=True)
    clima_data[np.broadcast_to(land_sea_mask, clima_data.shape) == 1] = np.nan

    return target_data, bg_data, clima_data, lats, lons


def calculate_metrics_and_store(pred, target, bg, clima, var, metrics_df, file_time_obj):

    corr = calculate_corr(pred=pred, target=target)
    corr_bg = calculate_corr(pred=bg, target=target)
    corr_clima = calculate_corr(pred=clima, target=target)

    rmse = calculate_rmse(pred=pred, target=target, latitude_weighted=False)[0]
    rmse_bg = calculate_rmse(pred=bg, target=target, latitude_weighted=False)[0]
    rmse_clima = calculate_rmse(pred=clima, target=target, latitude_weighted=False)[0]

    weighted_rmse = calculate_rmse(pred=pred, target=target, latitude_weighted=True)[0]
    weighted_rmse_bg = calculate_rmse(pred=bg, target=target, latitude_weighted=True)[0]
    weighted_rmse_clima = calculate_rmse(pred=clima, target=target, latitude_weighted=True)[0]

    acc = calculate_acc(pred=pred, target=target, clima=clima, latitude_weighted=False)[0]
    acc_bg = calculate_acc(pred=bg, target=target, clima=clima, latitude_weighted=False)[0]

    weighted_acc = calculate_acc(pred=pred, target=target, clima=clima, latitude_weighted=True)[0]
    weighted_acc_bg = calculate_acc(pred=bg, target=target, clima=clima, latitude_weighted=True)[0]

    # Append to DataFrame
    new_row = {
        "time": file_time_obj, "variable": var,
        "rmse": rmse, "weighted_rmse": weighted_rmse,
        "rmse_bg": rmse_bg, "weighted_rmse_bg": weighted_rmse_bg,
        "rmse_clima": rmse_clima, "weighted_rmse_clima": weighted_rmse_clima,
        "acc": acc, "weighted_acc": weighted_acc,
        "acc_bg": acc_bg, "weighted_acc_bg": weighted_acc_bg,
        "corr": corr, "corr_bg": corr_bg, "corr_clima": corr_clima,
    }
    return metrics_df.append(new_row, ignore_index=True)


def main():
    # Paths and configurations
    file_path = "./data/time_series.txt"
    filenames = read_filenames(file_path)
    target_dir = './data/glorys'
    clima_dir = '/data/glorys_clima_daily'
    bg_dir = './data/bg_forecast'
    mask_file = './mask/land_sea_mask_glorys_23_181_360.nc'
    bg_lead_time = 4
    variables = ['T', 'S', 'U', 'V', 'SSH']
    inf_dir = './exps/inference/' # paths for inference files 
    inf_img_dir = os.path.join(inf_dir, 'imgs')
    inf_metric_dir = os.path.join(inf_dir, 'metrics')

    os.makedirs(inf_img_dir, exist_ok=True)
    os.makedirs(inf_metric_dir, exist_ok=True)

    # Initialize DataFrame for metrics
    metrics_df = pd.DataFrame(columns=[
        "time", "variable",
        "rmse", "weighted_rmse",
        "rmse_bg", "weighted_rmse_bg",
        "rmse_clima", "weighted_rmse_clima",
        "acc", "weighted_acc",
        "acc_bg", "weighted_acc_bg",
        "corr", "corr_bg", "corr_clima",
    ])

    # Load mask
    land_sea_mask = read_mask(mask_file)

    # Process each file
    for filename in filenames:
        file_time_obj = datetime.strptime(filename.split('.')[0], '%Y%m%d')

        # Read data
        target_data, bg_data, clima_data, lats, lons = read_data(
            file_time_obj, target_dir, bg_dir, clima_dir, land_sea_mask, bg_lead_time, variables)

        # Read predictions
        pred_file = os.path.join(inf_dir, filename)
        pred_ds = xr.open_dataset(pred_file)

        for var_idx, var in enumerate(variables):
            bg = bg_data[:, :, :, var_idx]
            target = target_data[:, :, :, var_idx]
            clima = clima_data[:, :, :, var_idx]
            pred = pred_ds['prediction'].sel(variable=var).values

            # Apply mask
            pred[np.broadcast_to(land_sea_mask[:, :, 0], pred.shape)] = np.nan

            # Metric calculation
            metrics_df = calculate_metrics_and_store(pred, target, bg, clima, var, metrics_df, file_time_obj)

    # Save metrics
    metric_file = os.path.join(inf_metric_dir, 'metrics_all.csv')
    metrics_df.to_csv(metric_file, index=False)
    print(f"Metrics saved to {metric_file}")

    # Print summary
    grouped = metrics_df.groupby("variable")[["weighted_rmse", "weighted_rmse_bg", "weighted_acc", "weighted_acc_bg"]].mean()
    print(grouped)


if __name__ == "__main__":
    main()
