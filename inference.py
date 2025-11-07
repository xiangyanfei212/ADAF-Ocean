import os
import yaml
import glob
import torch
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from icecream import ic
from str2bool import str2bool
from utils.logger import log_to_file_and_screen
from utils.block import slice_array, reconstruct_array
from datetime import datetime, timedelta
from utils.mesh import *
from utils.sample import *
from utils.normalization import *
from utils.read import *
from utils.block import *
import models
torch.cuda.empty_cache()
print(torch.cuda.memory_summary())


def read_and_norm_input(target_time, target_center_coords, config):

    # %% Read background
    bg_time = target_time - timedelta(days = config['bg_lead_time']) # spot time
    bg_file_path = os.path.join(config['bg_dir'], bg_time.strftime("%Y%m%d")+'.nc')
    print(f'Read bg from {bg_file_path}')
    if not os.path.exists(bg_file_path):
        print(f'Bg file not exists! skip this.')
        return None
    bg, lats, lons, depths, valid_time = read_foracet_file(
        bg_file_path,
        config['bg_lead_time'],
        fill_land=True,
        interp=config['bg_interp'],
        variables=config['bg_vars'],
    )
    print(f'Bg valid time: {valid_time}')
    if valid_time != target_time:
        print(f'Time not matching')
        exit()

    bg_blocks, bg_block_coords = extract_blocks_from_coordinates_v1(
        bg, lats, lons, depths,
        target_center_coords,
        config['bg_block_shape'])
    n_blocks, lat_size, lon_size, depth_size, channel_num = bg_blocks.shape
    bg_blocks = bg_blocks.reshape(n_blocks, lat_size * lon_size * depth_size, channel_num)
    bg_block_coords = bg_block_coords.reshape(n_blocks, lat_size * lon_size * depth_size, 3)

    bg_blocks = normalize(
        data=bg_blocks,
        method=config['normalization'],
        stats_file=config['bg_stats_file'], 
        vars=config['bg_vars'])

    bg_block_coords = normalize_coordinates(
        coords=bg_block_coords,
        method=config['loc_normalization'],
        stats_file=config['location_stats_file'],
        variables=['lat', 'lon', 'depth'])
    print(f'bg_blocks: {bg_blocks.shape}')
    print(f'bg_block_coords: {bg_block_coords.shape}')

    # %% observation time of satellite (current day - 1 day)
    sate_date_obj = target_time - timedelta(days = 1)
    sate_date_str = sate_date_obj.strftime('%Y%m%d')

    # %% Satellite SSW
    sate_ssw_file = os.path.join(
        config['sate_ssw_dir'], str(sate_date_obj.year), f'{sate_date_str}.nc')
    print(f'Read satellite SSW from {sate_ssw_file}')
    ds = xr.open_dataset(sate_ssw_file)
    sate_ssw_wind_dir = ds['wind_dir'].values
    sate_ssw_wind_speed = ds['wind_speed'].values
    lats = ds.coords['latitude'].values
    lons = ds.coords['longitude'].values
    ds.close()
    sate_ssw = np.stack([sate_ssw_wind_dir, sate_ssw_wind_speed], axis=0)
    sate_ssw, sate_ssw_coords = extract_patch_from_coordinates_v1(
        np.squeeze(sate_ssw),
        lats, lons,
        target_center_coords,
        config['sate_ssw_patch_shape'],
    )
    sate_ssw = np.transpose(sate_ssw, (0, 2, 3, 1))
    n_blocks, lat_size, lon_size, channel_num = sate_ssw.shape
    sate_ssw = sate_ssw.reshape(n_blocks, lat_size * lon_size, channel_num)
    sate_ssw_coords = sate_ssw_coords.reshape(n_blocks, lat_size * lon_size, 3)
    sate_ssw = normalize(
        data=sate_ssw,
        method=config['normalization'],
        stats_file=config['sate_ssw_stats_file'], 
        vars=config['sate_ssw_vars'])

    sate_ssw_coords = normalize_coordinates(
        coords=sate_ssw_coords,
        method=config['loc_normalization'],
        stats_file=config['location_stats_file'],
        variables=['lat', 'lon', 'depth'])
    print(f'sate_ssw_coords: {sate_ssw_coords.shape}')
    print(f'sate_ssw: {sate_ssw.shape}')

    # %% Satellite SSS
    sate_sss_file = os.path.join(
        config['sate_sss_dir'], str(sate_date_obj.year), f'{sate_date_str}.nc')
    print(f'Read satellite SSS from {sate_sss_file}')
    ds = xr.open_dataset(sate_sss_file)
    sate_sss = np.expand_dims(ds[config['sate_sss_vars'][0]].values, axis=0)
    lats = ds.coords['latitude'].values
    lons = ds.coords['longitude'].values
    ds.close()

    sate_sss = sate_sss_qc(
            data=sate_sss,
            stats_file=config['sate_sss_stats_file'], 
            channel_names=config['sate_sss_vars'])
    
    sate_sss, sate_sss_coords = extract_patch_from_coordinates_v1(
        sate_sss,
        lats, lons,
        target_center_coords,
        config['sate_sss_patch_shape'],
    )
    sate_sss = np.transpose(sate_sss, (0, 2, 3, 1))
    n_blocks, lat_size, lon_size, channel_num = sate_sss.shape
    sate_sss = sate_sss.reshape(n_blocks, lat_size * lon_size, channel_num)
    sate_sss_coords = sate_sss_coords.reshape(n_blocks, lat_size * lon_size, 3)

    sate_sss = normalize(
        data=sate_sss,
        method=config['normalization'],
        stats_file=config['sate_sss_stats_file'], 
        vars=config['sate_sss_vars'])

    sate_sss_coords = normalize_coordinates(
        coords=sate_sss_coords,
        method=config['loc_normalization'],
        stats_file=config['location_stats_file'],
        variables=['lat', 'lon', 'depth'])
    print(f'sate_sss_coords: {sate_sss_coords.shape}')
    print(f'sate_sss: {sate_sss.shape}')

    # %% Read satellite SLA
    sate_sla_file = os.path.join(
        config['sate_sla_dir'], str(sate_date_obj.year), f'{sate_date_str}.nc')
    print(f'Read satellite SLA from {sate_sla_file}')

    if not os.path.exists(sate_sla_file):
        print(f'SSH satellite file not exists! skip this.')
        return None
    ds = xr.open_dataset(sate_sla_file)
    sate_sla = np.expand_dims(ds[config['sate_sla_vars'][0]].values, axis=0)
    # ic(sate_sla.shape)
    lats = ds.coords['lat'].values
    lons = ds.coords['lon'].values
    ds.close()
    sate_sla, sate_sla_coords = extract_patch_from_coordinates_v1(
        sate_sla,
        lats, lons,
        target_center_coords,
        config['sate_sla_patch_shape'],
    )
    sate_sla = np.transpose(sate_sla, (0, 2, 3, 1))
    n_blocks, lat_size, lon_size, channel_num = sate_sla.shape
    sate_sla = sate_sla.reshape(n_blocks, lat_size * lon_size, channel_num)
    sate_sla_coords = sate_sla_coords.reshape(n_blocks, lat_size * lon_size, 3)

    sate_sla = normalize(
        data=sate_sla,
        method=config['normalization'],
        stats_file=config['sate_sla_stats_file'], 
        vars=config['sate_sla_vars'])

    sate_sla_coords = normalize_coordinates(
        coords=sate_sla_coords,
        method=config['loc_normalization'],
        stats_file=config['location_stats_file'],
        variables=['lat', 'lon', 'depth'])
    print(f'sate_sla_coords: {sate_sla_coords.shape}')
    print(f'sate_sla: {sate_sla.shape}')

    # %% Read satellite's SST
    sate_sst_files = filter_AVHRR_pathfinder_files_by_time(
        config['sate_sst_dir'], sate_date_obj, target_time)
    print(f'Read satellite SST from {sate_sst_files}')
    if len(sate_sst_files) != 2:
        print(f'sate_sst_files is missing, skip')
        return None
    ds_0 = xr.open_dataset(sate_sst_files[0])
    sate_sst_0 = ds_0[config['sate_sst_vars'][0]].values
    lons = ds_0.coords['lon'].values
    lats = ds_0.coords['lat'].values
    ds_0.close()
    ds_1 = xr.open_dataset(sate_sst_files[1])
    sate_sst_1 = ds_1[config['sate_sst_vars'][0]].values
    ds_1.close()
    sate_sst = np.stack([sate_sst_0, sate_sst_1])
    sate_sst, sate_sst_coords = extract_patch_from_coordinates_v1(
        sate_sst,
        lats, lons,
        target_center_coords,
        config['sate_sst_patch_shape'],
    )
    sate_sst = np.transpose(sate_sst, (0, 2, 3, 1))
    n_blocks, lat_size, lon_size, channel_num = sate_sst.shape
    sate_sst = sate_sst.reshape(n_blocks, lat_size * lon_size, channel_num)
    sate_sst_coords = sate_sst_coords.reshape(n_blocks, lat_size * lon_size, 3)
    # sate_sst = sate_sst[:,:1000000,:]
    # sate_sst_coords = sate_sst_coords[:,:1000000,:]

    sate_sst = normalize(
        data=sate_sst,
        method=config['normalization'],
        stats_file=config['sate_sst_stats_file'], 
        vars=config['sate_sst_vars'])

    sate_sst_coords = normalize_coordinates(
        coords=sate_sst_coords,
        method=config['loc_normalization'],
        stats_file=config['location_stats_file'],
        variables=['lat', 'lon', 'depth'])
    print(f'sate_sst_coords: {sate_sst_coords.shape}')
    print(f'sate_sst: {sate_sst.shape}')
    
    # %% Read sea ice concentration
    sate_sic_file = os.path.join(
        config['sate_sic_dir'], str(sate_date_obj.year), f'{sate_date_str}.nc')
    print(f'Read satellite SIC from {sate_sic_file}')
    ds = xr.open_dataset(sate_sic_file)
    sate_sic = np.expand_dims(ds[config['sate_sic_vars'][0]].values, axis=0)
    lats = ds.coords['latitude'].values
    lons = ds.coords['longitude'].values
    ds.close()
    sate_sic, sate_sic_coords = extract_patch_from_coordinates_v1(
        sate_sic,
        lats, lons,
        target_center_coords,
        config['sate_sic_patch_shape'],
    )
    sate_sic = np.transpose(sate_sic, (0, 2, 3, 1))
    n_blocks, lat_size, lon_size, channel_num = sate_sic.shape
    sate_sic = sate_sic.reshape(n_blocks, lat_size * lon_size, channel_num)
    sate_sic_coords = sate_sic_coords.reshape(n_blocks, lat_size * lon_size, 3)

    sate_sic = normalize(
        data=sate_sic,
        method=config['normalization'],
        stats_file=config['sate_sic_stats_file'],
        vars=config['sate_sic_vars'])

    sate_sic_coords = normalize_coordinates(
        coords=sate_sic_coords,
        method=config['loc_normalization'],
        stats_file=config['location_stats_file'],
        variables=['lat', 'lon', 'depth'])
    print(f'sate_sic_coords: {sate_sic_coords.shape}')
    print(f'sate_sic: {sate_sic.shape}')

    # %% Read in situ observations
    in_situ_obs_file = os.path.join(
        config['in_situ_obs_dir'], str(sate_date_obj.year), f'{sate_date_str}.csv')
    print(f'Read in-situ observation from {in_situ_obs_file}')
    in_situ_obs, in_situ_obs_coords, _ = filter_in_situ_by_coords_and_ranges_v1(
        in_situ_obs_file,
        target_center_coords,
        config['in_situ_obs_patch_range'],
        config['in_situ_obs_pad'],
    )

    # QC check
    in_situ_obs = in_situ_obs_qc(
        data = in_situ_obs,
        stats_file = config['in_situ_obs_stats_file'],
        channel_names = config['in_situ_obs_vars'],
    )

    in_situ_obs = normalize(
        data=in_situ_obs,
        method=config['normalization'],
        stats_file=config['in_situ_obs_stats_file'],
        vars=config['in_situ_obs_vars'])

    in_situ_obs_coords = normalize_coordinates(
        coords=in_situ_obs_coords,
        method=config['loc_normalization'],
        stats_file=config['location_stats_file'],
        variables=['lat', 'lon', 'depth'])
    print(f'in_situ_obs_coords: {in_situ_obs_coords.shape}')
    print(f'in_situ_obs: {in_situ_obs.shape}')

    return {
        'bg_value': bg_blocks,
        'bg_coords': bg_block_coords,
        'sate_sss_value': sate_sss,
        'sate_sss_coords': sate_sss_coords,
        'sate_sst_value': sate_sst,
        'sate_sst_coords': sate_sst_coords,
        'sate_ssw_value': sate_ssw,
        'sate_ssw_coords': sate_ssw_coords,
        'sate_sla_value': sate_sla,
        'sate_sla_coords': sate_sla_coords,
        'sate_sic_value': sate_sic,
        'sate_sic_coords': sate_sic_coords,
        'in_situ_obs': in_situ_obs,
        'in_situ_obs_coords': in_situ_obs_coords,
    }


def save_to_nc(
        prediction,
        filename:str, 
        latitudes:list,
        longitudes:list,
        depths:list,
        variables:list
    ):

    lat_size, lon_size, depth_size, var_size = len(latitudes), len(longitudes), len(depths), len(variables)

    assert prediction.shape == (lat_size, lon_size, depth_size, var_size), \
        f"Input array shape must be ({lat_size}, {lon_size}, {depth_size}, {var_size}), but got {prediction.shape}"

    pre_array = xr.DataArray(
        data = prediction,
        dims = ["latitude", "longitude", "depth", "variable"],
        coords = {
            "latitude": latitudes,
            "longitude": longitudes,
            "depth": depths,
            "variable": variables
        },
    )
    

    dataset = xr.Dataset({
        "prediction": pre_array,
    })

    dataset.to_netcdf(filename, format="NETCDF4")
    print(f"Data saved to {filename}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="", type=str)
    parser.add_argument("--depths", default=[0], type=list)
    parser.add_argument("--min_lon", default=0, type=float)
    parser.add_argument("--max_lon", default=360, type=float)
    parser.add_argument("--min_lat", default=-90, type=float)
    parser.add_argument("--max_lat", default=90, type=float)
    parser.add_argument("--lon_res", default=1, type=float)
    parser.add_argument("--lat_res", default=1, type=float)
    parser.add_argument("--lat_step", default=5, type=int)
    parser.add_argument("--lon_step", default=5, type=int)
    parser.add_argument("--depth_step", default=1, type=int)
    parser.add_argument("--batch_block", default=300, type=int)
    args = parser.parse_args()

    if args.lon_res == 0.25:
        res = '0p25'
    elif args.lon_res == 0.5:
        res = '0p5'
    elif args.lon_res == 1:
        res = '1'
    else:
        print('Not implementation')

    # %% Read config
    config_path = os.path.join(args.exp_dir, "config.yaml")
    print(f'Reading config from {config_path}')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(f'config:{config}')
    ic(config['test_dataset']['normalization'])

    var_size=len(config['dataset_config']['target_vars'])
    lat_block=config['dataset_config']['target_block_shape'][0]
    lon_block=config['dataset_config']['target_block_shape'][1]
    depth_block=config['dataset_config']['target_block_shape'][2] 
    lat_size=config['dataset_config']['target_shape'][0]
    lon_size=config['dataset_config']['target_shape'][1]
    depth_size=config['dataset_config']['target_shape'][2]

    # %% logging utils
    log_file_path = os.path.join(
        args.exp_dir, f"inference.log")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    logger = log_to_file_and_screen(log_file_path=log_file_path)

    # %% prediction save path
    out_dir = os.path.join(
        args.exp_dir,
        f"inference_{res}")
    os.makedirs(out_dir, exist_ok=True) 

    # %% Device init
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = "cpu"

    # %% Model loading
    model_ckpt_path = os.path.join(
        args.exp_dir, "training_checkpoints", "checkpoint.pth")
    logger.info(f'Load model from {model_ckpt_path}')
    model_spec = torch.load(model_ckpt_path)['model']
    model = models.make(model_spec, load_sd=True).to(device)

    num_context_points = (
        multiply_list_elements(config['dataset_config']['bg_block_shape']),
        config['dataset_config']['in_situ_obs_pad'],
        multiply_list_elements(config['dataset_config']['sate_sss_patch_shape']),
        multiply_list_elements(config['dataset_config']['sate_sst_patch_shape']),
        multiply_list_elements(config['dataset_config']['sate_ssw_patch_shape']),
        multiply_list_elements(config['dataset_config']['sate_sla_patch_shape']),
        multiply_list_elements(config['dataset_config']['sate_sic_patch_shape']),
    )

    # %% Generate target’s mesh
    target_coords, target_center_coords, target_block_coords, target_lats, target_lons, target_depths = slice_coordinates(
        args.min_lon,
        args.max_lon-args.lon_res,
        args.min_lat,
        args.max_lat-args.lat_res,
        args.depths,
        args.lon_res,
        args.lat_res,
        lat_block,
        lon_block,
        depth_block,
        args.lat_step,
        args.lon_step,
        args.depth_step
    )
    target_lat_size, target_lon_size, target_depth_size, var_size = target_coords.shape
    num_blocks = target_center_coords.shape[0]
    print(f'target_coords: {target_coords.shape}')
    print(f'target_center_coords: {target_center_coords.shape}')
    print(f'target_block_coords: {target_block_coords.shape}')
    num_blocks_split = np.arange(0, num_blocks, args.batch_block).tolist() + [num_blocks]

    # normalize coordinates
    n_blocks, lat_size, lon_size, depth_size, _ = target_block_coords.shape
    target_block_coords = target_block_coords.reshape(
        n_blocks, lat_size * lon_size * depth_size, 3)
    target_block_coords = normalize_coordinates(
        coords=target_block_coords,
        method=config['test_dataset']['loc_normalization'],
        stats_file=config['test_dataset']['location_stats_file'],
        variables=['lat', 'lon', 'depth']
    )

    num_target_points = multiply_list_elements(
        config['dataset_config']['target_block_shape']) 
    print(f'num_target_points: {num_target_points}')

    start_date = pd.to_datetime(config['test_dataset']['time_range'][0], format='%Y%m%d')
    end_date = pd.to_datetime(config['test_dataset']['time_range'][1], format='%Y%m%d')
    # Generate hourly time series
    time_series = pd.date_range(
        start=start_date,
        end=end_date + pd.Timedelta(days=1) - pd.Timedelta(hours=1),
        freq='D')
    ic(time_series) 

    model.eval()
    with torch.no_grad():
        for i, time_obj in enumerate(time_series[::5]):
            logger.info(f'time #{i}/{len(time_series)}: {time_obj}')

            out_file = os.path.join(out_dir, time_obj.strftime("%Y%m%d")+'.nc')
            # if os.path.exists(out_file):
            #     print(f'{out_file} exists! skip')
            #     continue

            prediction = []
            file_exists_flag = True
            for bi in range(len(num_blocks_split)-1):
                
                block_idx = np.arange(num_blocks_split[bi], num_blocks_split[bi+1])
                print(f'block_idx: {block_idx[0]}~{block_idx[-1]}/{num_blocks}')

                target_center_coords_ = target_center_coords[block_idx]
                target_block_coords_ = target_block_coords[block_idx]
                print(f'target_center_coords_: {target_center_coords_.shape}')
                print(f'target_block_coords_: {target_block_coords_.shape}')
                
                batch = read_and_norm_input(
                    time_obj, target_center_coords_, config['test_dataset'])
                if batch == None:
                    file_exists_flag = False
                    break

                for k, v in batch.items():
                    if not (k in ['target_lats', 'target_lons', 'target_depths', 'target']):
                        batch[k] = torch.from_numpy(v).to(torch.float32).to(device)
                target_block_coords_ = torch.from_numpy(target_block_coords_).to(torch.float32).to(device)

                context_points = (
                    (batch['bg_coords'], batch['bg_value']),
                    (batch['in_situ_obs_coords'], batch['in_situ_obs']),
                    (batch['sate_sss_coords'], batch['sate_sss_value']),
                    (batch['sate_sst_coords'], batch['sate_sst_value']),
                    (batch['sate_ssw_coords'], batch['sate_ssw_value']),
                    (batch['sate_sla_coords'], batch['sate_sla_value']),
                    (batch['sate_sic_coords'], batch['sate_sic_value']),
                )
                
                # Forward pass
                pred_blocks = model(
                    context_points,
                    num_context_points,
                    num_target_points,
                    target_block_coords_,
                )

                if config['loss']['learn_residual']:
                    print('add residual:')
                    print(f'pred_blocks: {pred_blocks.shape}')
                    print(f"bg_value: {batch['bg_value'].shape}")
                    pred_blocks = pred_blocks + batch['bg_value'] 

                print(f'pred_blocks: {pred_blocks.shape}')
                print(f'pred_blocks: {torch.min(pred_blocks)}~{torch.max(pred_blocks)}')
                
                # Append results 
                prediction.append(pred_blocks.detach().cpu())  # GPU --> CPU

                # free GPU memory
                del pred_blocks
                torch.cuda.empty_cache()

            if file_exists_flag == False:
                continue
            
            prediction = torch.cat(prediction, dim=0).numpy()
            print(f'prediction: {prediction.shape}')
            prediction = denormalize(
                prediction,
                config['test_dataset']['normalization'],
                config['dataset_config']['target_stats_file'],
                config['dataset_config']['target_vars'])

            num_blocks, _, var_size = prediction.shape
            prediction = prediction.reshape(
                num_blocks, lat_block, lon_block, depth_block, var_size)
            print(f'prediction: {prediction.shape}')
            
            prediction = reconstruct_array(
                chunks=prediction,
                lat_size=target_lat_size,
                lon_size=target_lon_size,
                depth_size=target_depth_size,
                var_size=var_size,
                lat_block=lat_block,
                lon_block=lon_block,
                depth_block=depth_block, 
                lat_step=args.lat_step,
                lon_step=args.lon_step,
                depth_step=args.depth_step,
            )
            ic(prediction.shape)

            for var_idx, var in enumerate(config['dataset_config']['target_vars']):
                print(var)
                print(f'Denormalized prediction: {np.nanmin(prediction[...,var_idx])}~{np.nanmax(prediction[...,var_idx])}')

            save_to_nc(
                prediction=prediction,
                filename=out_file, #  os.path.join(out_dir, time_obj.strftime("%Y%m%d")+'.nc')
                latitudes=target_lats,
                longitudes=target_lons,
                depths=target_depths,
                variables=config['dataset_config']['target_vars'],
            )
