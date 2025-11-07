import os
import re
import glob
import torch
import random
import logging
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from icecream import ic
from datetime import datetime, timedelta
from torch.utils.data import Subset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from utils.read import *
from utils.block import *
from utils.sample import *
from utils.normalization import *


def get_data_loader(config, batch_size, distributed, num_workers=8, logger=None, train=True):

    dataset = GetDataset(config, train, logger)

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=train)
    else:
        None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # shuffle=False,
        sampler=sampler if train else None,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset
    

class GetDataset(Dataset):
    def __init__(self, config, train=True, logger=None):
        self.config = config
        self.logger = logger
        self.train = train
        self.time_range = config['time_range']
        self.num_blocks = config['num_blocks']
        self.location_stats_file = config['location_stats_file']
        self.land_sea_mask_file_path = config['land_sea_mask_file_path']

        self.target_dir = config['target_dir']
        self.target_vars = config['target_vars']
        self.target_shape = config['target_shape']
        self.target_stats_file = config['target_stats_file']
        self.target_block_shape = config['target_block_shape']
        self.target_lat_step = config['target_lat_step']
        self.target_lon_step = config['target_lon_step']
        self.target_depth_step = config['target_depth_step']

        self.bg_dir = config['bg_dir']
        self.bg_vars = config['bg_vars']
        self.bg_interp = config['bg_interp']
        self.bg_lead_time = config['bg_lead_time']
        self.bg_stats_file = config['bg_stats_file']
        self.bg_block_shape = config['bg_block_shape']
        
        self.in_situ_obs_pad = config['in_situ_obs_pad']
        self.in_situ_obs_dir = config['in_situ_obs_dir']
        self.in_situ_obs_vars = config['in_situ_obs_vars']
        self.in_situ_obs_patch_range = config['in_situ_obs_patch_range']
        self.in_situ_obs_stats_file = config['in_situ_obs_stats_file']

        self.sate_sla_dir = config['sate_sla_dir']
        self.sate_sla_vars = config['sate_sla_vars']
        self.sate_sla_patch_shape = config['sate_sla_patch_shape']
        
        self.sate_sla_dir = config['sate_sla_dir']
        self.sate_sla_vars = config['sate_sla_vars']
        self.sate_sla_patch_shape = config['sate_sla_patch_shape']
        self.sate_sla_stats_file = config['sate_sla_stats_file']

        self.sate_sic_dir = config['sate_sic_dir']
        self.sate_sic_vars = config['sate_sic_vars']
        self.sate_sic_patch_shape = config['sate_sic_patch_shape']
        self.sate_sic_stats_file = config['sate_sic_stats_file']

        self.sate_sss_vars = config['sate_sss_vars']
        self.sate_sss_dir = config['sate_sss_dir']
        self.sate_sss_patch_shape = config['sate_sss_patch_shape']
        self.sate_sss_stats_file = config['sate_sss_stats_file']

        self.sate_sst_dir = config['sate_sst_dir']
        self.sate_sst_vars = config['sate_sst_vars']
        self.sate_sst_patch_shape = config['sate_sst_patch_shape']
        self.sate_sst_stats_file = config['sate_sst_stats_file']

        self.sate_ssw_vars = config['sate_ssw_vars']
        self.sate_ssw_dir = config['sate_ssw_dir']
        self.sate_ssw_patch_shape = config['sate_ssw_patch_shape']
        self.sate_ssw_stats_file = config['sate_ssw_stats_file']

        self.normalization = config['normalization']
        self.loc_normalization = config['loc_normalization']

        # self.latitudes = np.arange(-90, 90, 1)
        # self.longitudes = np.arange(0, 360, 1)
        # self.depths = [0, 2, 5, 7, 11, 15, 21, 29, 40, 55, 77, 92, 109, 130, 155, 186, 222, 266, 318, 380, 453, 541, 643]

        self._get_target_file_stats()

    def _get_target_file_stats(self):
        file_paths = glob.glob(f"{self.target_dir}/**/*.h5", recursive=True)

        start_time = datetime.strptime(self.time_range[0], '%Y%m%d')
        end_time = datetime.strptime(self.time_range[1], '%Y%m%d')

        self.target_files = []
        for path in file_paths:
            file_name = path.split('/')[-1] 
            file_time = file_name.split('.')[0] 
            file_datetime = datetime.strptime(file_time[:8], '%Y%m%d')
            if start_time <= file_datetime <= end_time:
                self.target_files.append(path)

        self.n_samples_total = len(self.target_files)
        self.logger.info(f"Found target data at: {self.target_dir}")
        self.logger.info(f"Time range: {self.time_range}")
        self.logger.info(f"Number of samples: {self.n_samples_total}")

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, day_idx):

        # %% Read target (181, 360, 23, 5)
        target_file_path = self.target_files[day_idx] 
        print(f'Read target from {target_file_path}')
        target, lats, lons, depths, target_time = read_glorys_file_v2(
            target_file_path,
            self.target_vars,
            fill_land=False,
            only_surface=True
        )
        # ic(target.shape, lats.shape, lons.shape)
        # print(f'Target time: {target_time}')
        target_blocks, target_center_coords, target_block_coords = slice_array(
            target,
            lats, lons, depths,
            self.target_block_shape[0],
            self.target_block_shape[1],
            self.target_block_shape[2],
            self.target_lat_step,
            self.target_lon_step,
            self.target_depth_step)
        n_blocks, lat_size, lon_size, depth_size, channel_num = target_blocks.shape
        target_blocks = target_blocks.reshape(
            n_blocks, lat_size * lon_size * depth_size, channel_num)
        target_block_coords = target_block_coords.reshape(
            n_blocks, lat_size * lon_size * depth_size, 3)

        # Choose subset of target blocks
        block_idx = np.random.randint(0, n_blocks, size=self.num_blocks)
        target_blocks = target_blocks[block_idx]
        target_block_coords = target_block_coords[block_idx]
        target_center_coords = target_center_coords[block_idx]

        target_blocks = normalize(
            data=target_blocks,
            method=self.normalization,
            stats_file=self.target_stats_file, 
            vars=self.target_vars)

        target_block_coords = normalize_coordinates(
            coords=target_block_coords,
            method=self.loc_normalization,
            max_depth=20,
            stats_file=self.location_stats_file,
            variables=['lat', 'lon', 'depth'])
        
        # %% Get land sea mask, provide mask for loss function
        ds = xr.open_dataset(self.land_sea_mask_file_path) # the shape of land_sea_mask is same as target
        land_sea_mask = ds['land_sea_mask'].values[0, :self.target_shape[0]] # surface
        depths = [0]
        lats = ds.coords['latitude'].values
        lons = ds.coords['longitude'].values
        ds.close()
        land_sea_mask = np.expand_dims(land_sea_mask, axis=(2, 3))
        land_sea_mask_blocks, _, _ = slice_array(
            land_sea_mask,
            lats, lons, depths,
            self.target_block_shape[0],
            self.target_block_shape[1],
            self.target_block_shape[2],
            self.target_lat_step,
            self.target_lon_step,
            self.target_depth_step
        )
        n_blocks, lat_size, lon_size, depth_size, channel_num = land_sea_mask_blocks.shape
        land_sea_mask_blocks = land_sea_mask_blocks.reshape(
            n_blocks, lat_size * lon_size * depth_size, channel_num)
        land_sea_mask_blocks = land_sea_mask_blocks[block_idx]
        
        # %% Read background
        bg_time = target_time - timedelta(days = self.bg_lead_time) # spot time
        bg_file_path = os.path.join(self.bg_dir, bg_time.strftime("%Y%m%d")+'.nc')
        print(f'Read bg from {bg_file_path}')
        bg, lats, lons, depths, valid_time = read_foracet_file(
            bg_file_path,
            self.bg_lead_time,
            fill_land=True,
            interp=self.bg_interp,
            variables=self.bg_vars,
        )
        # print(f'Bg valid time: {valid_time}')
        if valid_time != target_time:
            print(f'Time not matching')
            exit()
        bg, bg_coords = extract_blocks_from_coordinates_v1( # field -> blocks
            bg,
            lats, lons, depths,
            target_center_coords,
            self.bg_block_shape)
        n_blocks, lat_size, lon_size, depth_size, channel_num = bg.shape
        bg = bg.reshape(n_blocks, lat_size * lon_size * depth_size, channel_num)
        bg_coords = bg_coords.reshape(n_blocks, lat_size * lon_size * depth_size, 3)

        bg = normalize(
            data=bg,
            method=self.normalization,
            stats_file=self.bg_stats_file, 
            vars=self.bg_vars)

        bg_coords = normalize_coordinates(
            coords=bg_coords,
            method=self.loc_normalization,
            max_depth=20,
            stats_file=self.location_stats_file,
            variables=['lat', 'lon', 'depth'])
        # ic(bg.shape)

        # %% observation time of satellite (current day - 1 day)
        sate_date_obj = target_time - timedelta(days = 1)
        sate_date_str = sate_date_obj.strftime('%Y%m%d')

        # %% Satellite SSW 
        sate_ssw_file = os.path.join(
            self.sate_ssw_dir, str(sate_date_obj.year), f'{sate_date_str}.nc')
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
            self.sate_ssw_patch_shape,
        )
        sate_ssw = np.transpose(sate_ssw, (0, 2, 3, 1))
        n_blocks, lat_size, lon_size, channel_num = sate_ssw.shape
        sate_ssw = sate_ssw.reshape(n_blocks, lat_size * lon_size, channel_num)
        sate_ssw_coords = sate_ssw_coords.reshape(n_blocks, lat_size * lon_size, 3)

        sate_ssw = normalize(
            data=sate_ssw,
            method=self.normalization,
            stats_file=self.sate_ssw_stats_file, 
            vars=self.sate_ssw_vars)

        sate_ssw_coords = normalize_coordinates(
            coords=sate_ssw_coords,
            method=self.loc_normalization,
            max_depth=20,
            stats_file=self.location_stats_file,
            variables=['lat', 'lon', 'depth'])
        # ic(sate_ssw.shape)

        # %% Satellite SSS
        sate_sss_file = os.path.join(
            self.sate_sss_dir, str(sate_date_obj.year), f'{sate_date_str}.nc')
        ds = xr.open_dataset(sate_sss_file)
        sate_sss = np.expand_dims(ds[self.sate_sss_vars[0]].values, axis=0)
        lats = ds.coords['latitude'].values
        lons = ds.coords['longitude'].values
        ds.close()

        # QC
        sate_sss = sate_sss_qc(
            data=sate_sss,
            stats_file=self.sate_sss_stats_file, 
            channel_names=self.sate_sss_vars)
        
        sate_sss, sate_sss_coords = extract_patch_from_coordinates_v1(
            sate_sss,
            lats, lons,
            target_center_coords,
            self.sate_sss_patch_shape,
        )
        sate_sss = np.transpose(sate_sss, (0, 2, 3, 1))
        n_blocks, lat_size, lon_size, channel_num = sate_sss.shape
        sate_sss = sate_sss.reshape(n_blocks, lat_size * lon_size, channel_num)
        sate_sss_coords = sate_sss_coords.reshape(n_blocks, lat_size * lon_size, 3)

        sate_sss = normalize(
            data=sate_sss,
            method=self.normalization,
            stats_file=self.sate_sss_stats_file, 
            vars=self.sate_sss_vars)

        sate_sss_coords = normalize_coordinates(
            coords=sate_sss_coords,
            method=self.loc_normalization,
            max_depth=20,
            stats_file=self.location_stats_file,
            variables=['lat', 'lon', 'depth'])
        # ic(sate_sss.shape)

        # %% Read satellite SLA
        sate_sla_file = os.path.join(
            self.sate_sla_dir, str(sate_date_obj.year), f'{sate_date_str}.nc')
        ds = xr.open_dataset(sate_sla_file)
        sate_sla = np.expand_dims(ds[self.sate_sla_vars[0]].values, axis=0)
        # ic(sate_sla.shape)
        lats = ds.coords['lat'].values
        lons = ds.coords['lon'].values
        ds.close()

        sate_sla, sate_sla_coords = extract_patch_from_coordinates_v1(
            sate_sla,
            lats, lons,
            target_center_coords,
            self.sate_sla_patch_shape,
        )
        sate_sla = np.transpose(sate_sla, (0, 2, 3, 1))
        n_blocks, lat_size, lon_size, channel_num = sate_sla.shape
        sate_sla = sate_sla.reshape(n_blocks, lat_size * lon_size, channel_num)
        sate_sla_coords = sate_sla_coords.reshape(n_blocks, lat_size * lon_size, 3)

        sate_sla = normalize(
            data=sate_sla,
            method=self.normalization,
            stats_file=self.sate_sla_stats_file, 
            vars=self.sate_sla_vars)

        sate_sla_coords = normalize_coordinates(
            coords=sate_sla_coords,
            method=self.loc_normalization,
            max_depth=20,
            stats_file=self.location_stats_file,
            variables=['lat', 'lon', 'depth'])
        # ic(sate_sla.shape)

        # %% Read satellite's SST
        sate_sst_files = filter_AVHRR_pathfinder_files_by_time(
            self.sate_sst_dir, sate_date_obj, target_time)
        ds_0 = xr.open_dataset(sate_sst_files[0])
        sate_sst_0 = ds_0[self.sate_sst_vars[0]].values
        lons = ds_0.coords['lon'].values
        lats = ds_0.coords['lat'].values
        ds_0.close()
        ds_1 = xr.open_dataset(sate_sst_files[1])
        sate_sst_1 = ds_1[self.sate_sst_vars[0]].values
        ds_1.close()
        sate_sst = np.stack([sate_sst_0, sate_sst_1])
        sate_sst, sate_sst_coords = extract_patch_from_coordinates_v1(
            sate_sst,
            lats, lons,
            target_center_coords,
            self.sate_sst_patch_shape,
        )
        sate_sst = np.transpose(sate_sst, (0, 2, 3, 1))
        n_blocks, lat_size, lon_size, channel_num = sate_sst.shape
        sate_sst = sate_sst.reshape(n_blocks, lat_size * lon_size, channel_num)
        sate_sst_coords = sate_sst_coords.reshape(n_blocks, lat_size * lon_size, 3)

        sate_sst = normalize(
            data=sate_sst,
            method=self.normalization,
            stats_file=self.sate_sst_stats_file, 
            vars=self.sate_sst_vars)

        sate_sst_coords = normalize_coordinates(
            coords=sate_sst_coords,
            method=self.loc_normalization,
            max_depth=20,
            stats_file=self.location_stats_file,
            variables=['lat', 'lon', 'depth'])

        # %% Read sea ice concentration
        sea_ice_file = os.path.join(
            self.sate_sic_dir, str(sate_date_obj.year), f'{sate_date_str}.nc')
        ds = xr.open_dataset(sea_ice_file)
        sate_sic = np.expand_dims(ds[self.sate_sic_vars[0]].values, axis=0)
        lats = ds.coords['latitude'].values
        lons = ds.coords['longitude'].values
        # print(f'lats: {lats}')
        # print(f'lons: {lons}')
        ds.close()
        sate_sic, sate_sic_coords = extract_patch_from_coordinates_v1(
            sate_sic,
            lats, lons,
            target_center_coords,
            self.sate_sic_patch_shape,
        )
        sate_sic = np.transpose(sate_sic, (0, 2, 3, 1))
        n_blocks, lat_size, lon_size, channel_num = sate_sic.shape
        sate_sic = sate_sic.reshape(n_blocks, lat_size * lon_size, channel_num)
        sate_sic_coords = sate_sic_coords.reshape(n_blocks, lat_size * lon_size, 3)

        sate_sic = normalize(
            data=sate_sic,
            method=self.normalization,
            stats_file=self.sate_sic_stats_file,
            vars=self.sate_sic_vars)

        sate_sic_coords = normalize_coordinates(
            coords=sate_sic_coords,
            method=self.loc_normalization,
            max_depth=20,
            stats_file=self.location_stats_file,
            variables=['lat', 'lon', 'depth'])

        # %% Read in situ observations
        in_situ_obs_file = os.path.join(
            self.in_situ_obs_dir, str(sate_date_obj.year), f'{sate_date_str}.csv')
        in_situ_obs, in_situ_obs_coords, _ = filter_in_situ_by_coords_and_ranges_v1( # include QC
            in_situ_obs_file,
            target_center_coords,
            self.in_situ_obs_patch_range,
            self.in_situ_obs_pad,
        )

        # QC check
        in_situ_obs = in_situ_obs_qc(
            data = in_situ_obs,
            stats_file = self.in_situ_obs_stats_file,
            channel_names = self.in_situ_obs_vars,
        )

        in_situ_obs = normalize(
            data=in_situ_obs,
            method=self.normalization,
            stats_file=self.in_situ_obs_stats_file,
            vars=self.in_situ_obs_vars)

        in_situ_obs_coords = normalize_coordinates(
            coords=in_situ_obs_coords,
            method=self.loc_normalization,
            max_depth=20,
            stats_file=self.location_stats_file,
            variables=['lat', 'lon', 'depth'])
        
        return {
            'land_sea_mask': land_sea_mask_blocks,
            'bg_value': bg,
            'bg_coords': bg_coords,
            'target_value': target_blocks,
            'target_coords': target_block_coords,
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
