import os
import time
import yaml
import random
import wandb
import swanlab
import datetime
import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from icecream import ic
from str2bool import str2bool
from shutil import copyfile
# from apex import optimizers
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel import DistributedDataParallel

from utils.YParams import YParams
from utils.averager import Averager
from utils.data_loader_multifiles import get_data_loader
from utils.logger import log_to_file_and_screen
from utils.make_optimizer import make_optimizer
from utils.loss_func import MaskedMSELoss

import models


class Trainer:

    def __init__(self, config, world_rank, logger):
        self.config = config
        self.world_rank = world_rank
        self.logger = logger

        # %% init GPU
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)
        logger.info(f"device: {self.device}")

        logger.info('Init model')
        self.model = models.make(
            self.config['model']).to(self.device)
        
        # %% Number of parameters
        params_count = self.count_parameters()
        logger.info(f'params_count: {params_count}')

        # %% DDP model
        if dist.is_initialized():
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=[local_rank],
                find_unused_parameters=True,
            )

        # %% Loss function
        if self.config['loss']['mask_land']:
            self.loss_fn = MaskedMSELoss() 
        else:
            self.loss_fn = nn.MSELoss()
        
        # %% Init optimizer after build DDP model
        self.optimizer = make_optimizer(
            self.model.parameters(),
            self.config['optimizer']
        )

        # %% Resume train, including load trained model and optimizer 
        if config['resume']:
            logger.info(f"Loading checkpoint from {config['checkpoint_dir']}")
            self.restore_checkpoint(config["checkpoint_dir"])
            self.epoch = self.startEpoch
        else:
            self.epoch = 0
        if config['watch'] == 'wandb':
            wandb.watch(self.model)

        # %% Load data
        logger.info("rank %d, begin data loader init" % world_rank)
        (self.train_data_loader, self.train_dataset, self.train_sampler) = get_data_loader(
            config=config['train_dataset'],
            batch_size=config['local_batch_size'],
            distributed=dist.is_initialized(),
            num_workers=config['num_workers'],
            logger=logger,
            train=True,
        )
        (self.valid_data_loader, self.valid_dataset, self.valid_sampler) = get_data_loader(
            config=config['valid_dataset'],
            batch_size=config['local_batch_size'],
            distributed=dist.is_initialized(),
            num_workers=config['num_workers'],
            logger=logger,
            train=True,
        )
        
        # %% Dynamical learning rate
        if self.config.get('multi_step_lr') is None:
            self.lr_scheduler = None
        else:
            self.lr_scheduler = MultiStepLR(
                self.optimizer,
                **self.config['multi_step_lr']
            )


    def count_parameters(self):
        count_params = 0
        for p in self.model.parameters():
            if p.requires_grad:
                count_params += p.numel()
        return count_params
    

    def restore_checkpoint(self, ckpt_dir):
        checkpoint_path = os.path.join(
            ckpt_dir, "epoch-" + str(self.config['resume_ep']) + '.pth')
        self.logger.info(f'ckpt_path: {checkpoint_path}')

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cuda:{}".format(self.config['local_rank'])
        )

        model_spec = checkpoint['model']
        if self.config['n_gpus'] == 1:
            self.model.load_state_dict(model_spec['state_dict'])
        else:
            new_state_dict = OrderedDict()
            for key, val in model_spec['state_dict'].items():
                name = 'module.' + key
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)

        # optimizer
        optimizer_spec = checkpoint['optimizer']
        self.optimizer.load_state_dict(optimizer_spec['state_dict'])

        # uses config specified lr.
        for g in self.optimizer.param_groups:
            g["lr"] = self.config['resume_lr']

        # epoch to start
        self.startEpoch = checkpoint["epoch"]

    def save_model(self, epoch):

        if self.config['n_gpus'] > 1:
            model_ = self.model.module
        else:
            model_ = self.model

        save_path = os.path.join(
            self.config['checkpoint_dir'],
            f'epoch-{epoch}.pth'
        )
        self.logger.info(f"Save model to {save_path}")

        model_spec = self.config['model']
        model_spec['state_dict'] = model_.state_dict()

        optimizer_spec = self.config['optimizer']
        optimizer_spec['state_dict'] = self.optimizer.state_dict()
        
        sv_data = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }
        torch.save(sv_data, save_path)


    def train_one_epoch(self):

        self.logger.info("Train one epoch...")

        train_loss = Averager()

        steps_in_one_epoch = 0
        loss_total = 0

        # self.model.train()
        for i, batch in enumerate(self.train_data_loader, 0):

            steps_in_one_epoch += 1

            # %% transfer to GPU
            for k, v in batch.items():
                batch[k] = v.to(torch.float32).to(self.device)
                print(f'{k}: shape={batch[k].shape}, {torch.min(batch[k])}~{torch.max(batch[k])}, mean={torch.mean(batch[k])}')

            land_sea_mask = batch['land_sea_mask']

            if self.config['loss']['learn_residual']:
                print("learn residual:")
                print(f"target: {batch['target'].shape}")
                print(f"bg: {batch['bg'].shape}")
                target = batch['target'] - batch['bg']
            else:
                target = batch['target']
                
            # Forward pass
            pred = self.model(batch['input'])
            print(f"prediction: {torch.min(pred)}~{torch.max(pred)}")
            print(f'prediction: {pred.shape}')
            print(f"target: {torch.min(target)}~{torch.max(target)}")
            print(f'target: {target.shape}')

            # Compute loss
            if self.config['loss']['mask_land']:
                print('Calculate loss after masking land')
                loss = self.loss_fn(pred, target, land_sea_mask)
            else:
                loss = self.loss_fn(pred, target)
            self.logger.info(f"step={steps_in_one_epoch} loss={loss.item()}")

            train_loss.add(loss.item())
            loss_total += loss

            # Backward pass and optimization
            self.optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Update model parameters

            pred = None; loss = None

        logs = {"train_loss": loss_total / steps_in_one_epoch}

        # All reduce
        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach())
                logs[key] = float(logs[key] / dist.get_world_size())

        return train_loss.item(), logs
    
    def valid_one_epoch(self):
        self.logger.info("Validate one epoch...")
        self.model.eval()

        buff = torch.zeros((2), dtype=torch.float32, device=self.device)
        loss_total = buff[0].view(-1)
        steps = buff[1].view(-1)

        valid_loss = Averager()

        with torch.no_grad():
            for i, batch in enumerate(self.valid_data_loader, 0):
                # %% transfer to GPU
                for k, v in batch.items():
                    batch[k] = v.to(torch.float32).to(self.device)
                    # logger.info(f'{k}: shape={batch[k].shape}, {torch.min(batch[k])}~{torch.max(batch[k])}, mean={torch.mean(batch[k])}')

                land_sea_mask = batch['land_sea_mask']

                if self.config['loss']['learn_residual']:
                    print("learn residual:")
                    print(f"target: {batch['target'].shape}")
                    print(f"bg: {batch['bg'].shape}")
                    target = batch['target'] - batch['bg']
                else:
                    target = batch['target']

                # Forward pass
                pred = self.model(batch['input'])

                # Compute loss
                if self.config['loss']['mask_land']:
                    print('Calculate loss after masking land')
                    loss = self.loss_fn(pred, target, land_sea_mask)
                else:
                    loss = self.loss_fn(pred, target)
                
                valid_loss.add(loss.item())
                loss_total += loss
                steps += 1

                pred = None; loss = None

        if dist.is_initialized():
            dist.all_reduce(buff)

        # Average
        buff[0:1] = buff[0:1] / buff[1]
        buff_cpu = buff.detach().cpu().numpy()

        if self.config['loss']['mask_land']:
            loss_name = 'valid_loss_MSE_mask_land'
        else:
            loss_name = 'valid_loss' 
        logs = {
            loss_name: buff_cpu[0],
        }

        return valid_loss.item(), logs

    def train(self):

        for epoch in range(self.epoch, self.config['epoch_max']):

            if dist.is_initialized():
                # different batch on each GPU
                self.train_sampler.set_epoch(epoch)
                # self.valid_sampler.set_epoch(epoch)

            # Train one epoch
            train_loss, logs = self.train_one_epoch()
            self.logger.info(f'epoch:{epoch}, train_loss: {train_loss}')
            self.logger.info(f"epoch:{epoch}, all reduce train_loss: {logs['train_loss']}")
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.config['loss']['mask_land']:
                loss_name = 'train_loss_MSE_mask_land'
            else:
                loss_name = 'train_loss' 
                
            if config['watch'] == 'wandb':
                wandb.log(
                    {
                        loss_name: logs['train_loss'],
                        "lr": current_lr
                    },
                    step=epoch
                )
            else:
                swanlab.log(
                    {
                        loss_name: logs['train_loss'],
                        "lr": current_lr
                    },
                    step=epoch
                )
                

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            #  Save model
            if (self.world_rank == 0 and epoch % self.config['save_model_freq'] == 0):
                self.save_model(epoch)

            if (epoch % self.config['epoch_valid_freq'] == 0):
                valid_loss, logs = self.valid_one_epoch()
                self.logger.info(f'epoch:{epoch}, valid_loss: {valid_loss}')
                # self.logger.info(f"epoch:{epoch}, all reduce valid_loss: {logs['valid_loss']}")

                if self.config['watch'] == 'wandb':
                    wandb.log(logs, step=epoch)
                else:
                    swanlab.log(logs, step=epoch)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default="./configs/***.yaml", type=str)
    parser.add_argument("--exp_dir", default="./exps/surface_NP_fcst_bg_glorys", type=str)
    parser.add_argument("--run_num", default="00", type=str)
    parser.add_argument("--resume", default=False, type=str2bool)
    parser.add_argument("--resume_ep", default=100, type=int)
    parser.add_argument("--resume_lr", default=False, type=float)
    parser.add_argument("--device", default="GPU", type=str)
    parser.add_argument("--n_gpus", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    # parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--save_model_freq", default=5, type=int)
    parser.add_argument("--epoch_valid_freq", default=5, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--watch", default='swanlab', type=str)
    args = parser.parse_args()

    if args.resume:
        yaml_config_file = os.path.join(args.exp_dir, args.run_num, 'config.yaml')
        print(f'Reading config from {yaml_config_file}')
        with open(yaml_config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print(f'Reading config from {args.yaml_config}')
        with open(args.yaml_config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    config['n_gpus'] = args.n_gpus
    config['save_model_freq'] = args.save_model_freq
    config['epoch_valid_freq'] = args.epoch_valid_freq
    config['resume_ep'] = args.resume_ep
    config['resume_lr'] = args.resume_lr
    config['watch'] = args.watch

    # %% Init distributed process
    if args.device == "GPU":
        print("Initialize distributed process group.")
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=5400)
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        config["local_rank"] = local_rank
        torch.backends.cudnn.benchmark = True

        world_rank = dist.get_rank()  # get current process's ID
        print(f"world_rank: {world_rank}")
    
    # %% Set up directory
    expDir = os.path.join(args.exp_dir, str(args.run_num))
    if (not args.resume) and (world_rank==0):
        os.makedirs(expDir, exist_ok=True)
        os.makedirs(
            os.path.join(expDir, "training_checkpoints"),
            exist_ok=True)
        copyfile(
            os.path.abspath(args.yaml_config),
            os.path.join(expDir, "config.yaml"))
    config["experiment_dir"] = os.path.abspath(expDir)
    config["checkpoint_dir"] = os.path.join(expDir, "training_checkpoints")
    config['resume'] = args.resume

    # %% Init wandb
    if args.watch == 'wandb':
        os.environ["WANDB_API_KEY"] = config['wandb']['api_key']
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            config=config,
            name=args.run_num,
            group=config['wandb']['group'],
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            settings={"_service_wait": 600, "init_timeout": 1200},
        )
    else:
        swanlab.init(
            project=config['swanlab']['project'],
            experiment_name=args.run_num,
            workspace=config['swanlab']['workspace'],
            config=config,
            mode='cloud',
        )

    if "WORLD_SIZE" in os.environ:
        config['world_size'] = int(os.environ["WORLD_SIZE"])
        # world_size = Number of process = GPUs * Nodes
    # batch size must be divisible by the number of gpu's
    config['local_batch_size'] = int(args.batch_size // config['world_size']) 
    config['num_workers'] = args.num_workers

    # %% logging utils
    logger = log_to_file_and_screen(log_file_path=os.path.join(expDir, "train.log"))   
    logger.info(f'config: {config}')

    set_random_seed(args.seed)

    trainer = Trainer(config, world_rank, logger)
    trainer.train()
    logger.info("DONE ---- rank %d" % world_rank)
