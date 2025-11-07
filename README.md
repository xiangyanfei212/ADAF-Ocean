# Description

Accurate and efficient global ocean state estimation remains a grand challenge for Earth system science, hindered by the dual bottlenecks of computational scalability and degraded data fidelity in traditional data assimilation (DA) and deep learning (DL) approaches. Here we present an AI-driven Data Assimilation Framework for Ocean (ADAF-Ocean) that directly assimilates multi-source and multi-scale observations, ranging from sparse in-situ measurements to 4 km satellite swaths, without any interpolation or data thinning. Inspired by Neural Processes, ADAF-Ocean learns a continuous mapping from heterogeneous inputs to ocean states, preserving native data fidelity. Through AI-driven super-resolution, it reconstructs 0.25° mesoscale dynamics from coarse 1° fields, which ensures both efficiency and scalability, with just 3.7% more parameters than the 1° configuration. When coupled with a DL forecasting system, ADAF-Ocean extends global forecast skill by up to 20 days compared to baselines without assimilation. This framework establishes a computationally viable and scientifically rigorous pathway toward real-time, high-resolution Earth system monitoring.

# Project Structure

The main structure of the repository is as follows:

```
ADAF-Ocean/
├── configs/ # Configuration files
├── data/ # Directory for storing data
├── models/ # Machine learning model-related code
├── stats/ # Statistical analysis tools
├── utils/ # Utility functions
├── calc_metrics.py # Script for calculating model performance metrics
├── inference.py # Script for loading models and making prediction
├── train.py # Script for training the AI models
```

# Data Preparation

The datasets used in this project include multi-source observations and forecast data.
Some examples of these data can be accessed at Zenodo[url: ], including NetCDF (`.nc`) files for three consecutive days: January 1, 2, and 3, 2020. 
Each dataset is stored in a folder based on the type of data it contains:

- **`forecast_bg/`**: Background forecast data.
- **`glorys/`**: Reanalysis data from the GLORYS model.
- **`insitu/`**: In-situ observational data (e.g., temperature and salinity from ships, buoys, floats).
- **`mask/`**: Land-sea mask file.
- **`SIC/`**: Sea Ice Concentration.
- **`SSH/`**: Sea Surface Height.
- **`SSS/`**: Sea Surface Salinity.
- **`SST/`**: Sea Surface Temperature.
- **`SSW/`**: Sea Surface Wind.

The datasets are sourced from the following repositories and services:

- **HadIOD Dataset**: © Met Office, provided under the Non-Commercial Government Licence.
- **GLORYS, SMOS Sea Surface Salinity, ASCAT Sea Surface Wind, and Sea Surface Heights Datasets**: © Copernicus Marine Service, licensed under Creative Commons Attribution 4.0 International.
- **AVHRR Pathfinder SST**: © NOAA National Centers for Environmental Information.

Please refer to the respective licenses for more details.

# Trained model weights

Model weights can be accessed at Zenodo[url: ].

# Usage


## Environments Installation

To ensure a consistent computational environment, we provide a `environment.yaml` file that specifies all required dependencies. 

Create and activate the Conda Environment**:  
   Use the provided `environment.yaml` file to create the environment:
   ```bash
   conda env create -f environment.yaml
   conda activate ADAF-Ocean
   ```
## Training Script

The repository provides an example training script to train the model using the datasets.

```
export CUDA_VISIBLE_DEVICES='0,1'

yaml_config='./configs/config.yaml'
exp_dir='./exps/'

run_num=$(date "+%Y%m%d-%H%M%S")
resume=False
resume_ep=0
resume_lr=1e-4

device='GPU'
batch_size=4
seed=42

torchrun --nproc_per_node=2 \
    --master_port=26900 \
    train.py \
    --yaml_config=${yaml_config} \
    --exp_dir=${exp_dir} \
    --resume=${resume} \
    --resume_ep=${resume_ep} \
    --resume_lr=${resume_lr} \
    --run_num=${run_num} \
    --batch_size=${batch_size} \
    --n_gpus=2 \
    --num_workers=8 \
    --device=${device} \
    > logs/train_${run_num}.log 2>&1 &
```


## Inference

The repository also provides an inference script to evaluate the trained model and generate predictions. Below is an example usage:
```
exp_dir='./exps/' # the trained model weights must put in this folder

nohup python -u inference.py \
    --exp_dir=${exp_dir} \
    --lon_res=1 \
    --lat_res=1 \
    --lat_step=5 \
    --lon_step=5 \
    --depth_step=1 \
    --batch_block=300 \
    > logs/inference.log 2>&1 &
```









