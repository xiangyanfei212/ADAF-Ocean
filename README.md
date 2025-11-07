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

## Environments installation



## Train

## Inference




