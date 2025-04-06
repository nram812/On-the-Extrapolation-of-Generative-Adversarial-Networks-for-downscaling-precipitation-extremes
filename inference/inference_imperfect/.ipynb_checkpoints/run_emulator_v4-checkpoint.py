"""Modified Laura's code for internal variability
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import tqdm
# from tqdm import tqdm
from functools import partial
import json
import tensorflow as tf
from tensorflow.keras import layers
# AUTOTUNE = tf.data.experimental.AUTOTUNE
from dask.diagnostics import ProgressBar
import pathlib

inp = sys.argv
GAN_flag = sys.argv[1]
variable = sys.argv[2]
input_data_path_base = sys.argv[3]  # "/nesi/project/niwa03712/CMIP6_data/Downscaled_Preprocessed"
gcm = sys.argv[4]
ssp = sys.argv[5]
variant = sys.argv[6]
output_path_setup = sys.argv[7]
emulator_type = sys.argv[8]
code_dir = sys.argv[9]

sys.path.append(f'{code_dir}/code/applying_ml_model/v3/src')
os.chdir(code_dir)
#try:
from src.util_functions import *
sys.path.append(f'/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/')
from src.layers import *
from src.models import *
from src.gan import *

configs = {"sfcWind": "NIWA-REMS_sfcWind_v050425",
           "sfcWindmax": "NIWA-REMS_sfcWindmax_v050425",
           "tasmax": "NIWA-REMS_tasmax_v050425",
           "tasmin": "NIWA-REMS_tasmin_v050425",
           "pr": "NIWA-REMS_pr_v050425"}
    
method = 'basic' # method of preprocessing input data
'''
Define config file and directories
'''
framework = "imperfect"
# define directories
ground_truth_dir = '/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/'

print('BEGINNING EMULATOR DOWNSCALING')
# Loading up the minimum value dataset
print('Variable: ' + variable + os.getcwd())
ml_model_name = configs[variable]
print(f"current path: {os.getcwd()}, code_dir: {code_dir}")
min_value = None#xr.open_dataset(r'./models/' + ml_model_name + '/min_value_outputs.nc')
print(input_data_path_base, gcm, ssp,
                                    variant, output_path_setup, variable, configs)
input_file, output_path = get_files(input_data_path_base, gcm, ssp,
                                    variant, output_path_setup, variable, configs)
if not os.path.exists(output_path):
    config_file = r'./models/' + ml_model_name + '/config_info.json'
    with open(config_file) as f:
        config = json.load(f)

    output_means = xr.open_dataset(config["means_output"])
    output_stds = xr.open_dataset(config["stds_output"])
    # LOAD ML MODEL
    print('LOAD ML MODEL\n')
    generator, unet_model = load_model(ml_model_name, './models')
    # PREP INPUT
    print('\nPREP INPUT')
    processed_GCM_data, mean_data, variance_data, vegt, orog, he, time_of_year = prepare_ML_inputs(input_file, config,
                                                                                                   framework, method = method)
    # PREP OUTPUT

    with ProgressBar():
        mean_data = mean_data.load()
        variance_data = variance_data.load()
        processed_GCM_data = processed_GCM_data.load()
        time_of_year = time_of_year.load()
        print(mean_data, time_of_year)
    print('PREP OUTPUT')
    output_shape = initialize_output_ds(processed_GCM_data, config)
    output_shape = output_shape.rename({"pr": variable})
    # APPLY MODEL
    print('APPLY ML MODEL')
    output = predict_parallel_resid_corrector_v4varname(generator, unet_model,
                                                        processed_GCM_data.transpose('time', 'lat', 'lon',
                                                                                     'channel').values, \
                                                        output_shape, 64, orog, output_means, output_stds, time_of_year, \
                                                        mean_data.transpose('time', 'channel'),
                                                        variance_data.transpose('time', 'channel'), gan=GAN_flag, \
                                                        min_value=min_value, varname=variable)

    output['GCM'] = f'{gcm}'
    output.coords['scenario'] = f'{ssp}'
    output.attrs['title'] = 'Generative AI Downscaled data'
    output.attrs['contact'] = 'Neelesh Rampal (neelesh.rampal@niwa.co.nz)'
    output.attrs['CMIP6_model'] = f'{gcm}'
    output.attrs['CMIP6_scenario'] = f'{ssp}'
    output.attrs['version'] = f"{configs[variable]} of CCAM Version: {output.attrs['version']}"
    output.attrs[
        'source'] = f"NIWA-REMS Generative AI Downscaling: https://github.com/nram812/On-the-Extrapolation-of-Generative-Adversarial-Networks-for-downscaling-precipitation-extremes \n Training Data is from {output.attrs['source']}"
    output.attrs['training_data'] = 'https://zenodo.org/records/13755688'
    parent_path = pathlib.Path(output_path).parents[0]

    if not os.path.exists(parent_path):
        os.makedirs(parent_path)

    output.to_netcdf(output_path)
    print('\n\n')

