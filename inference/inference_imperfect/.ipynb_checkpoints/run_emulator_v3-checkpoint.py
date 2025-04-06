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
#from tqdm import tqdm
from functools import partial
import json
import tensorflow as tf
from tensorflow.keras import layers
#AUTOTUNE = tf.data.experimental.AUTOTUNE
from dask.diagnostics import ProgressBar
import pathlib
print("imported vs")
#from ops.model_inference.src_eval_inference import *

inp = sys.argv
GAN_flag = sys.argv[1]
variable = sys.argv[2]
input_data_path_base = sys.argv[3]#"/nesi/project/niwa03712/CMIP6_data/Downscaled_Preprocessed"
gcm = sys.argv[4]
ssp = sys.argv[5]
variant = sys.argv[6]
output_path_setup = sys.argv[7]
emulator_type = sys.argv[8]
code_dir = sys.argv[9]
framework = "imperfect"
sys.path.append(f'{code_dir}/code/applying_ml_model/v3')
os.chdir(code_dir)
from src.layers import *
from src.models import *
from src.gan import *
from src.util_functions import *


if emulator_type == 'imperfect_emulator':
    configs = {"sfcwind": "doury_single_var_model_v1_sfcWind_from_scatch_imperfect",
               "sfcwindmax": "doury_single_var_model_v1_sfcWindmax_from_scatch_imperfect",
               "tasmax": "doury_single_var_model_tasmax_imperfect",
               "tasmin": "doury_single_var_model_tasmin_imperfect",
               "pr":"doury_single_var_model_pr_pr_from_scratch_v3_v2_finetuned_new_outputnorm_imperfect"}

else:
    configs = {"sfcwind": "NIWA-REMS_sfcWind_v280125",
           "sfcwindmax": "NIWA-REMS_sfcWindmax_v280125",
           "tasmax": "NIWA-REMS_tasmax_v280125",
           "tasmin": "NIWA-REMS_tasmin_v280125",
           "pr":"NIWA-REMS_pr_v280125"}

"currently changing filepath structure"

'''
Define config file and directories
'''
    
# define directories
ground_truth_dir = '/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/'

print('BEGINNING EMULATOR DOWNSCALING')
# Loading up the minimum value dataset
print('Variable: ' + variable + 'shit' + os.getcwd())
ml_model_name = configs[variable]
print(f"current path: {os.getcwd()}, code_dir: {code_dir}")
min_value = xr.open_dataset(r'./models/' + ml_model_name + '/min_value_outputs.nc')
input_file, output_path = get_files(input_data_path_base, gcm, ssp, variant, output_path_setup, variable, configs)

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
processed_GCM_data,mean_data,variance_data, vegt, orog,he,time_of_year = prepare_ML_inputs(input_file , config, framework)
# PREP OUTPUT
print('PREP OUTPUT')
output_shape = initialize_output_ds(processed_GCM_data, config)
output_shape = output_shape.rename({"pr":variable})
# APPLY MODEL
print('APPLY ML MODEL')
output = predict_parallel_resid_corrector_v4varname(generator, unet_model, processed_GCM_data.transpose('time','lat','lon','channel').values,\
                                                        output_shape, 256, orog, output_means, output_stds, time_of_year,\
                                                        mean_data.transpose('time','channel'), variance_data.transpose('time','channel'), gan=GAN_flag,\
                                                        min_value=min_value, varname = variable)

output['GCM'] = f'{gcm}'
output.coords['scenario'] = f'{ssp}'
output.attrs['title'] = 'Generative AI Downscaled data'
output.attrs['contact'] = 'Neelesh Rampal (neelesh.rampal@niwa.co.nz)'
output.attrs['CMIP6_model'] = f'{gcm}'
output.attrs['CMIP6_scenario'] = f'{ssp}'
output.attrs['version'] = f"{configs[variable]} of CCAM Version: {output.attrs['version']}"
output.attrs['source'] = f"NIWA-REMS Generative AI Downscaling: https://github.com/nram812/On-the-Extrapolation-of-Generative-Adversarial-Networks-for-downscaling-precipitation-extremes \n Training Data is from {output.attrs['source']}"
output.attrs['training_data'] = 'https://zenodo.org/records/13755688'
parent_path = pathlib.Path(output_path).parents[0]

if not os.path.exists(parent_path):
        os.makedirs(parent_path)

output.to_netcdf(output_path)
print('\n\n')

#
# quantiles = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9999, 0.99999]
# dset = []
# variants = []
# for variant in list_of_variants:
#     if varname == "pr":
#         OUTPUT_FILENAME_hist=f"/nesi/nobackup/niwa00018/rampaln/ML_emulator_downscaled/CMIP/CSIRO/ACCESS-ESM1-5/historical/{variant}/day/ML_Downscaled_Sfc_CMIP_CSIRO_ACCESS-ESM1-5_historical_{variant}_day_gn_*.nc"
#         OUTPUT_FILENAME_future=f"/nesi/nobackup/niwa00018/rampaln/ML_emulator_downscaled/ScenarioMIP/CSIRO/ACCESS-ESM1-5/ssp370/{variant}/day/ML_Downscaled_Sfc_ScenarioMIP_CSIRO_ACCESS-ESM1-5_ssp370_{variant}_day_gn_*.nc"
#     else:
#         OUTPUT_FILENAME_hist=f"/nesi/nobackup/niwa00018/rampaln/ML_emulator_downscaled/CMIP/CSIRO/ACCESS-ESM1-5/historical/{variant}/day/ML_Downscaled_{varname}_CMIP_CSIRO_ACCESS-ESM1-5_historical_{variant}_day_gn_*.nc"
#         OUTPUT_FILENAME_future=f"/nesi/nobackup/niwa00018/rampaln/ML_emulator_downscaled/ScenarioMIP/CSIRO/ACCESS-ESM1-5/ssp370/{variant}/day/ML_Downscaled_{varname}_ScenarioMIP_CSIRO_ACCESS-ESM1-5_ssp370_{variant}_day_gn_*.nc"
#     print(OUTPUT_FILENAME_future, OUTPUT_FILENAME_hist)
#     df_hist = xr.open_mfdataset(OUTPUT_FILENAME_hist)
#     df_future = xr.open_mfdataset(OUTPUT_FILENAME_future)
#
#     df_hist = df_hist.sel(time = slice("1985","2014")).pr
#     print(df_hist)
#     df_hist_climo = df_hist.groupby('time.season').mean()
#     if varname =="pr":
#         df_hist = df_hist.where(df_hist >0.5, np.nan).quantile(q = quantiles, dim ="time")
#     else:
#         df_hist = df_hist.quantile(q=quantiles, dim="time")
#
#     df_future = df_future.sel(time = slice("2070","2100")).pr
#     print(df_future)
#     df_future_climo = df_future.groupby('time.season').mean()
#     if varname == "pr":
#         df_future = df_future.where(df_future >0.5, np.nan).quantile(q = quantiles, dim ="time")
#     else:
#         df_future = df_future.quantile(q=quantiles, dim="time")
#
#     merged_variables = xr.merge([df_hist.to_dataset(name ="hist_quantiles"),
#                                  df_future.to_dataset(name ="future_quantiles"),
#                                  df_hist_climo.to_dataset(name = 'hist_climo'),
#                                  df_future_climo.to_dataset(name ='future_climo')])
#     dset.append(merged_variables)
#     variants.append(variant)
#
#
#
# concat = xr.concat(dset, dim = "variant")
# concat['variant'] = (('variant'), variants)
#
# #concat.to_netcdf('/nesi/nobackup/niwa00018/rampaln/ML_emulator_downscaled/ScenarioMIP/ACCESS-ESM1-5_ensemble_pr.nc')
#
# concat.to_netcdf(f'/nesi/nobackup/niwa00018/rampaln/ML_emulator_downscaled/ScenarioMIP/ACCESS-ESM1-5_ensemble_{varname}.nc')

