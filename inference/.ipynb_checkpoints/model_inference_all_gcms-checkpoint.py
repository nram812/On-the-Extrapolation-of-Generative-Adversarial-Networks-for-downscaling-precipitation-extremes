import xarray as xr
import sys
import matplotlib.pyplot as plt
import os
import tqdm
from tqdm import tqdm
import tqdm
import numpy as np
from functools import partial
import json
# changed activation function to hyperbolic tangent
import tensorflow as tf
from tensorflow.keras import layers
AUTOTUNE = tf.data.experimental.AUTOTUNE
from dask.diagnostics import ProgressBar
import pandas as pd
import sys
# Please make sure that this path is the directory of the repository
# or os.chdir(/path/of/repository)
#sys.path.append(os.getcwd())
os.chdir(r'/nesi/project/niwa00018/ML_downscaling_CCAM/On-the-Extrapolation-of-Generative-Adversarial-Networks-for-downscaling-precipitation-extremes')
sys.path.append(r'/nesi/project/niwa00018/ML_downscaling_CCAM/On-the-Extrapolation-of-Generative-Adversarial-Networks-for-downscaling-precipitation-extremes/src/')
from layers import *
from models import *
from gan import *
from src_eval_inference import *
config_file = sys.argv[-1]
# please note this is the config file for the "model"
output_norm =1
# not used but note
# config_file = r'./experiment_configs/Hist_vs_future_runs/historical.json'
# When you run the inference in python, you need to specify the experiment
# i.e. python model_inference.py /path/to/config/file
# see the slurm script for more information on how to run this experiment
config_file_for_test_data = r'./model_inference/metadata_all_gcms_v2.json'
# you may want to change directory to the repo-path i.e. os.chdir(/your/path/to/repo)
with open(config_file) as f:
    config = json.load(f)
    
with open(config_file_for_test_data) as f:
    config_test_data = json.load(f)
# the quantiles of which the climate change signal is computed over
quantiles = [ 0.5 , 0.7, 0.9, 0.925,
             0.95, 0.975, 0.98, 0.99,
             0.995, 0.998, 0.999, 0.9999]
# the periods of which the climate change signal / response is computed over
historical_period = slice("1985","2014")
future_period = slice("2070","2099")
future_period2 = slice("2040","2069")

def compute_quantiles(df,quantiles, period):
    df = df.sel(time = period)
    # this removes instances which have negative precipitation (os the minimum value is -0.0001) 
    # due to the leakyrelu activation function in GANs
    df = df.where(df>0.0, 0.0)
    seasonal_rainfall = df.groupby('time.season').mean()
    df = df.where(df>1, np.nan)
    quantiled_rain = df.quantile(q = quantiles, dim =["time"], skipna =True)
    return quantiled_rain, seasonal_rainfall


def compute_signal(df, quantiles, historical_period, future_period):

    historical_quantiles, seasonal_rainfall = compute_quantiles(df, quantiles, historical_period)
    future_quantiles, future_rainfall = compute_quantiles(df, quantiles, future_period)

    cc_signal = 100 * (future_rainfall - seasonal_rainfall)/seasonal_rainfall
    signal = 100 * (future_quantiles - historical_quantiles)/historical_quantiles
    historical_quantiles = historical_quantiles.rename({"pr":"hist_quantiles"})
    future_quantiles = future_quantiles.rename({"pr": "future_quantiles"})
    seasonal_rainfall = seasonal_rainfall.rename({"pr":"hist_clim_rainfall"})
    future_rainfall = future_rainfall.rename({"pr":"future_clim_rainfall"})
    signal = signal.rename({"pr":"cc_signal"})
    cc_signal = cc_signal.rename({"pr":"seas_cc_signal"})
    dset = xr.merge([historical_quantiles, future_quantiles,
                     signal, cc_signal, seasonal_rainfall, future_rainfall])
    return dset
# Please ensure that your config file has been modified so that it has the directories that match your files

stacked_X, y, vegt, orog, he = preprocess_input_data(config_test_data, match_index =False)

gan, unet, adv_factor = load_model_cascade(config["model_name"], None, './models', load_unet=True)
try:
    y = y.isel(GCM =0)[['pr']]
except:
    y =y[['pr']]
for gcm in stacked_X.GCM.values:
    print(f"prepraring data fpr a GCM {gcm}")
    output_shape = create_output(stacked_X, y)
    output_shape.pr.values = output_shape.pr.values * 0.0
    
    # normalization is now with a 1
    output_hist = xr.concat([predict_parallel_resid(gan, unet,
                                   stacked_X.sel( GCM =gcm).transpose('time','lat','lon','channel').sel(time = historical_period).values,
                                   output_shape.sel(time = historical_period), 64, orog.values, he.values, vegt.values, model_type='GAN', output_add_factor = output_norm) for i in range(5)],
                            dim ="member")
    output_hist_reg = xr.concat([predict_parallel_resid(gan, unet,
                                   stacked_X.sel( GCM =gcm).transpose('time','lat','lon','channel').sel(time = historical_period).values,
                                   output_shape.sel(time = historical_period), 64, orog.values, he.values, vegt.values, model_type='UNET', output_add_factor = output_norm) for i in range(1)],
                            dim ="member")

    output_future = xr.concat([predict_parallel_resid(gan, unet,
                                         stacked_X.sel(GCM=gcm).transpose('time', 'lat', 'lon', 'channel').sel(
                                             time=future_period).values,
                                         output_shape.sel(time=future_period), 64, orog.values, he.values,
                                         vegt.values, model_type='GAN', output_add_factor = output_norm) for i in range(5)], dim ="member")
    output_future_reg = xr.concat([predict_parallel_resid(gan, unet,
                                         stacked_X.sel(GCM=gcm).transpose('time', 'lat', 'lon', 'channel').sel(
                                             time=future_period).values,
                                         output_shape.sel(time=future_period), 64, orog.values, he.values,
                                         vegt.values, model_type='UNET', output_add_factor = output_norm) for i in range(1)], dim ="member")
    
    output_future2 = xr.concat([predict_parallel_resid(gan, unet,
                                         stacked_X.sel(GCM=gcm).transpose('time', 'lat', 'lon', 'channel').sel(
                                             time=future_period2).values,
                                         output_shape.sel(time=future_period2), 64, orog.values, he.values,
                                         vegt.values, model_type='GAN', output_add_factor = output_norm) for i in range(5)], dim ="member")
    output_future_reg2 = xr.concat([predict_parallel_resid(gan, unet,
                                         stacked_X.sel(GCM=gcm).transpose('time', 'lat', 'lon', 'channel').sel(
                                             time=future_period2).values,
                                         output_shape.sel(time=future_period2), 64, orog.values, he.values,
                                         vegt.values, model_type='UNET', output_add_factor = output_norm) for i in range(1)], dim ="member")
    outputs = xr.concat([output_hist, output_future], dim ="time")
    outputs_reg = xr.concat([output_hist_reg, output_future_reg], dim="time")
    outputs_test = outputs.sel(time = slice("2098","2099"))
    outputs_reg_test = outputs_reg.sel(time=slice("2098", "2099"))
    outputs = compute_signal(outputs[['pr']], quantiles, historical_period, future_period)
    outputs_reg = compute_signal(outputs_reg[['pr']], quantiles, historical_period, future_period)
    
    outputs2 = xr.concat([output_hist, output_future2], dim ="time")
    outputs_reg2 = xr.concat([output_hist_reg, output_future_reg2], dim="time")
    outputs2 = compute_signal(outputs2[['pr']], quantiles, historical_period, future_period2)
    outputs_reg2 = compute_signal(outputs_reg2[['pr']], quantiles, historical_period, future_period2)
    
    #outputs.attrs['title'] = outputs.attrs['title'] + f'   /n ML Emulated NIWA-REMS GAN v1 GCM: {gcm}'
    if not os.path.exists(f'./outputs/{config["model_name"]}'):
        os.makedirs(f'./outputs/{config["model_name"]}')
    outputs.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_ens_v2.nc')
    outputs_reg.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_unet_v2.nc')
    outputs2.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_ens_mid_century_v2.nc')
    outputs_reg2.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_unet_mid_century_v2.nc')
    outputs_test.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_ens_test_sample_v2.nc')
    outputs_reg_test.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_unet_test_sample_v2.nc')
    with open(f'./outputs/{config["model_name"]}/config_info.json', 'w') as f:
        json.dump(config, f)
