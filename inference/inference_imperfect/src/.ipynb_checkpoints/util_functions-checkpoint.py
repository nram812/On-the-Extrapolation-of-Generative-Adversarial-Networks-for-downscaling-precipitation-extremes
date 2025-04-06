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

'''
add src / ops directory to path, import functions
'''
#sys.path.append('/nesi/project/niwa03712/queenle/ML_emulator/code/applying_ml_model/v3')
''' Due to updates in the ML downscaling repo I have modified this'''
# sys.path.append('/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan')

# from src.layers import *
# from src.models import *
# from src.gan import *

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow
import xarray as xr
from dask.diagnostics import ProgressBar
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd

sys.path.append('/nesi/project/niwa00018/ML_downscaling_CCAM/DL_training_160325/multi-task-downscaling')
from src.layers import *
from src.models import *
from src.gan import *
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Ones, Zeros


def load_model(name, model_path):
    """Modified 05/04/25"""
    generator = tf.keras.models.load_model(f'{model_path}/{name}/generator_final.h5',
                                           custom_objects={"BicubicUpSampling2D": BicubicUpSampling2D,
                                                           "SymmetricPadding2D": SymmetricPadding2D,"FourierLayer":FourierLayer,
                                                           "ComplexLinear":ComplexLinear})
    unet_model = tf.keras.models.load_model(f'{model_path}/{name}/unet_final.h5',
                                            custom_objects={"BicubicUpSampling2D": BicubicUpSampling2D,
                                                            "SymmetricPadding2D": SymmetricPadding2D, "FourierLayer":FourierLayer,
                                                            "ComplexLinear":ComplexLinear})
    return generator, unet_model


def get_files(input_data_path_base, gcm, ssp, variant, output_dir, variable, configs):
    
    if 'ssp' in ssp:
        files = f'{input_data_path_base}/ScenarioMIP/*/{gcm}/{ssp}/{variant}/day/ScenarioMIP_*_{gcm}_{ssp}_{variant}_day_??_v????????.nc'
    else:
        files = f'{input_data_path_base}/CMIP/*/{gcm}/{ssp}/{variant}/day/CMIP_*_{gcm}_{ssp}_{variant}_day_??_v????????.nc'
    files = glob.glob(files, recursive =True)
    filename = files[0].split('/')[-1]
    version = filename.split('_')[-1].strip('.nc')
    grid_label =filename.split('_')[-2]
    new_filename = f"{variable}_{filename.replace(version, configs[variable]).replace(grid_label,'NZ12km')}"
    output_path = f"{output_dir}{files[0].split(input_data_path_base)[1]}".split(filename)[0]
    output_path = f'{output_path}{variable}/NZ_Domain/{version}/{new_filename}'
    return files[0], output_path

@tf.function
def predict_batch_residual_single(model, unet, latent_vectors, data_batch, orog, time_of_year, spatial_means,
                                  spatial_stds, gan=True, varname = "pr"):
    rain = unet([data_batch, orog], training =False)#, time_of_year, spatial_means, spatial_stds], training=False)

    if gan:

        rain_resid = model(
            [latent_vectors[0], latent_vectors[1], data_batch, orog, rain], training=False)#, time_of_year, spatial_means, spatial_stds],
        if varname == "pr":
            rain_resid  = tf.clip_by_value(rain_resid, -10,5
)
                #training=False)
        # multiple residuals
        rain = rain + rain_resid

    return rain


def expand_conditional_inputs(X, batch_size):
    expanded_image = tf.expand_dims(X, axis=0)  # Shape: (1, 172, 179)

    # Repeat the image to match the desired batch size
    expanded_image = tf.repeat(expanded_image, repeats=batch_size, axis=0)  # Shape: (batch_size, 172, 179)

    # Create a new axis (1) on the last axis
    expanded_image = tf.expand_dims(expanded_image, axis=-1)
    return expanded_image

def predict_parallel_resid_corrector_v4varname(model, unet, inputs, output_shape, batch_size, orog_vector,
                                               means, stds, time_of_year, spatial_means, spatial_stds, gan=True,
                                               min_value=None, varname='tasmax'):
    """Modified on 06/04/25, predict_batch_residual_single"""
    n_iterations = inputs.shape[0] // batch_size
    remainder = inputs.shape[0] - n_iterations * batch_size

    rainfall = []
    sfcwinds = []
    sfcwindmaxs = []
    tasmaxs = []
    tasmins = []

    with tqdm.tqdm(total=n_iterations, desc="Predicting", unit="batch") as pbar:

        for i in range(n_iterations):
            tf.random.set_seed(np.random.randint(0, 10000))
            data_batch = inputs[i * batch_size: (i + 1) * batch_size]
            random_latent_vectors1 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[0].shape[1:]))

            random_latent_vectors2 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[1].shape[1:]))
            # print(random_latent_vectors1.numpy()[0, 1, 1, 1], random_latent_vectors2.numpy()[0, 1, 1, 1])
            orog = expand_conditional_inputs(orog_vector, batch_size)  # ex, he_vector, vegt_vector

#             rain = predict_batch_residual_single(model, unet,
#                                                  [random_latent_vectors1, random_latent_vectors2],
#                                                  data_batch, orog, time_of_year[i * batch_size: (i + 1) * batch_size],
#                                                  spatial_means[i * batch_size: (i + 1) * batch_size],
#                                                  spatial_stds[i * batch_size: (i + 1) * batch_size], gan=gan)
            # In the line below, the time_of_year, spatial_means, and stds are "None" objects
            rain = predict_batch_residual_single(model, unet,
                                                 [random_latent_vectors1, random_latent_vectors2],
                                                 data_batch, orog, time_of_year,
                                                 spatial_means,
                                                 spatial_stds, gan=gan, varname = varname)

            if varname == "sfcWind":
                rainfall += ((rain.numpy()[:, :, :, 0]) * stds['sfcWind'].mean().values +
                             means['sfcWind'].mean().values).tolist()
            elif varname == 'sfcWindmax':
                rainfall += ((rain.numpy()[:, :, :, 0]) * stds[
                    'sfcWind'].mean().values + means['sfcWind'].mean().values).tolist()
            elif varname == "pr":
                rainfall_instant = np.exp(rain.numpy()[:, :, :, 0]) - 1
                rainfall_instant = np.clip(rainfall_instant, a_min=0, a_max=1500)
                rainfall += (rainfall_instant).tolist()
            else:
                rainfall += ((rain.numpy()[:, :, :, 0]) * stds[varname].mean().values +
                             means[varname].mean().values).tolist()

            pbar.update(1)

    if remainder != 0:
        tf.random.set_seed(np.random.randint(0, 10000))
        random_latent_vectors1 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[0].shape[1:]))
        random_latent_vectors2 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[1].shape[1:]))
        # random_latent_vectors2 = tf.repeat(random_latent_vectors2, repeats=batch_size, axis=0)
        orog = expand_conditional_inputs(orog_vector, remainder)
#         rain = predict_batch_residual_single(model, unet, [
#             random_latent_vectors1[:remainder], random_latent_vectors2[:remainder]],
#                                              inputs[
#                                              inputs.shape[0] - remainder:],
#                                              orog, time_of_year[inputs.shape[0] - remainder:],
#                                              spatial_means[inputs.shape[0] - remainder:],
#                                              spatial_stds[inputs.shape[0] - remainder:], gan=gan)
        # The below are "None" objects
        rain = predict_batch_residual_single(model, unet, [
            random_latent_vectors1[:remainder], random_latent_vectors2[:remainder]],
                                             inputs[
                                             inputs.shape[0] - remainder:],
                                             orog, time_of_year,
                                             spatial_means,
                                             spatial_stds, gan=gan, varname = varname)

        if varname == "sfcWind":
            rainfall += ((rain.numpy()[:, :, :, 0]) * stds['sfcWind'].mean().values +
                         means['sfcWind'].mean().values).tolist()
        elif varname == 'sfcWindmax':
            rainfall += ((rain.numpy()[:, :, :, 0]) * stds['sfcWind'].mean().values +
                         means['sfcWind'].mean().values).tolist()
        elif varname == "pr":
            rainfall_instant = np.exp(rain.numpy()[:, :, :, 0]) - 1
            rainfall_instant = np.clip(rainfall_instant, a_min=0, a_max=1500)
            rainfall += (rainfall_instant).tolist()
        else:
            rainfall += ((rain.numpy()[:, :, :, 0]) * stds[varname].mean().values + means[
                varname].mean().values).tolist()

    output_shape[varname] = (('time', 'lat', 'lon'), rainfall)

    return output_shape


'''
MY FUNCTIONS FOR PREPROCESSING INPUTS
'''


def prepare_ML_inputs(GCM_input_path, config, framework, spectral_filtering = False, method ='normal_means'):
    """Modified on 06/04/25"""
    with ProgressBar():
        ds = xr.open_mfdataset(GCM_input_path).load()

        print('\t- processing GCM input data')
        processed_GCM_data, means, stds = reformat_GCM_data(ds, config, framework, spectral_filtering = spectral_filtering, method = method)
        processed_GCM_data = processed_GCM_data.load()
        means = means.load()
        stds = stds.load()
        print('\t- processing mean, variance, and time data')
        mean_data, variance_data, time_of_year = process_mean_variance_time(ds, config, framework, means, stds)
        mean_data = mean_data.load()
        variance_data = variance_data.load()
        time_of_year = time_of_year.load()
        print('\t- processing static fields')
        vegt, orog, he = prepare_static_fields(config)
        vegt = vegt.load()
        orog = orog.load()
        he = he.load()
        print('\t- calculating time of year array')

    return (processed_GCM_data, mean_data, variance_data, vegt, orog, he, time_of_year)


def reformat_GCM_data(ds, config, framework, spectral_filtering = False, method ='doury'):
    if framework == 'imperfect':
        # Step 1: unstack pressure levels, change variable names
        print('\t\t- unstacking pressure levels')
        ds = unstack_pressure_levels(ds)

    # Step 2: normalize dataset by mean and st. dev.
    print('\t\t- normalizing by mean and standard deviation')
    ds, means, stds = normalize(ds, config, spectral_filtering = spectral_filtering, spectral_threshold=0.48, method =method)

    # Step 3: concatenate variable dimension
    print('\t\t- concatenating variables to channel dimension')
    da = concatenate_variable_dimension(ds, config)

    return da, means, stds


def unstack_pressure_levels(ds):
    unstacked_ds = ds.copy()

    var_name_dict = {'hus': 'q', 'ta': 't', 'ua': 'u', 'va': 'v'}

    for var in ['hus', 'ta', 'ua', 'va']:
        for lev in unstacked_ds.plev.values:
            if lev > 1e4:
                data = unstacked_ds[var].sel(plev=lev)
                lev = int(lev / 100)
            else:
                data = unstacked_ds[var].sel(plev=lev)

            unstacked_ds[f'{var_name_dict[var]}_{int(lev)}'] = data

        unstacked_ds = unstacked_ds.drop([var])

    return (unstacked_ds)


def normalize(ds, config, spectral_filtering = False, spectral_threshold=0.48, method ='normal_means'):
    var_list = config['var_names']
    
    if spectral_filtering:
        print('\t\t\t- Spectral filtering of the input variables')
        ds_new = ds[var_list].copy() * np.nan
        for varname in var_list:
            ds_new[varname].data = low_pass_filter(ds[varname].transpose("time","lat","lon").values, cutoff = spectral_threshold)
            print(f"Spectrally modified variable: {varname}")
        ds = ds_new 
    if method == 'doury':    
        means = ds[var_list].mean(['lat', 'lon'])
        stds = ds[var_list].std(['lat', 'lon'])
        norm_ds = (ds[var_list] - means) / stds
    else:
        means_NORM = xr.open_dataset(config["mean"])
        stds_NORM = xr.open_dataset(config["std"])
        norm_ds = (ds[var_list] - means_NORM.mean(["lat","lon"])) / stds_NORM.mean(["lat","lon"])
        means = ds[var_list].mean(['lat', 'lon'])
        stds = ds[var_list].std(['lat', 'lon'])

    return norm_ds, means, stds



def concatenate_variable_dimension(ds, config):
    var_list = config['var_names']

    # concatenate dataarrays of each variables along dimension 'channel'
    concatenated_da = xr.concat([ds[var] for var in var_list], dim="channel")
    concatenated_da = concatenated_da.rename('GCM_da')

    # name channel dimension values by variables names
    concatenated_da['channel'] = (('channel'), var_list)

    return (concatenated_da)


def process_mean_variance_time(ds, config, framework, GCM_spatial_means, GCM_spatial_stds):
    var_list = config['var_names']

    if framework == 'imperfect':
        ds = unstack_pressure_levels(ds)

    ds = ds[var_list]

    # single values
    predictor_means_mean = xr.open_dataset(config["input_means_means"])
    predictor_means_variance = xr.open_dataset(config["input_means_stds"])

    predictor_stds_mean = xr.open_dataset(config["input_stds_means"])
    predictor_stds_variance = xr.open_dataset(config["input_stds_stds"])

    # time series
    # = ds.mean(['lat', 'lon'])
    #GCM_spatial_stds = ds.std(['lat', 'lon'])

    # stack normalized means
    norm_spatial_means = (GCM_spatial_means - predictor_means_mean) / predictor_means_variance
    norm_spatial_means = xr.concat([norm_spatial_means[i] for i in var_list], dim="channel")
    norm_spatial_means['channel'] = (('channel'), var_list)

    # stack normalized st. deviations
    norm_spatial_stds = (GCM_spatial_stds - predictor_stds_mean) / predictor_stds_variance
    norm_spatial_stds = xr.concat([norm_spatial_stds[i] for i in var_list], dim="channel")
    norm_spatial_stds['channel'] = (('channel'), var_list)

    time_of_year = np.sin(2 * np.pi * norm_spatial_means.time.dt.dayofyear / 365)

    return (norm_spatial_means, norm_spatial_stds, time_of_year)


def prepare_static_fields(config):
    topography_data = xr.open_dataset(config["static_predictors"])

    vegt = topography_data.vegt
    orog = topography_data.orog
    he = topography_data.he

    # normalize to the range [0,1]
    vegt = (vegt - vegt.min()) / (vegt.max() - vegt.min())
    orog = (orog - orog.min()) / (orog.max() - orog.min())
    he = (he - he.min()) / (he.max() - he.min())

    return (vegt, orog, he)


def initialize_output_ds(input_ds, config):
    print('\t- initializing output data structure')

    example_output = xr.open_dataset(config['train_y'])

    try:
        example_output = example_output.isel(GCM=0)[['pr']]
    except:
        example_output = example_output[['pr']]

    output_shape = example_output.isel(time=0).drop(['time'])
    output_shape = output_shape.expand_dims({"time": input_ds.time.size})
    output_shape['time'] = (('time'), input_ds.time.to_index())

    output_shape.pr.values = output_shape.pr.values * 0

    return (output_shape)


def determine_file_count(frameworks, variables, GCMs, scens):
    count = 0
    if 'imperfect' in frameworks:
        count += len(variables) * len(GCMs) * len(scens)

    if 'perfect' in frameworks:
        if 'historical' in scens:
            count += len(variables) * len(GCMs) * (
                        len(scens) - 1)  # don't count historical scens, will be skipped because coarsened RCM files have hist+ssp combined
        else:
            count += len(variables) * len(GCMs) * len(scens)

    return str(count)


