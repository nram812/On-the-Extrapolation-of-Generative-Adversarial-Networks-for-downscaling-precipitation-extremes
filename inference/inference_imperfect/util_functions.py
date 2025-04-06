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
sys.path.append('/nesi/project/niwa00018/ML_downscaling_CCAM/DL_training_160325/multi-task-downscaling')
from src.layers import *
from src.models import *
from src.gan import *
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Ones, Zeros
class ComplexLinear(layers.Layer):
    def __init__(self, filters=16, weight_thres=0.25, n_modes=20, **kwargs):
        super(ComplexLinear, self).__init__(**kwargs)
        self.n_modes = n_modes
        self.filters = filters
        self.weight_thres = weight_thres

    @staticmethod
    def create_weighting_matrix(width, height):
        # Create meshgrid coordinates
        x_indices, y_indices = tf.meshgrid(tf.range(width), tf.range(height), indexing='ij')

        # Calculate the center coordinates
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0

        # Compute the distances from the center for each point
        distances = tf.sqrt(tf.square(tf.cast(x_indices, tf.float32) - center_x) +
                            tf.square(tf.cast(y_indices, tf.float32) - center_y))

        # Normalize the distance matrix by dividing by the maximum distance
        max_distance = tf.reduce_max(distances)
        normalized_distances = distances / max_distance

        return normalized_distances

    def build(self, input_shape):
        # Initialize real and imaginary parts separately
        self.units = (input_shape[1], input_shape[2], self.filters)
        self.weight_matrix = self.create_weighting_matrix(self.units[0], self.units[1])
        self.condition = tf.less(self.weight_matrix, self.weight_thres)
        self.n_modes = int(tf.reduce_sum(tf.cast(self.condition, 'float32')).numpy())

        self.real_weights = self.add_weight(
            shape=(input_shape[1], input_shape[2],input_shape[-1], self.filters),
            initializer='random_normal',
            trainable=True,
            name='real_kernel'
        )
        self.imag_weights = self.add_weight(
            shape=(input_shape[1], input_shape[2],input_shape[-1], self.filters),
            initializer='random_normal',
            trainable=True,
            name='imag_kernel'
        )

    #@tf.custom_gradient
    def custom_operation(self, real_kernel, imag_kernel, inputs):

        x_ft = tf.signal.fft2d(tf.cast(inputs,'complex64'))

        real_kernel_only = real_kernel
        complex_kernel_only = imag_kernel
        
        #out_real = torch.einsum('bi...,io...->bo...', input_real, weights_real) - torch.einsum('bi...,io...->bo...', input_imag, weights_imag)
        #out_imag = torch.einsum('bi...,io...->bo...', input_real, weights_imag) + torch.einsum('bi...,io...->bo...', input_imag, weights_real)
        real_part = tf.einsum('abcd,bcde-> abce', tf.cast(tf.math.real(x_ft), 'complex64'),  tf.cast(self.real_weights,'complex64'))
        real_part =real_part - tf.einsum('abcd,bcde-> abce', tf.cast(tf.math.imag(x_ft),'complex64'),  tf.cast(self.imag_weights, 'complex64'))
        imag_part = tf.einsum('abcd,bcde-> abce', tf.cast(x_ft,'complex64'),  tf.cast(self.imag_weights,'complex64')) + tf.einsum('abcd,bcde-> abce', tf.cast(tf.math.imag(x_ft),'complex64'),  tf.cast(self.real_weights,'complex64')) #x_ft * self.imag_weights + tf.math.imag(x_ft) * self.real_weights
        x_ft_transformed = tf.complex(tf.math.real(real_part), tf.math.imag(imag_part))
        real_output = tf.cast(tf.math.real(tf.signal.ifft2d(x_ft_transformed)), 'float32')

        return real_output#, grad
        # super(ComplexLinear, self).build(input_shape)

    def call(self, inputs):
        return self.custom_operation(self.real_weights, self.imag_weights, inputs)


class FourierLayer(tf.keras.layers.Layer):
    def __init__(self, n_fourier_filters, weight_thres, name=None, custom_activation = None, bn = False, **kwargs):
        super(FourierLayer, self).__init__(name=name, **kwargs)
        self.n_fourier_filters = n_fourier_filters
        self.weight_thres = weight_thres
        self.bn = bn
        if name is None:
            random_number=np.random.randint(0,100)
            name = f'test_{random_number}'
            # update name if random interger

        # Initialize layers here
        self.conv2d = tf.keras.layers.Conv2D(filters=n_fourier_filters[-1],
                                             kernel_size=3,
                                             padding='same',
                                             name=name + '_1')
        self.complex_linear1 = ComplexLinear(filters=n_fourier_filters[0],
                                             weight_thres=weight_thres[0], name=name + '_2')
        self.complex_linear2 = ComplexLinear(filters=n_fourier_filters[-1],
                                             weight_thres=weight_thres[1], name=name + '_3')
        self.add_layer = tf.keras.layers.Add()
        if custom_activation is not None:
            self.activation = custom_activation
        else:
            self.activation = tf.keras.layers.LeakyReLU(0.1)

    def call(self, input_vector):
        x1 = self.conv2d(input_vector)
        output = self.complex_linear1(input_vector)
        output = self.complex_linear2(output)
        residual_layer = self.add_layer([x1, output])
        activation = self.activation(residual_layer)
        #if self.bn:
        #    activation = tf.keras.layers.BatchNormalization()(activation)
        return activation

    def get_config(self):
        config = super(FourierLayer, self).get_config()
        config.update({
            'n_fourier_filters': self.n_fourier_filters,
            'weight_thres': self.weight_thres
        })
        return config


class DyT(Layer):
    def __init__(self, init_alpha=1.0, **kwargs):
        super(DyT, self).__init__(**kwargs)
        self.init_alpha = init_alpha

    def build(self, input_shape):
        # Trainable parameters
        print(input_shape[1:-1])
        self.alpha = self.add_weight(
            shape=(input_shape[1:-1]), initializer=tf.constant_initializer(self.init_alpha), trainable=True, name="alpha"
        )
        self.gamma = self.add_weight(
            shape=(input_shape[-1]), initializer=Ones(), trainable=True, name="gamma"
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1]), initializer=Zeros(), trainable=True, name="beta"
        )
        #self.beta = tf.expand_dims(self.beta, axis =0)
        
    def call(self, x):
        alpha = tf.expand_dims(tf.expand_dims(self.alpha, axis =-1), axis =0)
        gamma = self.gamma#tf.expand_dims(tf.expand_dims(self.gamma, axis =-1), axis =0)
        x = tf.math.tanh(alpha * x)  # Apply scaled tanh
        return gamma * x + self.beta  # Apply learned scaling and shift
    def get_config(self):
        """ Required for saving the model """
        config = super(DyT, self).get_config()
        config.update({
            "init_alpha": self.init_alpha
        })
        return config
# from ops.model_inference.src_eval_inference import

def load_model(name, model_path):
    generator = tf.keras.models.load_model(f'{model_path}/{name}/generator_best_weights.h5',
                                           custom_objects={"BicubicUpSampling2D": BicubicUpSampling2D,
                                                           "SymmetricPadding2D": SymmetricPadding2D,"FourierLayer":FourierLayer,
                                                           "ComplexLinear":ComplexLinear})
    unet_model = tf.keras.models.load_model(f'{model_path}/{name}/unet_best_weights.h5',
                                            custom_objects={"BicubicUpSampling2D": BicubicUpSampling2D,
                                                            "SymmetricPadding2D": SymmetricPadding2D, "FourierLayer":FourierLayer})
    return generator, unet_model


@tf.function
def predict_batch_residual_single(model, unet, latent_vectors, data_batch, orog, time_of_year, spatial_means,
                                  spatial_stds, gan=True):
    rain = unet([data_batch, orog, time_of_year, spatial_means, spatial_stds], training=False)

    if gan:
        rain_resid = model(
            [latent_vectors[0], latent_vectors[1], data_batch, orog, rain, time_of_year, spatial_means, spatial_stds],
            training=False)
        # multiple residuals
        rain = rain + rain_resid

    return rain

def get_files(input_data_path_base, gcm, ssp, variant, output_dir, variable, configs):
    if 'ssp' in ssp:
        files = f'{input_data_path_base}/ScenarioMIP/*/{gcm}/{ssp}/{variant}/day/ScenarioMIP_*_{gcm}_{ssp}_{variant}_day*.nc'
    else:
        files = f'{input_data_path_base}/CMIP/*/{gcm}/{ssp}/{variant}/day/CMIP_*_{gcm}_{ssp}_{variant}_day*.nc'
    files = glob.glob(files, recursive =True)
    filename = files[0].split('/')[-1]
    version = filename.split('_')[-1].strip('.nc')
    grid_label =filename.split('_')[-2]
    new_filename = f"{variable}_{filename.replace(version, configs[variable]).replace(grid_label,'NZ12km')}"
    output_path = f"{output_dir}{files[0].split(input_data_path_base)[1]}".split(filename)[0]
    output_path = f'{output_path}{variable}/NZ_Domain/{version}/{new_filename}'
    return files[0], output_path


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

            rain = predict_batch_residual_single(model, unet,
                                                 [random_latent_vectors1, random_latent_vectors2],
                                                 data_batch, orog, time_of_year[i * batch_size: (i + 1) * batch_size],
                                                 spatial_means[i * batch_size: (i + 1) * batch_size],
                                                 spatial_stds[i * batch_size: (i + 1) * batch_size], gan=gan)

            if varname == "sfcwind":
                rainfall += ((rain.numpy()[:, :, :, 0] + min_value['sfcWind'].values) * stds['sfcWind'].mean().values +
                             means['sfcWind'].mean().values).tolist()
            elif varname == 'sfcwindmax':
                rainfall += ((rain.numpy()[:, :, :, 0] + min_value['sfcWindmax'].values) * stds[
                    'sfcWindmax'].mean().values + means['sfcWindmax'].mean().values).tolist()
            elif varname == "pr":
                rainfall_instant = np.exp(rain.numpy()[:, :, :, 0] + min_value['pr'].values) - 1
                rainfall_instant = np.clip(rainfall_instant, a_min=0, a_max=None)
                rainfall += (rainfall_instant).tolist()
            else:
                rainfall += ((rain.numpy()[:, :, :, 0] + min_value[varname].values) * stds[varname].mean().values +
                             means[varname].mean().values).tolist()

            pbar.update(1)

    if remainder != 0:
        tf.random.set_seed(np.random.randint(0, 10000))
        random_latent_vectors1 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[0].shape[1:]))
        random_latent_vectors2 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[1].shape[1:]))
        # random_latent_vectors2 = tf.repeat(random_latent_vectors2, repeats=batch_size, axis=0)
        orog = expand_conditional_inputs(orog_vector, remainder)
        rain = predict_batch_residual_single(model, unet, [
            random_latent_vectors1[:remainder], random_latent_vectors2[:remainder]],
                                             inputs[
                                             inputs.shape[0] - remainder:],
                                             orog, time_of_year[inputs.shape[0] - remainder:],
                                             spatial_means[inputs.shape[0] - remainder:],
                                             spatial_stds[inputs.shape[0] - remainder:], gan=gan)

        if varname == "sfcwind":
            rainfall += ((rain.numpy()[:, :, :, 0] + min_value['sfcWind'].values) * stds['sfcWind'].mean().values +
                         means['sfcWind'].mean().values).tolist()
        elif varname == 'sfcwindmax':
            rainfall += (
                        (rain.numpy()[:, :, :, 0] + min_value['sfcWindmax'].values) * stds['sfcWindmax'].mean().values +
                        means['sfcWindmax'].mean().values).tolist()
        elif varname == "pr":
            rainfall_instant = np.exp(rain.numpy()[:, :, :, 0] + min_value['pr'].values) - 1
            rainfall_instant = np.clip(rainfall_instant, a_min=0, a_max=None)
            rainfall += (rainfall_instant).tolist()
        else:
            rainfall += ((rain.numpy()[:, :, :, 0] + min_value[varname].values) * stds[varname].mean().values + means[
                varname].mean().values).tolist()

    output_shape[varname] = (('time', 'lat', 'lon'), rainfall)

    return output_shape


'''
MY FUNCTIONS FOR PREPROCESSING INPUTS
'''


def prepare_ML_inputs(GCM_input_path, config, framework):
    ds = xr.open_mfdataset(GCM_input_path)
    with ProgressBar():
        ds = ds.load()

    print('\t- processing GCM input data')
    processed_GCM_data = reformat_GCM_data(ds, config, framework)
    print('\t- processing mean, variance, and time data')
    mean_data, variance_data, time_of_year = process_mean_variance_time(ds, config)
    print('\t- processing static fields')
    vegt, orog, he = prepare_static_fields(config)
    print('\t- calculating time of year array')

    return (processed_GCM_data, mean_data, variance_data, vegt, orog, he, time_of_year)


def reformat_GCM_data(ds, config, framework):
    if framework == 'imperfect':
        # Step 1: unstack pressure levels, change variable names
        print('\t\t- unstacking pressure levels')
        ds = unstack_pressure_levels(ds)

    # Step 2: normalize dataset by mean and st. dev.
    print('\t\t- normalizing by mean and standard deviation')
    ds = normalize(ds, config)

    # Step 3: concatenate variable dimension
    print('\t\t- concatenating variables to channel dimension')
    da = concatenate_variable_dimension(ds, config)

    return (da)


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


def normalize(ds, config):
    var_list = config['var_names']
    # norm_ds = ds[var_list].copy() * np.nan
    # for variable in var_list:
    #     print(variable)
    #     means = ds[variable].mean(['lat', 'lon'])
    #     stds = ds[variable].std(['lat', 'lon'])
    #     norm_ds[variable] = (ds[variable] - means)/stds

    norm_ds = (ds[var_list] - ds[var_list].mean(['lat', 'lon'])) / ds[var_list].std(['lat', 'lon'])

    return (norm_ds)


def concatenate_variable_dimension(ds, config):
    var_list = config['var_names']

    # concatenate dataarrays of each variables along dimension 'channel'
    concatenated_da = xr.concat([ds[var] for var in var_list], dim="channel")
    concatenated_da = concatenated_da.rename('GCM_da')

    # name channel dimension values by variables names
    concatenated_da['channel'] = (('channel'), var_list)

    return (concatenated_da)


def process_mean_variance_time(ds, config):
    var_list = config['var_names']
    framework ="imperfect"
    if framework == 'imperfect':
        ds = unstack_pressure_levels(ds)

    ds = ds[var_list]

    # single values
    predictor_means_mean = xr.open_dataset(config["input_means_means"])
    predictor_means_variance = xr.open_dataset(config["input_means_stds"])

    predictor_stds_mean = xr.open_dataset(config["input_stds_means"])
    predictor_stds_variance = xr.open_dataset(config["input_stds_stds"])

    # time series
    GCM_spatial_means = ds.mean(['lat', 'lon'])
    GCM_spatial_stds = ds.std(['lat', 'lon'])

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

def get_files(input_data_path_base, gcm, ssp, variant, output_dir, variable, configs):
    if 'ssp' in ssp:
        files = f'{input_data_path_base}/ScenarioMIP/*/{gcm}/{ssp}/{variant}/day/ScenarioMIP_*_{gcm}_{ssp}_{variant}_day*.nc'
    else:
        files = f'{input_data_path_base}/CMIP/*/{gcm}/{ssp}/{variant}/day/CMIP_*_{gcm}_{ssp}_{variant}_day*.nc'
    files = glob.glob(files, recursive =True)
    filename = files[0].split('/')[-1]
    version = filename.split('_')[-1].strip('.nc')
    grid_label =filename.split('_')[-2]
    new_filename = f"{variable}_{filename.replace(version, configs[variable]).replace(grid_label,'NZ12km')}"
    output_path = f"{output_dir}{files[0].split(input_data_path_base)[1]}".split(filename)[0]
    output_path = f'{output_path}{variable}/NZ_Domain/{version}/{new_filename}'
    return files[0], output_path

