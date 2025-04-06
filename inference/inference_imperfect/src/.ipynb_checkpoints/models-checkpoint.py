import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow
import xarray as xr
from dask.diagnostics import ProgressBar
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from src.layers import res_block_initial, BicubicUpSampling2D,upsample, conv_block,decoder_noise,down_block,up_block,SymmetricPadding2D 

from tensorflow.keras.layers import Lambda


def unet_linear_v6(input_size, resize_output, num_filters, kernel_size, num_channels=8, num_classes=1, resize=True,
                          final_activation = tf.keras.layers.LeakyReLU(1)):

    """have modified the architecture with a new concatenation layer"""

    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])

    concat_image = high_res_fields#tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2,high_res_fields3])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    inputs_abstract = low_res#tf.keras.layers.Concatenate(-1)([low_res, noise])
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=5, i =0)
    x, temp2 = down_block(x, num_filters[1], kernel_size=5, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=5, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x])
    # decode
    x = up_block(x, temp3, kernel_size=5, filters = num_filters[2], i =0, concat = False)
    x = up_block(x, temp2, kernel_size=5, filters = num_filters[1], i =2, concat = False)
    x = up_block(x, temp1, kernel_size=5, filters = num_filters[0], i =3, concat = False)
    output = tf.image.resize(x, (resize_output[0], resize_output[1]),
                    method=tf.image.ResizeMethod.BILINEAR)
    output = res_block_initial(output, [64], 3, [1, 1], "output_convbbb1234567", sym_padding=True)
    output = res_block_initial(output, [32], 3, [1, 1], "output_convbbb12347", sym_padding=True)
    output = SymmetricPadding2D(padding=[1, 1])(output)
    output = tf.keras.layers.Conv2D(32, 3, activation=final_activation, padding ='valid')(output)
    output = SymmetricPadding2D(padding=[1, 1])(output)
    output = tf.keras.layers.Conv2D(16, 3, activation=final_activation, padding ='valid')(output)
    output = tf.keras.layers.Conv2D(1, 1, activation=final_activation, padding ='valid')(output)
    output = tf.image.resize(output, (resize_output[0], resize_output[1]),
                    method=tf.image.ResizeMethod.BILINEAR)
    input_layers = [low_res, high_res_fields]
    model = tf.keras.models.Model(input_layers, output, name='unet')
    model.summary()
    return model
def res_linear_activation_v6(input_size, resize_output, num_filters, kernel_size, num_channels, num_classes, resize=True,
                          final_activation = tf.keras.layers.LeakyReLU(1)):

    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields4 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])

    concat_image = tf.keras.layers.Concatenate(-1)([high_res_fields,  high_res_fields4])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])
    inputs_abstract = tf.keras.layers.Concatenate(-1)([low_res, noise])
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=3, i =0)
    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x])
    # decode
    x = up_block(x, temp3, kernel_size=3, filters = num_filters[2], i =0)
    noise2 = tf.keras.layers.Input(shape=[ x.shape[1],  x.shape[2], 16])
    x = tf.keras.layers.Concatenate(-1)([x, noise2])
    x = up_block(x, temp2, kernel_size=3, filters = num_filters[1], i =2)
    x = up_block(x, temp1, kernel_size=3, filters = num_filters[0], i =3)
    output = tf.image.resize(x, (resize_output[0], resize_output[1]),
                    method=tf.image.ResizeMethod.BILINEAR)
    output = res_block_initial(output, [64], 3, [1, 1], "output_convbbb123456", sym_padding=True)
    output = res_block_initial(output, [32], 3, [1, 1], "output_convbbb1234", sym_padding=True)
    output = SymmetricPadding2D(padding=[1, 1])(output)
    output = tf.keras.layers.Conv2D(32, 3, activation=final_activation, padding ='valid')(output)
    output = SymmetricPadding2D(padding=[1, 1])(output)
    output = tf.keras.layers.Conv2D(16, 3, activation=final_activation, padding ='valid')(output)
    output = tf.keras.layers.Conv2D(num_classes, 1, activation=final_activation, padding ='valid')(output)
#     output = tf.image.resize(output, (resize_output[0], resize_output[1]),
#                     method=tf.image.ResizeMethod.BILINEAR)
    input_layers = [noise, noise2] + [low_res, high_res_fields, high_res_fields4]
    model = tf.keras.models.Model(input_layers, output, name='unet')
    model.summary()
    return model

# Define a Lambda layer to compute the mean across specified axes
def get_discriminator_model_v4(high_resolution_fields_size,
                            low_resolution_fields_size, use_bn=False,
                            use_dropout=False, use_bias=True, low_resolution_feature_channels=(32, 64, 128),
                            low_resolution_dense_neurons =6,
                            high_resolution_feature_channels=(16, 32, 64, 128)):
    """
    Discriminator no longer uses the unet model to demonstrate realism
    **Purpose:**
      * To create a discriminator model that takes two streams of inputs, one from the low resolution predictor fields(X)
      and auxilary inputs (topography), it also takes in the high-resolution "regression prediction",
      which is used for residuals

    **Parameters:**
      * **high_resolution_fields_size (tuple):**  The size of the 2D high-resolution RCM fields, over the NZ region this (172, 179)
      * **low_resolution_fields_size (tuple):**  The size of the 2D low-resolution predictor fields (23, 26) over the New Zealand domain
      * **use_bn (bool, optional):** whether to use batchnormalization or not (default no bn)
      * **use_dropout (bool, optional):** whether to use dropout or not(default no dropout)
      * **use_bias (bool, optional):** whether to use bias or not (default bias =True)

    **Returns:**
        * a tf.keras.models.Model class

    **Example Usage:**
    ```python
    discriminator_model = get_discriminator_model((172, 179), (23, 26))
    ```
    """
    IMG_SHAPE = high_resolution_fields_size
    IMG_SHAPE2 = low_resolution_fields_size

    img_input = layers.Input(shape=IMG_SHAPE) # real or fake predictions
    img_input2 = layers.Input(shape=IMG_SHAPE2) # boundary conditions or predictor fields

    # these are static inputs to the model
    img_input3 = layers.Input(shape=IMG_SHAPE) # Topography predictor variable
    img_input6 = layers.Input(shape=IMG_SHAPE) # UNET regressoin predictor.
    # now we concatenate these input a single vector
    # high-resolution data stream
    # first we put "real or fake data" with 32 channels, to allow it to be more important
    inputs_high_res = res_block_initial(img_input, [high_resolution_feature_channels[0]], 3, [1, 1], "output_convbbb")
    x = conv_block(inputs_high_res, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(2, 2),
                   use_bn=use_bn, use_bias=use_bias,
                   use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())

    x = conv_block(x, high_resolution_feature_channels[2], kernel_size=(3, 3), strides=(2, 2),
                   use_bn=use_bn, use_bias=use_bias,
                   use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    # reducing the dimensionality to speed up the computational cost
    # x = tf.keras.layers.AveragePooling2D((3,3))(x)

    x_init_raw = conv_block(x, high_resolution_feature_channels[3], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    images_low_res = conv_block(img_input2, high_resolution_feature_channels[0], kernel_size=(3, 3), strides=(1, 1),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    images_low_res = conv_block(images_low_res, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(1, 1),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    images_low_res = tf.image.resize(images_low_res, [x_init_raw.shape[1], x_init_raw.shape[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
    concat_outputs = tf.keras.layers.Concatenate(-1)([x_init_raw, images_low_res])
    concat_outputs = res_block_initial(concat_outputs, [high_resolution_feature_channels[3]], 3, [1, 1], "output_convbbb1")
    x_init_raw = conv_block(concat_outputs, high_resolution_feature_channels[3], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0,
                            activation=tf.keras.layers.LeakyReLU())
    x_init_raw = conv_block(x_init_raw, high_resolution_feature_channels[3], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0,
                            activation=tf.keras.layers.LeakyReLU())
    flattened_output = tf.keras.layers.Flatten()(x_init_raw)
    dense2 = tf.keras.layers.Dense(32)(flattened_output)

    x = layers.Dense(1)(dense2)

    d_model = keras.models.Model([img_input, img_input2, img_input3, img_input6], x,
                                 name="discriminator")
    return d_model

def get_discriminator_model_v1(high_resolution_fields_size,
                            low_resolution_fields_size, use_bn=False,
                            use_dropout=False, use_bias=True, low_resolution_feature_channels=(32, 64, 128),
                            low_resolution_dense_neurons =6,
                            high_resolution_feature_channels=(16, 32, 64, 128)):
    """
    Discriminator no longer uses the unet model to demonstrate realism
    **Purpose:**
      * To create a discriminator model that takes two streams of inputs, one from the low resolution predictor fields(X)
      and auxilary inputs (topography), it also takes in the high-resolution "regression prediction",
      which is used for residuals

    **Parameters:**
      * **high_resolution_fields_size (tuple):**  The size of the 2D high-resolution RCM fields, over the NZ region this (172, 179)
      * **low_resolution_fields_size (tuple):**  The size of the 2D low-resolution predictor fields (23, 26) over the New Zealand domain
      * **use_bn (bool, optional):** whether to use batchnormalization or not (default no bn)
      * **use_dropout (bool, optional):** whether to use dropout or not(default no dropout)
      * **use_bias (bool, optional):** whether to use bias or not (default bias =True)

    **Returns:**
        * a tf.keras.models.Model class

    **Example Usage:**
    ```python
    discriminator_model = get_discriminator_model((172, 179), (23, 26))
    ```
    """
    IMG_SHAPE = high_resolution_fields_size
    IMG_SHAPE2 = low_resolution_fields_size

    img_input = layers.Input(shape=IMG_SHAPE) # real or fake predictions
    img_input2 = layers.Input(shape=IMG_SHAPE2) # boundary conditions or predictor fields

    # these are static inputs to the model
    img_input3 = layers.Input(shape=IMG_SHAPE) # Topography predictor variable
    img_input4 = layers.Input(shape=IMG_SHAPE) # unet

    # now we concatenate these input a single vector
    # high-resolution data stream
    # first we put "real or fake data" with 32 channels, to allow it to be more important
    inputs_high_res = res_block_initial(img_input, [high_resolution_feature_channels[0]], 3, [1, 1], "output_convbbb")
    x = conv_block(inputs_high_res, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(2, 2),
                   use_bn=use_bn, use_bias=use_bias,
                   use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())

    x = conv_block(x, high_resolution_feature_channels[2], kernel_size=(3, 3), strides=(2, 2),
                   use_bn=use_bn, use_bias=use_bias,
                   use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    # reducing the dimensionality to speed up the computational cost
    # x = tf.keras.layers.AveragePooling2D((3,3))(x)

    x_init_raw = conv_block(x, high_resolution_feature_channels[3], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
#     images_low_res = conv_block(img_input2, high_resolution_feature_channels[0], kernel_size=(3, 3), strides=(1, 1),
#                             use_bn=use_bn, use_bias=use_bias,
#                             use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
#     images_low_res = conv_block(images_low_res, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(1, 1),
#                             use_bn=use_bn, use_bias=use_bias,
#                             use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
#     images_low_res = tf.image.resize(images_low_res, [x_init_raw.shape[1], x_init_raw.shape[2]],
#                     method=tf.image.ResizeMethod.BILINEAR)
    concat_outputs = x_init_raw#tf.keras.layers.Concatenate(-1)([x_init_raw, images_low_res])
    concat_outputs = res_block_initial(concat_outputs, [high_resolution_feature_channels[3]], 3, [1, 1], "output_convbbb1")
    concat_outputs = res_block_initial(concat_outputs, [high_resolution_feature_channels[3]], 3, [1, 1], "output_convbbb2")
    concat_outputs = res_block_initial(concat_outputs, [high_resolution_feature_channels[3]], 3, [1, 1], "output_convbbb3")
    x_init_raw = conv_block(concat_outputs, high_resolution_feature_channels[3], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0,
                            activation=tf.keras.layers.LeakyReLU())
    x_init_raw = conv_block(x_init_raw, high_resolution_feature_channels[3], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0,
                            activation=tf.keras.layers.LeakyReLU())
    flattened_output = tf.keras.layers.Flatten()(x_init_raw)
    dense2 = tf.keras.layers.Dense(32)(flattened_output)

    x = layers.Dense(1)(dense2)

    d_model = keras.models.Model([img_input, img_input2, img_input3, img_input4], x,
                                 name="discriminator")
    return d_model



def get_discriminator_model_v3(high_resolution_fields_size,
                            low_resolution_fields_size, use_bn=False,
                            use_dropout=False, use_bias=True, low_resolution_feature_channels=(32, 64, 128),
                            low_resolution_dense_neurons =6,
                            high_resolution_feature_channels=(16, 32, 64, 128)):
    """
    Discriminator no longer uses the unet model to demonstrate realism
    **Purpose:**
      * To create a discriminator model that takes two streams of inputs, one from the low resolution predictor fields(X)
      and auxilary inputs (topography), it also takes in the high-resolution "regression prediction",
      which is used for residuals

    **Parameters:**
      * **high_resolution_fields_size (tuple):**  The size of the 2D high-resolution RCM fields, over the NZ region this (172, 179)
      * **low_resolution_fields_size (tuple):**  The size of the 2D low-resolution predictor fields (23, 26) over the New Zealand domain
      * **use_bn (bool, optional):** whether to use batchnormalization or not (default no bn)
      * **use_dropout (bool, optional):** whether to use dropout or not(default no dropout)
      * **use_bias (bool, optional):** whether to use bias or not (default bias =True)

    **Returns:**
        * a tf.keras.models.Model class

    **Example Usage:**
    ```python
    discriminator_model = get_discriminator_model((172, 179), (23, 26))
    ```
    """
    IMG_SHAPE = high_resolution_fields_size
    IMG_SHAPE2 = low_resolution_fields_size

    img_input = layers.Input(shape=IMG_SHAPE) # real or fake predictions
    img_input2 = layers.Input(shape=IMG_SHAPE2) # boundary conditions or predictor fields

    # these are static inputs to the model
    img_input3 = layers.Input(shape=IMG_SHAPE) # Topography predictor variable
    img_input4 = layers.Input(shape=IMG_SHAPE) # unet
    
    img_input5 = layers.Input(shape=(1,)) # timeofyear
    img_input6 = layers.Input(shape=(8,)) # means
    img_input7 = layers.Input(shape=(8,)) # stds
    # removed time_of_year as a predictor
    #combined = tf.keras.layers.Concatenate(-1)([img_input6, img_input7])
    #dense_time_of_year = tf.keras.layers.Dense(2*2*2)(combined)
    #reshaped = tf.keras.layers.Reshape((2, 2, 2))(dense_time_of_year)
    

    # now we concatenate these input a single vector
    # high-resolution data stream
    # first we put "real or fake data" with 32 channels, to allow it to be more important
    inputs_high_res = res_block_initial(img_input, [high_resolution_feature_channels[0]], 3, [1, 1], "output_convbbb")
    x = conv_block(inputs_high_res, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(2, 2),
                   use_bn=use_bn, use_bias=use_bias,
                   use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())

    x = conv_block(x, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(2, 2),
                   use_bn=use_bn, use_bias=use_bias,
                   use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    # reducing the dimensionality to speed up the computational cost
    # x = tf.keras.layers.AveragePooling2D((3,3))(x)

    x_init_raw = conv_block(x, high_resolution_feature_channels[2], kernel_size=(7, 7), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    #reshaped = tf.image.resize(reshaped, [img_input2.shape[1], img_input2.shape[2]],
    #                method=tf.image.ResizeMethod.BILINEAR)
#     updated_low_res = tf.keras.layers.Concatenate(-1)([img_input2, reshaped])
#     images_low_res = conv_block(updated_low_res, high_resolution_feature_channels[0], kernel_size=(3, 3), strides=(1, 1),
#                             use_bn=use_bn, use_bias=use_bias,
#                             use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
#     images_low_res = conv_block(images_low_res, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(1, 1),
#                             use_bn=use_bn, use_bias=use_bias,
#                             use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
#     images_low_res = conv_block(images_low_res, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(1, 1),
#                             use_bn=use_bn, use_bias=use_bias,
#                             use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
#     images_low_res = conv_block(images_low_res, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(1, 1),
#                             use_bn=use_bn, use_bias=use_bias,
#                             use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
#     images_low_res = tf.image.resize(images_low_res, [x_init_raw.shape[1], x_init_raw.shape[2]],
#                     method=tf.image.ResizeMethod.BILINEAR)
    concat_outputs = x_init_raw#tf.keras.layers.Concatenate(-1)([x_init_raw, images_low_res])
    concat_outputs = res_block_initial(concat_outputs, [high_resolution_feature_channels[2]], 3, [1, 1], "output_convbbbu")
    concat_outputs = res_block_initial(concat_outputs, [high_resolution_feature_channels[2]], 5, [1, 1], "output_convbbbb")
    concat_outputs = res_block_initial(concat_outputs, [high_resolution_feature_channels[2]], 3, [1, 1], "output_convbbbccc")
    concat_outputs = res_block_initial(concat_outputs, [high_resolution_feature_channels[3]], 3, [1, 1], "output_convbbb1")
    concat_outputs = res_block_initial(concat_outputs, [high_resolution_feature_channels[3]], 3, [1, 1], "output_convbbb2")
    concat_outputs = res_block_initial(concat_outputs, [high_resolution_feature_channels[3]], 3, [1, 1], "output_convbbb3")
    x_init_raw = conv_block(concat_outputs, high_resolution_feature_channels[3], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0,
                            activation=tf.keras.layers.LeakyReLU())
    x_init_raw = conv_block(x_init_raw, high_resolution_feature_channels[3], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0,
                            activation=tf.keras.layers.LeakyReLU())
    x_init_raw = conv_block(x_init_raw, high_resolution_feature_channels[3], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0,
                            activation=tf.keras.layers.LeakyReLU())
    flattened_output = tf.keras.layers.Flatten()(x_init_raw)
    dense2 = tf.keras.layers.Dense(32)(flattened_output)
    x = layers.Dense(1)(dense2)

    d_model = keras.models.Model([img_input, img_input2, img_input3, img_input4, img_input5, img_input6,img_input7], x,
                                 name="discriminator")
    return d_model



def get_discriminator_model_v2(high_resolution_fields_size,
                            low_resolution_fields_size, use_bn=False,
                            use_dropout=False, use_bias=True, low_resolution_feature_channels=(32, 64, 128),
                            low_resolution_dense_neurons =6,
                            high_resolution_feature_channels=(16, 32, 64, 128)):
    """
    Discriminator no longer uses the unet model to demonstrate realism
    **Purpose:**
      * To create a discriminator model that takes two streams of inputs, one from the low resolution predictor fields(X)
      and auxilary inputs (topography), it also takes in the high-resolution "regression prediction",
      which is used for residuals

    **Parameters:**
      * **high_resolution_fields_size (tuple):**  The size of the 2D high-resolution RCM fields, over the NZ region this (172, 179)
      * **low_resolution_fields_size (tuple):**  The size of the 2D low-resolution predictor fields (23, 26) over the New Zealand domain
      * **use_bn (bool, optional):** whether to use batchnormalization or not (default no bn)
      * **use_dropout (bool, optional):** whether to use dropout or not(default no dropout)
      * **use_bias (bool, optional):** whether to use bias or not (default bias =True)

    **Returns:**
        * a tf.keras.models.Model class

    **Example Usage:**
    ```python
    discriminator_model = get_discriminator_model((172, 179), (23, 26))
    ```
    """
    IMG_SHAPE = high_resolution_fields_size
    IMG_SHAPE2 = low_resolution_fields_size

    img_input = layers.Input(shape=IMG_SHAPE) # real or fake predictions
    img_input2 = layers.Input(shape=IMG_SHAPE2) # boundary conditions or predictor fields

    # these are static inputs to the model
    img_input3 = layers.Input(shape=IMG_SHAPE) # Topography predictor variable
    img_input4 = layers.Input(shape=IMG_SHAPE) # unet
    
    img_input5 = layers.Input(shape=(1,)) # timeofyear
    img_input6 = layers.Input(shape=(8,)) # means
    img_input7 = layers.Input(shape=(8,)) # stds
    # removed time_of_year as a predictor
    combined = tf.keras.layers.Concatenate(-1)([img_input6, img_input7])
    dense_time_of_year = tf.keras.layers.Dense(2*2*2)(combined)
    reshaped = tf.keras.layers.Reshape((2, 2, 2))(dense_time_of_year)
    

    # now we concatenate these input a single vector
    # high-resolution data stream
    # first we put "real or fake data" with 32 channels, to allow it to be more important
    inputs_high_res = res_block_initial(img_input, [high_resolution_feature_channels[0]], 3, [1, 1], "output_convbbb")
    x = conv_block(inputs_high_res, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(2, 2),
                   use_bn=use_bn, use_bias=use_bias,
                   use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())

    x = conv_block(x, high_resolution_feature_channels[2], kernel_size=(3, 3), strides=(2, 2),
                   use_bn=use_bn, use_bias=use_bias,
                   use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    # reducing the dimensionality to speed up the computational cost
    # x = tf.keras.layers.AveragePooling2D((3,3))(x)

    x_init_raw = conv_block(x, high_resolution_feature_channels[3], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    reshaped = tf.image.resize(reshaped, [img_input2.shape[1], img_input2.shape[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
    updated_low_res = tf.keras.layers.Concatenate(-1)([img_input2, reshaped])
    images_low_res = conv_block(updated_low_res, high_resolution_feature_channels[0], kernel_size=(3, 3), strides=(1, 1),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    images_low_res = conv_block(images_low_res, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(1, 1),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    images_low_res = conv_block(images_low_res, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(1, 1),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    images_low_res = conv_block(images_low_res, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(1, 1),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    images_low_res = tf.image.resize(images_low_res, [x_init_raw.shape[1], x_init_raw.shape[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
    concat_outputs = tf.keras.layers.Concatenate(-1)([x_init_raw, images_low_res])
    concat_outputs = res_block_initial(concat_outputs, [high_resolution_feature_channels[3]], 3, [1, 1], "output_convbbb1")
    concat_outputs = res_block_initial(concat_outputs, [high_resolution_feature_channels[3]], 3, [1, 1], "output_convbbb2")
    concat_outputs = res_block_initial(concat_outputs, [high_resolution_feature_channels[3]], 3, [1, 1], "output_convbbb3")
    x_init_raw = conv_block(concat_outputs, high_resolution_feature_channels[3], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0,
                            activation=tf.keras.layers.LeakyReLU())
    x_init_raw = conv_block(x_init_raw, high_resolution_feature_channels[3], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0,
                            activation=tf.keras.layers.LeakyReLU())
    flattened_output = tf.keras.layers.Flatten()(x_init_raw)
    dense2 = tf.keras.layers.Dense(32)(flattened_output)
    x = layers.Dense(1)(dense2)

    d_model = keras.models.Model([img_input, img_input2, img_input3, img_input4, img_input5, img_input6,img_input7], x,
                                 name="discriminator")
    return d_model

def res_linear_activation_v2(input_size, resize_output, num_filters, num_channels,
                             final_activations=None):

    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    # we also added an interactive noise layer
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    # precip, tasmin, sfcwind
    high_res_fields2 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields3 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields4 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields5 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields6 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])

    concat_image = tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2, high_res_fields3,  high_res_fields4, high_res_fields5, high_res_fields6])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    time_of_year = tf.keras.layers.Input(shape=[1])
    means = tf.keras.layers.Input(shape=[8])
    stds = tf.keras.layers.Input(shape=[8])
    # Example: mean across axis 1 and 2

    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])
    inputs_abstract = tf.keras.layers.Concatenate(-1)([low_res, noise])
    inputs_abstract = res_block_initial(inputs_abstract, [num_filters[0]], 3, [1, 1], f"noise_blocka", sym_padding=True)
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=3, i =0)
    noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_filters[0]//2)])
    x = tf.keras.layers.Concatenate(-1)([x, noise2])
    x = res_block_initial(x, [num_filters[0]], 3, [1, 1], f"noise_blockb", sym_padding=True)
    # this ensures that there is interaction of variables at multiple scales
    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    
    combined = tf.keras.layers.Concatenate(-1)([time_of_year, means, stds])
    dense_time_of_year = tf.keras.layers.Dense(2*2*2)(combined)
    reshaped = tf.keras.layers.Reshape((2, 2, 2))(dense_time_of_year)
    
    reshaped = tf.image.resize(reshaped, [x1.shape[1], x1.shape[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x, reshaped])
    x = res_block_initial(x, [num_filters[2]], 3, [1, 1], f"noise_blockc", sym_padding=True)
    x = res_block_initial(x, [num_filters[2]], 3, [1, 1], f"noise_blockd", sym_padding=True)
    

    tas_block = res_block_initial(x, [128], 3, [1, 1], f"tas_block", sym_padding=True)
    tas_block = res_block_initial(tas_block, [128], 3, [1, 1], f"tas_block2", sym_padding=True)
    wind_block = res_block_initial(x, [128], 3, [1, 1], f"wind_block", sym_padding=True)
    wind_block = res_block_initial(wind_block, [128], 3, [1, 1], f"wind_block2", sym_padding=True)



    tasmin = create_output_layer(tas_block, temp3, temp2, temp1, num_filters, resize_output,final_activations[1], name="tasmin")
    tasmax = create_output_layer(tas_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.8), name="tasmax")
    sfcwind = create_output_layer(wind_block, temp3, temp2, temp1, num_filters, resize_output,final_activations[1], name="sfcwind")
    sfcwindmax = create_output_layer(wind_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.8), name="sfcwind_max")
    
    
    
    
        
    concat_image = tf.keras.layers.Concatenate(-1)([high_res_fields, tasmin, tasmax, sfcwind, sfcwindmax, high_res_fields2, high_res_fields3,  high_res_fields4, high_res_fields5, high_res_fields6])
    concatted_highres2 = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                   method=tf.image.ResizeMethod.BILINEAR)
    
    inputs_abstract = tf.keras.layers.Concatenate(-1)([low_res, noise])
    x, temp1 = down_block(concatted_highres2, num_filters[0], kernel_size=5, i =10)
    x, temp2 = down_block(x, num_filters[1], kernel_size=5, i =11)
    x, temp3 = down_block(x, num_filters[2], kernel_size=5, i =22)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=44, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(np.ceil(resize_output[0]/8)), int(np.ceil(resize_output[1]/8))],
                    method=tf.image.ResizeMethod.BILINEAR)
    combined = tf.keras.layers.Concatenate(-1)([time_of_year, means, stds])
    dense_time_of_year = tf.keras.layers.Dense(2*2*2)(combined)
    reshaped = tf.keras.layers.Reshape((2, 2, 2))(dense_time_of_year)
    
    reshaped = tf.image.resize(reshaped, [x1.shape[1], x1.shape[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x, reshaped])
    
    
    pr_block = res_block_initial(x, [128], 3, [1, 1], f"pr_block", sym_padding=True)
    pr_block = res_block_initial(pr_block, [128], 3, [1, 1], f"pr_block2", sym_padding=True)
    precip_output = create_output_layer(pr_block, temp3,temp2, temp1, num_filters, resize_output,final_activations[0], name="precip_layer", concat=True)
    
    outputs = [precip_output, tasmin, tasmax, sfcwind, sfcwindmax]
    # decode

    input_layers = [noise, noise2] + [low_res, high_res_fields, high_res_fields2,high_res_fields3, high_res_fields4, high_res_fields5, high_res_fields6, time_of_year, means, stds]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')

    return model



def res_linear_activation_v2updated(input_size, resize_output, num_filters, num_channels,
                             final_activations=None):

    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    # we also added an interactive noise layer
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    # precip, tasmin, sfcwind
    high_res_fields2 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields3 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields4 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields5 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields6 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])

    concat_image = tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2, high_res_fields3,  high_res_fields4, high_res_fields5, high_res_fields6])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    time_of_year = tf.keras.layers.Input(shape=[1])
    means = tf.keras.layers.Input(shape=[8])
    stds = tf.keras.layers.Input(shape=[8])
    # Example: mean across axis 1 and 2

    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], int(num_channels//2)])
    inputs_abstract = tf.keras.layers.Concatenate(-1)([low_res, noise])
    inputs_abstract = res_block_initial(inputs_abstract, [num_filters[0]], 3, [1, 1], f"noise_blocka", sym_padding=False)
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=3, i =0, sym_padding =False)
    #noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_filters[0]//2)])
    x = x#tf.keras.layers.Concatenate(-1)([x, noise2])
    x = res_block_initial(x, [num_filters[0]], 3, [1, 1], f"noise_blockb", sym_padding=False)
    # this ensures that there is interaction of variables at multiple scales
    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1, sym_padding =False)
    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2, sym_padding =False)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False, sym_padding =False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    
    combined = tf.keras.layers.Concatenate(-1)([time_of_year, means, stds])
    dense_time_of_year = tf.keras.layers.Dense(2*2*2)(combined)
    reshaped = tf.keras.layers.Reshape((2, 2, 2))(dense_time_of_year)
    
    reshaped = tf.image.resize(reshaped, [x1.shape[1], x1.shape[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x, reshaped])
    x = res_block_initial(x, [num_filters[2]], 3, [1, 1], f"noise_blockc", sym_padding=False)
    xa = res_block_initial(x, [num_filters[2]], 3, [1, 1], f"noise_blockd", sym_padding=False)
    

    tas_block = res_block_initial(xa, [128], 3, [1, 1], f"tas_block", sym_padding=False)
    tas_block = res_block_initial(tas_block, [128], 3, [1, 1], f"tas_block2", sym_padding=False)
    concat_block = tf.keras.layers.Concatenate(-1)([tas_block, xa])
    wind_block = res_block_initial(concat_block, [128], 3, [1, 1], f"wind_block", sym_padding=False)
    wind_block = res_block_initial(wind_block, [128], 5, [1, 1], f"wind_block2", sym_padding=False)



    tasmin, noise2 = create_output_layer(tas_block, temp3, temp2, temp1, num_filters, resize_output,final_activations[1], name="tasmin", noise="fake")
    tasmax = create_output_layer(tas_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.8), name="tasmax", noise = False)
    sfcwind = create_output_layer(wind_block, temp3, temp2, temp1, num_filters, resize_output,final_activations[1], name="sfcwind", noise = False)
    sfcwindmax = create_output_layer(wind_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.8), name="sfcwind_max", noise =False)
 
    concat_block2 = tf.keras.layers.Concatenate(-1)([wind_block, tas_block, xa])
    
    pr_block = res_block_initial(concat_block2, [128], 3, [1, 1], f"pr_block", sym_padding=False)
    pr_block = res_block_initial(pr_block, [128], 3, [1, 1], f"pr_block2", sym_padding=False)
    pr_block = res_block_initial(pr_block, [256], 3, [1, 1], f"pr_block23", sym_padding=False)
    precip_output = create_output_layer(pr_block, temp3,temp2, temp1, num_filters, resize_output,final_activations[0], name="precip_layer", concat=True, noise =False)
    
    outputs = [precip_output, tasmin, tasmax, sfcwind, sfcwindmax]

    input_layers = [noise, noise2] + [low_res, high_res_fields, high_res_fields2,high_res_fields3, high_res_fields4, high_res_fields5, high_res_fields6, time_of_year, means, stds]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')

    return model


def res_linear_activation_single(input_size, resize_output, num_filters, num_channels,
                             final_activations=None):

    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    # we also added an interactive noise layer
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    # precip, tasmin, sfcwind
    high_res_fields2 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])


    concat_image = tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    time_of_year = tf.keras.layers.Input(shape=[1])
    means = tf.keras.layers.Input(shape=[8])
    stds = tf.keras.layers.Input(shape=[8])
    # Example: mean across axis 1 and 2

    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])
    inputs_abstract = tf.keras.layers.Concatenate(-1)([low_res, noise])
    inputs_abstract = res_block_initial(inputs_abstract, [num_filters[0]], 3, [1, 1], f"noise_blocka", sym_padding=True)
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=3, i =0)
    #noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_filters[0]//2)])
    x = x#tf.keras.layers.Concatenate(-1)([x, noise2])
    x = res_block_initial(x, [num_filters[0]], 3, [1, 1], f"noise_blockb", sym_padding=True)
    # this ensures that there is interaction of variables at multiple scales
    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    
    combined = tf.keras.layers.Concatenate(-1)([time_of_year, means, stds])
    dense_time_of_year = tf.keras.layers.Dense(2*2*2)(combined)
    reshaped = tf.keras.layers.Reshape((2, 2, 2))(dense_time_of_year)
    
    reshaped = tf.image.resize(reshaped, [x1.shape[1], x1.shape[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x, reshaped])
    x = res_block_initial(x, [num_filters[2]], 5, [1, 1], f"noise_blockc", sym_padding=True)
    x = res_block_initial(x, [num_filters[2]], 5, [1, 1], f"noise_blockd", sym_padding=True)
    wind_block = res_block_initial(x, [128], 5, [1, 1], f"wind_block", sym_padding=True)
    wind_block = res_block_initial(wind_block, [128], 5, [1, 1], f"wind_block2", sym_padding=True)
    sfcwindmax, noise2 = create_output_layer(wind_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(1), name="sfcwind_max", noise =True)
    
    outputs = [sfcwindmax]
    # decode

    input_layers = [noise, noise2] + [low_res, high_res_fields, high_res_fields2, time_of_year, means, stds]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')

    return model



def res_linear_activation_single2(input_size, resize_output, num_filters, num_channels,
                             final_activations=None):

    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    # we also added an interactive noise layer
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    # precip, tasmin, sfcwind
    high_res_fields2 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])


    concat_image = tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    time_of_year = tf.keras.layers.Input(shape=[1])
    means = tf.keras.layers.Input(shape=[8])
    stds = tf.keras.layers.Input(shape=[8])
    # Example: mean across axis 1 and 2

    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], int(num_channels)])
    inputs_abstract = tf.keras.layers.Concatenate(-1)([low_res, noise])
    inputs_abstract = res_block_initial(inputs_abstract, [num_filters[0]], 3, [1, 1], f"noise_blocka", sym_padding=False)
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=3, i =0)
    #noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_filters[0]//2)])
    x = x#tf.keras.layers.Concatenate(-1)([x, noise2])
    x = res_block_initial(x, [num_filters[0]], 3, [1, 1], f"noise_blockb", sym_padding=False)
    # this ensures that there is interaction of variables at multiple scales
    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    
    combined = tf.keras.layers.Concatenate(-1)([time_of_year, means, stds])
    dense_time_of_year = tf.keras.layers.Dense(2*2*2)(combined)
    reshaped = tf.keras.layers.Reshape((2, 2, 2))(dense_time_of_year)
    
    reshaped = tf.image.resize(reshaped, [x1.shape[1], x1.shape[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x, reshaped])
    x = res_block_initial(x, [num_filters[2]], 3, [1, 1], f"noise_blockc", sym_padding=False)
    x = res_block_initial(x, [num_filters[2]], 3, [1, 1], f"noise_blockd", sym_padding=False)
    wind_block = res_block_initial(x, [128], 3, [1, 1], f"wind_block", sym_padding=False)
    wind_block = res_block_initial(wind_block, [128], 3, [1, 1], f"wind_block2",sym_padding=False)
    sfcwindmax, noise2 = create_output_layer(wind_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.8), name="sfcwind_max", noise ="fake")
    
    outputs = [sfcwindmax]
    # decode

    input_layers = [noise, noise2] + [low_res, high_res_fields, high_res_fields2, time_of_year, means, stds]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')

    return model


def res_linear_activation_single3(input_size, resize_output, num_filters, num_channels,
                             final_activations=None, noise2 = "fake"):

    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    # we also added an interactive noise layer
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    # precip, tasmin, sfcwind
    high_res_fields2 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])


    concat_image = high_res_fields#tf.keras.layers.Concatenate(-1)([high_res_fields])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    time_of_year = tf.keras.layers.Input(shape=[1])
    means = tf.keras.layers.Input(shape=[8])
    stds = tf.keras.layers.Input(shape=[8])
    # Example: mean across axis 1 and 2

    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])
    means_rs = tf.keras.layers.Reshape([1,1, 8])(means)
    std_rs = tf.keras.layers.Reshape([1,1, 8])(means)
    as_inputs = low_res * std_rs + means_rs
    inputs_abstract = tf.keras.layers.Concatenate(-1)([as_inputs, noise])
    inputs_abstract = res_block_initial(inputs_abstract, [num_filters[0]], 3, [1, 1], f"noise_blocka", sym_padding=True)
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=3, i =0)
    #noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_filters[0]//2)])
    x = x#tf.keras.layers.Concatenate(-1)([x, noise2])
    x = res_block_initial(x, [num_filters[0]], 3, [1, 1], f"noise_blockb", sym_padding=True)
    # this ensures that there is interaction of variables at multiple scales
    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    
#     combined = tf.keras.layers.Concatenate(-1)([time_of_year, means, stds])
#     dense_time_of_year = tf.keras.layers.Dense(2*2*2)(combined)
#     reshaped = tf.keras.layers.Reshape((2, 2, 2))(dense_time_of_year)
    
#     reshaped = tf.image.resize(reshaped, [x1.shape[1], x1.shape[2]],
#                     method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x])
    x = res_block_initial(x, [num_filters[2]], 5, [1, 1], f"noise_blockc", sym_padding=True)
    x = res_block_initial(x, [num_filters[2]], 5, [1, 1], f"noise_blockd", sym_padding=True)
    wind_block = res_block_initial(x, [128], 5, [1, 1], f"wind_block", sym_padding=True)
    wind_block = res_block_initial(wind_block, [128], 5, [1, 1], f"wind_block2", sym_padding=True)
    sfcwindmax, noise2 = create_output_layer(wind_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.5), name="sfcwind_max", noise =noise2)
    
    outputs = [sfcwindmax]
    # decode

    input_layers = [noise, noise2] + [low_res, high_res_fields, high_res_fields2, time_of_year, means, stds]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')

    return model


def rainfall_gan_v2(input_size, resize_output, num_filters, num_channels,
                             final_activations=None):

    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    # we also added an interactive noise layer
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    # precip, tasmin, sfcwind
    high_res_fields2 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields3 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields4 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields5 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields6 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    
    
#     high_res_fields7 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
#     high_res_fields8 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
#     high_res_fields9 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
#     high_res_fields10 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
#     high_res_fields11 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])

    concat_image = tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2, high_res_fields3,  high_res_fields4, high_res_fields5, high_res_fields6])
    # While we are testing the approach these are not concatenated into te model 
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    time_of_year = tf.keras.layers.Input(shape=[1])
    means = tf.keras.layers.Input(shape=[8])
    stds = tf.keras.layers.Input(shape=[8])
    # Example: mean across axis 1 and 2

    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])
    inputs_abstract = tf.keras.layers.Concatenate(-1)([low_res, noise])
    inputs_abstract = res_block_initial(inputs_abstract, [num_filters[0]], 3, [1, 1], f"noise_blocka", sym_padding=True)
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=3, i =0)
    noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_filters[0]//2)])
    x = tf.keras.layers.Concatenate(-1)([x, noise2])
    x = res_block_initial(x, [num_filters[0]], 3, [1, 1], f"noise_blockbd", sym_padding=True)
    x = res_block_initial(x, [num_filters[1]], 3, [1, 1], f"noise_blockef", sym_padding=True)
    # this ensures that there is interaction of variables at multiple scales
    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    
    combined = tf.keras.layers.Concatenate(-1)([time_of_year, means, stds])
    dense_time_of_year = tf.keras.layers.Dense(2*2*2)(combined)
    reshaped = tf.keras.layers.Reshape((2, 2, 2))(dense_time_of_year)
    
    reshaped = tf.image.resize(reshaped, [x1.shape[1], x1.shape[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x, reshaped])
    x = res_block_initial(x, [num_filters[2]], 3, [1, 1], f"noise_blockc", sym_padding=True)
    x = res_block_initial(x, [num_filters[2]], 3, [1, 1], f"noise_blockd", sym_padding=True)
    

    pr_block = res_block_initial(x, [256], 3, [1, 1], f"pr_block", sym_padding=True)
    pr_block = res_block_initial(pr_block, [256], 3, [1, 1], f"pr_block2", sym_padding=True)


    precip = create_output_layer(pr_block, temp3, temp2, temp1, num_filters, resize_output,final_activations[1], name="precip")
   
    outputs = [precip]
    # decode

    input_layers = [noise, noise2] + [low_res, high_res_fields, high_res_fields2,high_res_fields3, high_res_fields4, high_res_fields5, high_res_fields6, time_of_year, means, stds]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')

    return model


def rainfall_gan_v3(input_size, resize_output, num_filters, num_channels,
                             final_activations=None):

    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    # we also added an interactive noise layer
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    # precip, tasmin, sfcwind
    high_res_fields2 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    
    
#     high_res_fields7 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
#     high_res_fields8 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
#     high_res_fields9 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
#     high_res_fields10 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
#     high_res_fields11 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])

    concat_image = tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2])
    # While we are testing the approach these are not concatenated into te model 
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    time_of_year = tf.keras.layers.Input(shape=[1])
    means = tf.keras.layers.Input(shape=[8])
    stds = tf.keras.layers.Input(shape=[8])
    # Example: mean across axis 1 and 2

    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])
    inputs_abstract = tf.keras.layers.Concatenate(-1)([low_res, noise])
    inputs_abstract = res_block_initial(inputs_abstract, [num_filters[0]], 3, [1, 1], f"noise_blocka", sym_padding=True)
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=3, i =0)
    noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_filters[0]//2)])
    x = tf.keras.layers.Concatenate(-1)([x, noise2])
    x = res_block_initial(x, [num_filters[0]], 3, [1, 1], f"noise_blockbd", sym_padding=True)
    x = res_block_initial(x, [num_filters[1]], 3, [1, 1], f"noise_blockef", sym_padding=True)
    # this ensures that there is interaction of variables at multiple scales
    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    
    #combined = tf.keras.layers.Concatenate(-1)([time_of_year, means, stds])
    #dense_time_of_year = tf.keras.layers.Dense(2*2*2)(combined)
    #reshaped = tf.keras.layers.Reshape((2, 2, 2))(dense_time_of_year)
    
    #reshaped = tf.image.resize(reshaped, [x1.shape[1], x1.shape[2]],
    #                method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x])#, reshaped])
    x = res_block_initial(x, [num_filters[2]], 3, [1, 1], f"noise_blockc", sym_padding=True)
    x = res_block_initial(x, [num_filters[2]], 3, [1, 1], f"noise_blockd", sym_padding=True)
    

    pr_block = res_block_initial(x, [256], 3, [1, 1], f"pr_block", sym_padding=True)
    pr_block = res_block_initial(pr_block, [256], 3, [1, 1], f"pr_block2", sym_padding=True)


    precip = create_output_layer(pr_block, temp3, temp2, temp1, num_filters, resize_output,final_activations[1], name="precip")
   
    outputs = [precip]
    # decode

    input_layers = [noise, noise2] + [low_res, high_res_fields, high_res_fields2, time_of_year, means, stds]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')

    return model

def res_linear_activation_v5(input_size, resize_output, num_filters, num_channels,
                             final_activations=None):

    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    # we also added an interactive noise layer
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    # precip, tasmin, sfcwind
    high_res_fields2 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields3 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields4 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields5 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields6 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])

    concat_image = tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2, high_res_fields3,  high_res_fields4, high_res_fields5, high_res_fields6])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    time_of_year = tf.keras.layers.Input(shape=[1])
    means = tf.keras.layers.Input(shape=[8])
    stds = tf.keras.layers.Input(shape=[8])
    # Example: mean across axis 1 and 2

    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])
    inputs_abstract = tf.keras.layers.Concatenate(-1)([low_res, noise])
    inputs_abstract = res_block_initial(inputs_abstract, [num_filters[0]], 3, [1, 1], f"noise_blocka", sym_padding=True)
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=3, i =0)
    noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_filters[0]//2)])
    x = tf.keras.layers.Concatenate(-1)([x, noise2])
    x = res_block_initial(x, [num_filters[0]], 3, [1, 1], f"noise_blockb", sym_padding=True)
    # this ensures that there is interaction of variables at multiple scales
    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    
    combined = tf.keras.layers.Concatenate(-1)([time_of_year, means, stds])
    dense_time_of_year = tf.keras.layers.Dense(2*2*2)(combined)
    reshaped = tf.keras.layers.Reshape((2, 2, 2))(dense_time_of_year)
    
    reshaped = tf.image.resize(reshaped, [x1.shape[1], x1.shape[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x, reshaped])
    x = res_block_initial(x, [num_filters[2]], 3, [1, 1], f"noise_blockc", sym_padding=True)
    x = res_block_initial(x, [num_filters[2]], 3, [1, 1], f"noise_blockd", sym_padding=True)
    

    tas_block = res_block_initial(x, [128], 3, [1, 1], f"tas_block", sym_padding=True)
    tas_block = res_block_initial(tas_block, [128], 3, [1, 1], f"tas_block2", sym_padding=True)
    wind_block = res_block_initial(x, [128], 3, [1, 1], f"wind_block", sym_padding=True)
    wind_block = res_block_initial(wind_block, [128], 3, [1, 1], f"wind_block2", sym_padding=True)



    tasmin = create_output_layer(tas_block, temp3, temp2, temp1, num_filters, resize_output,final_activations[1], name="tasmin")
    tasmax = create_output_layer(tas_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.8), name="tasmax")
    sfcwind = create_output_layer(wind_block, temp3, temp2, temp1, num_filters, resize_output,final_activations[1], name="sfcwind")
    sfcwindmax = create_output_layer(wind_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.8), name="sfcwind_max")
    
   
    outputs = [tasmin, tasmax, sfcwind, sfcwindmax]
    # decode

    input_layers = [noise, noise2] + [low_res, high_res_fields, high_res_fields2,high_res_fields3, high_res_fields4, high_res_fields5, high_res_fields6, time_of_year, means, stds]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')

    return model
def res_linear_activation_v4(input_size, resize_output, num_filters, num_channels,
                             final_activations=None):

    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    # we also added an interactive noise layer
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    # precip, tasmin, sfcwind
    high_res_fields2 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields3 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields4 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])

    concat_image = tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2, high_res_fields3,  high_res_fields4])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])

    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])
    inputs_abstract = low_res#tf.keras.layers.Concatenate(-1)([low_res, noise])
    inputs_abstract = res_block_initial(inputs_abstract, [num_filters[0]], 3, [1, 1], f"noise_blocka", sym_padding=True)
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=3, i =0)
    noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_filters[0]//4)])
    x = tf.keras.layers.Concatenate(-1)([x, noise2])
    x = res_block_initial(x, [num_filters[0]], 3, [1, 1], f"noise_blockb", sym_padding=True)
    # this ensures that there is interaction of variables at multiple scales
    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)

    x = tf.keras.layers.Concatenate(-1)([x1, x])
    x = res_block_initial(x, [num_filters[2]], 3, [1, 1], f"noise_blockc", sym_padding=True)
    x = res_block_initial(x, [num_filters[2]], 3, [1, 1], f"noise_blockd", sym_padding=True)
    
    pr_block = res_block_initial(x, [128], 3, [1, 1], f"pr_block", sym_padding=True)
    pr_block = res_block_initial(pr_block, [128], 3, [1, 1], f"pr_block2", sym_padding=True)
    tas_block = res_block_initial(x, [128], 3, [1, 1], f"tas_block", sym_padding=True)
    tas_block = res_block_initial(tas_block, [128], 3, [1, 1], f"tas_block2", sym_padding=True)
    wind_block = res_block_initial(x, [128], 3, [1, 1], f"wind_block", sym_padding=True)
    wind_block = res_block_initial(wind_block, [128], 3, [1, 1], f"wind_block2", sym_padding=True)


    precip_output = create_output_layer(pr_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.001), name="precip_layer")
    tasmin = create_output_layer(tas_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.001), name="tasmin")
    tasmax = create_output_layer(tas_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.001), name="tasmax")
    sfcwind = create_output_layer(wind_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.001), name="sfcwind")
    sfcwindmax = create_output_layer(wind_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.001), name="sfcwind_max")
    sfcwindmax = sfcwind + sfcwindmax
    tasmax = tasmin + tasmax
    outputs = [precip_output, sfcwind, sfcwindmax, tasmax, tasmin]
    # decode
    input_layers = [noise, noise2] + [low_res, high_res_fields, high_res_fields2,high_res_fields3, high_res_fields4]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')

    return model

def res_linear_activation_v3(input_size, resize_output, num_filters, num_channels,
                             final_activations=None):

    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    # we also added an interactive noise layer
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    # precip, tasmin, sfcwind
    high_res_fields2 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields3 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields4 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])

    concat_image = tf.keras.layers.Concatenate(-1)([high_res_fields2, high_res_fields3,  high_res_fields4])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    # Example: mean across axis 1 and 2

    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])
    inputs_abstract = tf.keras.layers.Concatenate(-1)([low_res, noise])
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=3, i =0)
    noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_filters[0]//8)])
    x = tf.keras.layers.Concatenate(-1)([x, noise2])
    # this ensures that there is interaction of variables at multiple scales
    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x])
    tas_block = res_block_initial(x, [128], 3, [1, 1], f"tas_block", sym_padding=True)
    #tas_block = res_block_initial(tas_block, [256], 3, [1, 1], f"tas_block2", sym_padding=True)
    wind_block = res_block_initial(x, [128], 3, [1, 1], f"wind_block", sym_padding=True)
    #wind_block = res_block_initial(wind_block, [256], 3, [1, 1], f"wind_block2", sym_padding=True)


    #precip_output = create_output_layer(pr_block, temp3, temp2, temp1, num_filters, resize_output,final_activations[0], name="precip_layer")
    tasmin = create_output_layer(tas_block, temp3, temp2, temp1, num_filters, resize_output,final_activations[1], name="tasmin")
    tasmax = create_output_layer(tas_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.0001), name="tasmax")
    sfcwind = create_output_layer(wind_block, temp3, temp2, temp1, num_filters, resize_output,final_activations[1], name="sfcwind")
    sfcwindmax = create_output_layer(wind_block, temp3, temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.0001), name="sfcwind_max")
    sfcwindmax = sfcwind * ( 1+ sfcwindmax) 
    tasmax = tasmin + tasmax
    outputs = [tasmin, tasmax, sfcwind, sfcwindmax]
    # decode

    input_layers = [noise, noise2] + [low_res, high_res_fields, high_res_fields2,high_res_fields3, high_res_fields4]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')

    return model





def raingan(input_size, resize_output, num_filters, num_channels,
                             final_activations=None):

    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    # we also added an interactive noise layer
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    # precip_unet, tasmin, sfcwind, sfcwindmax, tasmax
    high_res_fields2 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields3 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields4 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields5 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields6 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])

    concat_image = tf.keras.layers.Concatenate(-1)([high_res_fields])#, high_res_fields2, high_res_fields3,  high_res_fields4, high_res_fields5, high_res_fields6])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)
    concatted_highres= res_block_initial(concatted_highres, [8], 3, [1, 1], f"rain_block3", sym_padding=True)
    concatted_highres= res_block_initial(concatted_highres, [16], 3, [1, 1], f"rain_block5", sym_padding=True)
    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])

    # Example: mean across axis 1 and 2

    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])
    inputs_abstract = tf.keras.layers.Concatenate(-1)([low_res, noise])
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=3, i =0)
    noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_filters[0]//8)])
    x = tf.keras.layers.Concatenate(-1)([x, noise2])
    # this ensures that there is interaction of variables at multiple scales
    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x])
    rain = res_block_initial(x, [128], 3, [1, 1], f"rain_block", sym_padding=True)

    precip_output = create_output_layer( rain, temp3, temp2, temp1, num_filters, resize_output,final_activations[0], name="precip_layer")

    outputs = [precip_output]
    # decode

    input_layers = [noise, noise2] + [low_res, high_res_fields, high_res_fields2,high_res_fields3, high_res_fields4, high_res_fields5, high_res_fields6]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')

    return model


# def create_output_layer(x, temp3, temp2, temp1, num_filters, resize_output, final_activation, name ="precip_layer", concat = True, noise=True):
#     # decode
#     x = up_block(x, temp3, kernel_size=3, filters = num_filters[2], i =str(name) + "0", concat = concat)
#     x = up_block(x, temp2, kernel_size=3, filters = num_filters[1], i =str(name) + "1", concat = concat)
#     if noise:
#         noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_filters[1]//2)])
#         x = tf.keras.layers.Concatenate(-1)([x, noise2])
#     x = up_block(x, temp1, kernel_size=3, filters = num_filters[0], i =str(name) + "2", concat = concat)
#     output = tf.image.resize(x, (resize_output[0], resize_output[1]),
#                     method=tf.image.ResizeMethod.BILINEAR)
#     output = res_block_initial(output, [128], 3, [1, 1], f"output_{name}_0", sym_padding=True)
#     output = res_block_initial(output, [64], 3, [1, 1], f"output_{name}_2", sym_padding=True)
#     output = res_block_initial(output, [32], 3, [1, 1], f"output_{name}_3", sym_padding=True)
#     output = SymmetricPadding2D(padding=[1, 1])(output)
#     output = tf.keras.layers.Conv2D(32, 3, activation=final_activation, padding ='valid')(output)
#     output = SymmetricPadding2D(padding=[1, 1])(output)
#     output = tf.keras.layers.Conv2D(16, 3, activation=final_activation, padding ='valid')(output)
#     output = tf.keras.layers.Conv2D(1, 1, activation=final_activation, padding ='valid')(output)
#     #removed output resizing#
#     if noise:
#         return output, noise2
#     else:
#         return output
    
def create_output_layer(x, temp3, temp2, temp1, num_filters, resize_output, final_activation, name ="precip_layer", concat = True, noise=False, noise_input=None):
    # decode
    x = up_block(x, temp3, kernel_size=3, filters = num_filters[2], i =str(name) + "0", concat = concat, sym_padding =False)
    if noise:
        noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], 1])
        x = tf.keras.layers.Concatenate(-1)([x, noise2])
    elif noise =="false":
        noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_filters[1]//2)])
        x = x#tf.keras.layers.Concatenate(-1)([x, noise2])
    x = up_block(x, temp2, kernel_size=3, filters = num_filters[1], i =str(name) + "1", concat = concat, sym_padding =False)
    if noise_input is not None:
        x = tf.keras.layers.Concatenate(-1)([x, noise_input])
    x = up_block(x, temp1, kernel_size=3, filters = num_filters[0], i =str(name) + "2", concat = concat, sym_padding =False)
    output = tf.image.resize(x, (resize_output[0], resize_output[1]),
                    method=tf.image.ResizeMethod.BILINEAR)
    output = res_block_initial(output, [128], 3, [1, 1], f"output_{name}_0", sym_padding=False)
    output = res_block_initial(output, [64], 3, [1, 1], f"output_{name}_2", sym_padding=False)
    output = res_block_initial(output, [32], 3, [1, 1], f"output_{name}_3", sym_padding=False)
    output = SymmetricPadding2D(padding=[1, 1])(output)
    output = tf.keras.layers.Conv2D(32, 3, activation=final_activation, padding ='valid')(output)
    output = SymmetricPadding2D(padding=[1, 1])(output)
    output = tf.keras.layers.Conv2D(16, 3, activation='linear', padding ='valid')(output)
    output = tf.keras.layers.Conv2D(1, 1, activation='linear', padding ='valid')(output)
    #removed output resizing#
    if noise:
        return output, noise2
    elif noise == "fake":
        return output, noise2
    else:
        return output


def unet_linear_single(input_size, resize_output, num_filters, num_channels,
                          final_activations = None):

    """have modified the architecture with a new concatenation layer"""
    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(0.6), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])


    concat_image = high_res_fields#tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2,high_res_fields3])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                   method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    
    time_of_year = tf.keras.layers.Input(shape=[1])
    means = tf.keras.layers.Input(shape=[8])
    stds = tf.keras.layers.Input(shape=[8])
    # Example: mean across axis 1 and 2
    inputs_abstract = low_res#tf.keras.layers.Concatenate(-1)([low_res, noise])
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=5, i =0)
    x, temp2 = down_block(x, num_filters[1], kernel_size=5, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=5, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(np.ceil(resize_output[0]/8)), int(np.ceil(resize_output[1]/8))],
                    method=tf.image.ResizeMethod.BILINEAR)
    combined = tf.keras.layers.Concatenate(-1)([time_of_year, means, stds])
    dense_time_of_year = tf.keras.layers.Dense(2*2*2)(combined)
    reshaped = tf.keras.layers.Reshape((2, 2, 2))(dense_time_of_year)
    
    reshaped = tf.image.resize(reshaped, [x1.shape[1], x1.shape[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x, reshaped])

    wind_block = res_block_initial(x, [128], 3, [1, 1], f"wind_block", sym_padding=True)
    wind_block = res_block_initial(wind_block, [128], 3, [1, 1], f"wind_block2", sym_padding=True)
    # decode
    sfcwindmax = create_output_layer(wind_block, temp3,temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.001),name="sfcwindmax", concat=True)
    
    outputs = [sfcwindmax]


    input_layers =[low_res, high_res_fields, time_of_year, means, stds]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')
    return model

def unet_linear_v2(input_size, resize_output, num_filters, num_channels,
                          final_activations = None):

    """have modified the architecture with a new concatenation layer"""
    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(0.6), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])


    concat_image = high_res_fields#tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2,high_res_fields3])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                   method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    
    time_of_year = tf.keras.layers.Input(shape=[1])
    means = tf.keras.layers.Input(shape=[8])
    stds = tf.keras.layers.Input(shape=[8])
    # Example: mean across axis 1 and 2
    inputs_abstract = low_res#tf.keras.layers.Concatenate(-1)([low_res, noise])
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=5, i =0)
    x, temp2 = down_block(x, num_filters[1], kernel_size=5, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=5, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(np.ceil(resize_output[0]/8)), int(np.ceil(resize_output[1]/8))],
                    method=tf.image.ResizeMethod.BILINEAR)
    combined = tf.keras.layers.Concatenate(-1)([time_of_year, means, stds])
    dense_time_of_year = tf.keras.layers.Dense(2*2*2)(combined)
    reshaped = tf.keras.layers.Reshape((2, 2, 2))(dense_time_of_year)
    
    reshaped = tf.image.resize(reshaped, [x1.shape[1], x1.shape[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
    xa = tf.keras.layers.Concatenate(-1)([x1, x, reshaped])

    tas_block = res_block_initial(xa, [128], 3, [1, 1], f"tas_block", sym_padding=True)
    tas_block = res_block_initial(tas_block, [128], 3, [1, 1], f"tas_block2", sym_padding=True)
    concat_block = tf.keras.layers.Concatenate(-1)([tas_block, xa])
    wind_block = res_block_initial(concat_block, [128], 3, [1, 1], f"wind_block", sym_padding=True)
    wind_block = res_block_initial(wind_block, [128], 3, [1, 1], f"wind_block2", sym_padding=True)
    # decode

    tasmin = create_output_layer(tas_block, temp3,temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.001), name="tasmin", concat=True)
    tasmax = create_output_layer(tas_block, temp3,temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.001), name="tasmax", concat=True)
    sfcwind = create_output_layer(wind_block, temp3,temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.001), name="sfcwind", concat=True)
    sfcwindmax = create_output_layer(wind_block, temp3,temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.001),name="sfcwindmax", concat=True)
    concat_block2 = tf.keras.layers.Concatenate(-1)([tas_block, wind_block, xa])
    
    pr_block = res_block_initial(concat_block2, [256], 3, [1, 1], f"pr_block", sym_padding=True)
    pr_block = res_block_initial(pr_block, [128], 3, [1, 1], f"pr_block2", sym_padding=True)
    precip_output = create_output_layer(pr_block, temp3,temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.001), name="precip_layer", concat=True)
    
    
    outputs = [precip_output, tasmin, tasmax, sfcwind, sfcwindmax]


    input_layers =[low_res, high_res_fields, time_of_year, means, stds]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')
    return model



def unet_linear_v4(input_size, resize_output, num_filters, num_channels,
                          final_activations = None):

    """have modified the architecture with a new concatenation layer"""
    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(0.6), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])


    concat_image = high_res_fields#tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2,high_res_fields3])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                   method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])

    inputs_abstract = low_res#tf.keras.layers.Concatenate(-1)([low_res, noise])
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=5, i =0)
    x, temp2 = down_block(x, num_filters[1], kernel_size=5, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=5, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(np.ceil(resize_output[0]/8)), int(np.ceil(resize_output[1]/8))],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x])
    pr_block = res_block_initial(x, [128], 3, [1, 1], f"pr_block", sym_padding=True)
    pr_block = res_block_initial(pr_block, [128], 3, [1, 1], f"pr_block2", sym_padding=True)
    tas_block = res_block_initial(x, [128], 3, [1, 1], f"tas_block", sym_padding=True)
    tas_block = res_block_initial(tas_block, [128], 3, [1, 1], f"tas_block2", sym_padding=True)
    wind_block = res_block_initial(x, [128], 3, [1, 1], f"wind_block", sym_padding=True)
    wind_block = res_block_initial(wind_block, [128], 3, [1, 1], f"wind_block2", sym_padding=True)
    # decode
    precip_output = create_output_layer(pr_block, temp3,temp2, temp1, num_filters, resize_output,final_activations[0], name="precip_layer", concat=True)
    tasmin = create_output_layer(tas_block, temp3,temp2, temp1, num_filters, resize_output,final_activations[1], name="tasmin", concat=True)
    #tasmax = create_output_layer(tas_block, temp3,temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.001), name="tasmax", concat=True)
    sfcwind = create_output_layer(wind_block, temp3,temp2, temp1, num_filters, resize_output,final_activations[1], name="sfcwind", concat=True)
    #sfcwindmax = create_output_layer(wind_block, temp3,temp2, temp1, num_filters, resize_output,tf.keras.layers.LeakyReLU(0.001) name="sfcwindmax", concat=True)
    outputs = [precip_output, sfcwind, tasmin]


    input_layers =[low_res, high_res_fields]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')
    return model


def unet_linear_v3(input_size, resize_output, num_filters, num_channels,
                          final_activations = None):

    """have modified the architecture with a new concatenation layer"""
    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(0.6), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])


    concat_image = high_res_fields#tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2,high_res_fields3])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                   method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    
    # Example: mean across axis 1 and 2
    inputs_abstract = low_res#tf.keras.layers.Concatenate(-1)([low_res, noise])
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=5, i =0)
    x, temp2 = down_block(x, num_filters[1], kernel_size=5, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=5, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [x.shape[1], x.shape[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x])
    tas_block = res_block_initial(x, [128], 3, [1, 1], f"tas_block", sym_padding=True)
    wind_block = res_block_initial(x, [128], 3, [1, 1], f"wind_block", sym_padding=True)
    tasmin = create_output_layer(tas_block, temp3,temp2,temp1, num_filters, resize_output,final_activations[1], name="tasmin", concat=False) 
    sfcwind = create_output_layer(wind_block,temp3,temp2,temp1, num_filters, resize_output,final_activations[1], name="sfcwind", concat=False)
    outputs = [tasmin, sfcwind]
    input_layers =[low_res, high_res_fields]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')
    return model

def unet_rain(input_size, resize_output, num_filters, num_channels,
                          final_activations = None):

    """have modified the architecture with a new concatenation layer"""
    if final_activations is None:
        final_activations = [tf.keras.layers.LeakyReLU(0.6), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1),
                             tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1), tf.keras.layers.LeakyReLU(1)]
    # topography as an input
    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])


    concat_image = high_res_fields#tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2,high_res_fields3])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                   method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    # Example: mean across axis 1 and 2
    inputs_abstract = low_res#tf.keras.layers.Concatenate(-1)([low_res, noise])
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=5, i =0)
    x, temp2 = down_block(x, num_filters[1], kernel_size=5, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=5, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [x.shape[1], x.shape[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x])
    pr_block = res_block_initial(x, [128], 3, [1, 1], f"pr_block", sym_padding=True)
    pr_block = res_block_initial(pr_block, [128], 3, [1, 1], f"pr_block2", sym_padding=True)
    precip_output = create_output_layer(pr_block, temp3,temp2, temp1, num_filters, resize_output,final_activations[0], name="precip_layer", concat=True)
    outputs = [precip_output]
    input_layers =[low_res, high_res_fields]
    model = tf.keras.models.Model(input_layers, outputs, name='unet')
    return model



def create_output_prediction_layer(decoder_output, resize_output, img_inputs,final_activation_function, layer_name,
                                   num_classes=1, concat =True, n_layers=8, initializer_zeros = True, n_fouier_filters=[9, 6]):
        resized_output = tf.image.resize(decoder_output, (resize_output[0], resize_output[1]),
                                 method=tf.image.ResizeMethod.BILINEAR)
        if concat:
            resized_output = tf.keras.layers.Concatenate(-1)([resized_output, img_inputs])
        else:
            resized_output = resized_output

        output = FourierLayer(n_fourier_filters=n_fouier_filters, weight_thres=[0.28, 0.28], name =layer_name+"updated_v5",custom_activation =final_activation_function )(resized_output)
        output = FourierLayer(n_fourier_filters=[n_layers,1], weight_thres=[0.3, 0.3],
                              name=layer_name + "updated_v6", custom_activation=final_activation_function)(output)
        #output = FourierLayer(n_fourier_filters=[n_layers,1], weight_thres=[0.4, 0.4], name =layer_name+"output",custom_activation =final_activation_function)(output)#fourier_layer(output, [n_layers,1], layer_name+"updated_v6", [0.7, 0.5])
        return output

