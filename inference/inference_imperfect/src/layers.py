
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow
import xarray as xr
from dask.diagnostics import ProgressBar
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd


def res_block_initial(x, num_filters, kernel_size, strides, name, sym_padding =True):
    """Residual Unet block layer for first layer
    In the residual unet the first residual block does not contain an
    initial batch normalization and activation so we create this separate
    block for it.
    Args:
        x: tensor, image or image activation
        num_filters: list, contains the number of filters for each subblock
        kernel_size: int, size of the convolutional kernel
        strides: list, contains the stride for each subblock convolution
        name: name of the layer
    Returns:
        x1: tensor, output from residual connection of x and x1
    """

    if len(num_filters) == 1:
        num_filters = [num_filters[0], num_filters[0]]
    if sym_padding:
        x1 = SymmetricPadding2D(padding=[int((kernel_size-1)//2), int((kernel_size-1)//2)])(x)
        x1 = tf.keras.layers.Conv2D(filters=num_filters[0],
                                    kernel_size=kernel_size,
                                    strides=strides[0],
                                    padding='valid',
                                    name=name + '_1')(x1)
    else:
        x1 = tf.keras.layers.Conv2D(filters=num_filters[0],
                                    kernel_size=kernel_size,
                                    strides=strides[0],
                                    padding='same',
                                    name=name + '_1')(x)

    x1 = tf.keras.layers.LeakyReLU(0.1)(x1)
    if sym_padding:
        x1 = SymmetricPadding2D(padding=[int((kernel_size - 1) // 2), int((kernel_size - 1) // 2)])(x1)
        x1 = tf.keras.layers.Conv2D(filters=num_filters[1],
                                    kernel_size=kernel_size,
                                    strides=strides[1],
                                    padding='valid',
                                    name=name + '_2')(x1)

        x = tf.keras.layers.Conv2D(filters=num_filters[-1],
                                   kernel_size=1,
                                   strides=1,
                                   padding='valid',
                                   name=name + '_shortcut')(x)
    else:
        x1 = tf.keras.layers.Conv2D(filters=num_filters[1],
                                    kernel_size=kernel_size,
                                    strides=strides[1],
                                    padding='same',
                                    name=name + '_2')(x1)

        x = tf.keras.layers.Conv2D(filters=num_filters[-1],
                                   kernel_size=1,
                                   strides=1,
                                   padding='same',
                                   name=name + '_shortcut')(x)
        # if bn:
        #
        #     x = tf.keras.layers.BatchNormalization()(x)

    x1 = tf.keras.layers.Add()([x, x1])
    x1 = tf.keras.layers.LeakyReLU(0.1)(x1)
    return x1


class BicubicUpSampling2D(tf.keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super(BicubicUpSampling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs):
        return tf.image.resize(inputs, [int(inputs.shape[1] * self.size[0]), int(inputs.shape[2] * self.size[1])],
                               method=tf.image.ResizeMethod.BILINEAR)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.size
        })
        return config


class SymmetricPadding2D(tf.keras.layers.Layer):

    def __init__(self, padding=[1,1], **kwargs):

        super(SymmetricPadding2D, self).__init__(**kwargs)
        self.padding = padding

    def build(self, input_shape):
        super(SymmetricPadding2D, self).build(input_shape)

    def call(self, inputs):
        if self.padding[0] >1:
            pad = [[0, 0]] + [[1, 1], [1, 1]] + [[0, 0]]
            paddings = tf.constant(pad)
            out = tf.pad(inputs, paddings, "SYMMETRIC")
            for i in range(self.padding[0]-1):
                pad = [[0, 0]] + [[1, 1], [1, 1]] + [[0, 0]]
                paddings = tf.constant(pad)
                out = tf.pad(out, paddings, "SYMMETRIC")
            return out
        else:

            pad = [[0, 0]] + [[1, 1], [1, 1]] + [[0, 0]]
            paddings = tf.constant(pad)
            out = tf.pad(inputs, paddings, "SYMMETRIC")
            return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + self.padding[0],
                input_shape[2] + self.padding[1], input_shape[-1])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'padding': self.padding
        })
        return config



def upsample(x, target_size):
    """"Upsampling function, upsamples the feature map
    Deep Residual Unet paper does not describe the upsampling function
    in detail. Original Unet uses a transpose convolution that downsamples
    the number of feature maps. In order to restrict the number of
    parameters here we use a bilinear resampling layer. This results in
    the concatentation layer concatenting feature maps with n and n/2
    features as opposed to n/2  and n/2 in the original unet.
    Args:
        x: tensor, feature map
        target_size: size to resize feature map to
    Returns:
        x_resized: tensor, upsampled feature map
    """

    x_resized = BicubicUpSampling2D((target_size, target_size))(x)  # tf.keras.layers.Lambda(lambda x: tf.image.resize(x, target_size))(x)
    return x_resized



def conv_block(x, filters, activation, kernel_size=(7, 7), strides=(2, 2), padding="same",
               use_bias=True, use_bn=True, use_dropout=True, drop_value=0.5):
    x = SymmetricPadding2D(padding=[int((kernel_size[0] - 1) // 2), int((kernel_size[0] - 1) // 2)])(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding='same', use_bias=use_bias)(x)
    x = tf.keras.layers.LeakyReLU(0.01)(x)

    return x


def decoder_noise(x, num_filters, kernel_size):
    """Unet decoder
    Args:
        x: tensor, output from previous layer
        encoder_output: list, output from all previous encoder layers
        num_filters: list, number of filters for each decoder layer
        kernel_size: int, size of the convolutional kernel
    Returns:
        x: tensor, output from last layer of decoder
    """
    noise_inputs = []# at some intermediate layers
    for i in range(1, len(num_filters) + 1):
        layer2 = 'decoder_layer_v2' + str(i)
        x = upsample(x, 2)
        x = res_block_initial(x, [num_filters[-i]], kernel_size, strides=[1, 1], name='decoder_layer_v2' + str(i),
                              sym_padding =False)
    return x, noise_inputs


def down_block(x, filters, kernel_size, i =1, use_pool=True, method ='unet', sym_padding =True):

    x = res_block_initial(x, [filters], kernel_size, strides=[1, 1],
                          name='decoder_layer_v2' + str(i),
                              sym_padding = sym_padding)
    if use_pool == True:
        return tf.keras.layers.AveragePooling2D(strides=(2, 2))(x), x
    else:
        return x


def up_block(x, y, filters, kernel_size, i =1, method ='unet', concat = True, sym_padding =True):
    x = upsample(x, 2)
    if concat:
        x = tf.keras.layers.Concatenate(axis=-1)([x, y])
    x = res_block_initial(x, [filters], kernel_size, strides=[1, 1],
                          name='encoder_layer_v2' + str(i),sym_padding = sym_padding)
    return x



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

        self.real_kernel = self.add_weight(
            shape=(1, input_shape[-1] * self.n_modes * self.filters, 1, 1),
            initializer='random_normal',
            trainable=True,
            name='real_kernel'
        )
        self.imag_kernel = self.add_weight(
            shape=(1, input_shape[-1] * self.n_modes * self.filters, 1, 1),
            initializer='random_normal',
            trainable=True,
            name='imag_kernel'
        )
        
        self.weighting_matrix = tf.cast(tf.repeat(
            tf.repeat(tf.expand_dims(tf.expand_dims(self.weight_matrix, axis=-1), axis=-2), input_shape[-1],
                      axis=-2), self.filters, axis=-1), 'float32')
        self.indices = tf.where(tf.less(self.weighting_matrix, self.weight_thres))
        self.complex_output_final = tf.cast(tf.zeros(self.weighting_matrix.shape), 'float32')

    #@tf.custom_gradient
    def custom_operation(self, real_kernel, imag_kernel, inputs):

        def custom_scatter_nd(indices, updates, shape):
            result = tf.scatter_nd(tf.cast(indices, 'int64'), tf.squeeze(updates), tf.cast(shape, 'int64'))
            result = tf.signal.fftshift(result,axes =[0,1])
            return result

        fft_inputs = tf.signal.fft3d(tf.cast(inputs,'complex64'))
        real_kernel_only = custom_scatter_nd(self.indices, real_kernel, self.complex_output_final.shape)
        complex_kernel_only = custom_scatter_nd(self.indices, imag_kernel, self.complex_output_final.shape)
        complex_kernel = tf.complex(real_kernel_only, complex_kernel_only),  # real_kernel, complex_kernel)
        complex_output = tf.einsum('abcd,bcde-> abce', fft_inputs,  complex_kernel[0])
        real_output = tf.cast(tf.math.real(tf.signal.ifft3d(complex_output)), 'float32')

        # def grad(dy):
        #     grad_K = tf.cast(tf.math.real(tf.signal.ifft3d(complex_kernel)), 'float32')
        #     grad_K = tf.reduce_mean(grad_K, axis=-2)
        #     grad_inputs = tf.matmul(dy, grad_K, transpose_b=True)
        #
        #     real_fouier = tf.cast(real_kernel_only,'complex64') * fft_inputs
        #     inverse_real = tf.cast(tf.math.real(tf.signal.ifft3d(real_fouier)), 'float32')
        #     complex_fouier = tf.cast(complex_kernel_only,'complex64') * fft_inputs
        #     inverse_imag = tf.cast(tf.math.real(tf.signal.ifft3d(complex_fouier)), 'float32')
        #     grad_real = tf.reduce_mean(inverse_real * tf.expand_dims(dy, axis=3), axis =0)
        #     grad_real = tf.reshape(tf.gather_nd(grad_real, self.indices), real_kernel.shape)
        #     grad_imag= tf.reduce_mean(inverse_imag * tf.expand_dims(dy, axis=3), axis =0)
        #     grad_imag = tf.reshape(tf.gather_nd(grad_imag, self.indices), real_kernel.shape)
        #
        #     return grad_real, grad_imag, grad_inputs

        return real_output#, grad
        # super(ComplexLinear, self).build(input_shape)

    def call(self, inputs):
        return self.custom_operation(self.real_kernel, self.imag_kernel, inputs)


class FourierLayer(tf.keras.layers.Layer):
    def __init__(self, n_fourier_filters, weight_thres, name=None, custom_activation = None, bn = True, **kwargs):
        super(FourierLayer, self).__init__(name=name, **kwargs)
        self.n_fourier_filters = n_fourier_filters
        self.weight_thres = weight_thres
        self.bn = bn

        # Initialize layers here
        self.conv2d = tf.keras.layers.Conv2D(filters=n_fourier_filters[-1],
                                             kernel_size=1,
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

