
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow
import xarray as xr
from dask.diagnostics import ProgressBar
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd

batch_size = 32
rx3day_file = None
import tensorflow.keras.backend as K


class WGAN_Cascaded_IP(keras.Model):
    """
    A residual GAN to downscale precipitatoin, this GAN incorparates an Intensity Constraint
    """

    def __init__(self, discriminator, generator, latent_dim,
                 discriminator_extra_steps=3, gp_weight=10.0, ad_loss_factor=1e-3,
                 latent_loss=5e-2, orog=None, he=None,
                 vegt=None, unet=None, train_unet=True, intensity_weight = 1, average_intensity_weight =0.0, land_weight = 5):
        super(WGAN_Cascaded_IP, self).__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.ad_loss_factor = ad_loss_factor
        self.latent_loss = latent_loss
        self.orog = orog
        self.he = he
        self.vegt = vegt
        self.unet = unet
        self.train_unet = train_unet
        self.intensity_weight = intensity_weight
        self.average_itensity_weight = average_intensity_weight
        self.land_weight = land_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn,
                g_loss_fn, u_loss_fn, u_optimizer):
        super(WGAN_Cascaded_IP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.u_loss_fn = u_loss_fn
        self.u_optimizer = u_optimizer

    def gradient_penalty(self, batch_size, real_images, fake_images, average, orog_vector,
                         unet_preds):
        """
        need to modify
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator([interpolated, average, orog_vector, unet_preds],
                                      training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @staticmethod
    def expand_conditional_inputs(X, batch_size):
        expanded_image = tf.expand_dims(X, axis=0)  # Shape: (1, 172, 179)

        # Repeat the image to match the desired batch size
        expanded_image = tf.repeat(expanded_image, repeats=batch_size, axis=0)  # Shape: (batch_size, 172, 179)

        # Create a new axis (1) on the last axis
        expanded_image = tf.expand_dims(expanded_image, axis=-1)
        return expanded_image
    
    @staticmethod
    def process_real_images(real_images_obj):
        output_vars, averages = real_images_obj  # Unpack the input

        # Extract relevant variables from the output_vars dictionary
        real_images = [
            output_vars['pr']
        ]

        real_images_future = [
            output_vars['pr_future'],
            
        ]

        # Extract average and average_future
        average = averages["X"]
        average_future = averages["X_future"]

        # Combine variables into single tensors
        real_images = tf.concat([tf.expand_dims(img, axis=-1) for img in real_images], axis=-1)
        real_images_future = tf.concat([tf.expand_dims(img, axis=-1) for img in real_images_future], axis=-1)

        # Combine all GCMs into one single batch timestep
        real_images = tf.concat([real_images[:, :, :, i, :] for i in range(real_images.shape[3])], axis=0)
        real_images_future = tf.concat([real_images_future[:, :, :, i, :] for i in range(real_images_future.shape[3])],
                                       axis=0)
        average = tf.concat([average[:, :, :, i, :] for i in range(average.shape[3])], axis=0)
        average_future = tf.concat([average_future[:, :, :, i, :] for i in range(average_future.shape[3])], axis=0)
        
        average_combined = tf.concat([average, average_future], axis =0)
        real_images_combined = tf.concat([real_images, real_images_future], axis =0)
        return real_images_combined, average_combined

    def train_step(self, real_images):
        real_images, average = self.process_real_images(real_images)

            # here the average represents the conditional input

        batch_size = tf.shape(real_images)[0]
        orog_vector = self.expand_conditional_inputs(self.orog, batch_size)
        # make sure the auxiliary inputs are the same shape as the training batch
        # if the U-Net is trained, apply gradients otherwise only use inference mode from the U-Net
        if self.train_unet:
            with tf.GradientTape() as tape:
                random_latent_vectors = tf.random.normal(
                    shape=(batch_size,) + self.latent_dim[0]
                )

                init_prediction = self.unet([average,
                                             orog_vector], training=True)
                mae_unet = self.u_loss_fn(real_images[:, :, :], init_prediction)
            u_gradient = tape.gradient(mae_unet, self.unet.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.u_optimizer.apply_gradients(zip(u_gradient, self.unet.trainable_variables))
        else:
            with tf.GradientTape() as tape:
                random_latent_vectors = tf.random.normal(
                    shape=(batch_size,) + self.latent_dim[0]
                )
                random_latent_vectors1 = tf.random.normal(
                    shape=(batch_size,) + self.latent_dim[1]
                )

                init_prediction = self.unet([average,
                                             orog_vector], training=True)
                mae_unet = self.u_loss_fn(real_images[:, :, :], init_prediction)
        # loop through the discriminator steps
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size,) + self.latent_dim[0]
            )
            random_latent_vectors1 = tf.random.normal(
                shape=(batch_size,) + self.latent_dim[1]
            )

            with tf.GradientTape() as tape:

                init_prediction_unet = self.unet([average,
                                                  orog_vector], training=True)
                # compute ground truth residuals
                residual_gt = (real_images - init_prediction_unet)
                init_prediction = init_prediction_unet
                # crete fake residuals (these are residual by default)
                fake_images = self.generator([random_latent_vectors, random_latent_vectors1, average,
                                              orog_vector, init_prediction], training=True)

                fake_logits = self.discriminator(
                    [fake_images, average, orog_vector, init_prediction], training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(
                    [residual_gt, average, orog_vector, init_prediction], training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, residual_gt, fake_images, average, orog_vector, init_prediction)

                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight  # + #50 * tf.keras.losses.mean_squared_error(average, fake_image_average)

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # Train the generator

        # Generator steps
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            init_prediction_unet = self.unet([average,
                                              orog_vector], training=True)

            init_prediction = init_prediction_unet  # (init_prediction_unet - min_value)/(max_value - min_value)
            # compute ground truth residuals
            residual_gt = (real_images - init_prediction_unet)
            # creatinging an ensemble
            random_latent_vectors = tf.random.normal(
            shape=(3,) + (batch_size,) + self.latent_dim[0]
        )
            random_latent_vectors1 = tf.random.normal(
            shape=(3,) + (batch_size,) + self.latent_dim[1]
        )
            generated_images_v2 = [tf.expand_dims(self.generator(
                [random_latent_vectors[i],random_latent_vectors1[i], average, orog_vector, init_prediction],
                training=True), axis =0) for i in range(3)]
            generated_images_v1 = generated_images_v2[0][0]
            # an ensemble mean across 8-members
            generated_images_v2 = tf.math.reduce_mean(tf.concat(generated_images_v2, axis =0), axis =0)

            generated_images = generated_images_v1  # tf.math.exp(generated_images_v1[:,:,:,0] +
            # residual predictions from the GAN

            gen_img_logits = self.discriminator(
                [generated_images, average, orog_vector, init_prediction], training=True)
            # compute the content loss or the MSE, this is the errors in the residuals
            mae = tf.keras.losses.mean_squared_error(residual_gt, generated_images_v2)
            # compute the "true" error.
            
            gan_mae = mae#tf.keras.losses.mean_squared_error(residual_gt, generated_images_v2)

            # compute the intensity on the batch across each individual timestep (not the 0th dimension)
            if self.land_weight >0:
                
                mask_orog = tf.cast(tf.squeeze(orog_vector) > 0.0, 'float32')
                land_mae =tf.keras.losses.mean_squared_error(tf.squeeze(residual_gt) * mask_orog , tf.squeeze(generated_images_v2) * mask_orog) 
                ocean_mae =tf.keras.losses.mean_squared_error(tf.squeeze(residual_gt) * (1- mask_orog) , tf.squeeze(generated_images_v2) * (1-mask_orog))
                gamma_loss_func = (1/(self.land_weight +1)) *  (self.land_weight *land_mae + ocean_mae)
           
            else:
                gamma_loss_func = mae
            maximum_intensity = tf.math.reduce_max(
                real_images, axis=[-1, -2, -3])
            maximum_intensity_predicted = tf.math.reduce_max(generated_images_v1 + init_prediction_unet,
                                                             axis=[-1, -2, -3])

            average_intensity = tf.math.reduce_mean(
                real_images, axis=[-1, -4])
            average_intensity_predicted = tf.math.reduce_mean(generated_images_v1 + init_prediction_unet,
                                                              axis=[-1, -4])

            average_intensity_error = tf.reduce_mean(
               tf.abs(average_intensity - average_intensity_predicted) ** 2)
            maximum_intensity_error = tf.reduce_mean(
                tf.abs(maximum_intensity - maximum_intensity_predicted) ** 2)
            adv_loss = self.ad_loss_factor * self.g_loss_fn(gen_img_logits)
            # Calculate the generator loss
            g_loss = adv_loss + gamma_loss_func + self.average_itensity_weight * average_intensity_error + self.intensity_weight * maximum_intensity_error ## + self.latent_loss * latent_loss
        # + tf.reduce_mean(
        #     tf.abs(average_intensity - average_intensity_predicted)) ** 2
        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss, "residual_loss": gamma_loss_func, "adv_loss": adv_loss,
                "unet_loss": mae_unet, "gan_mae": gan_mae}



class WGAN_Cascaded_Multi_v3(keras.Model):
    """
    adapted from https://arxiv.org/pdf/2207.01561.pdf
    https://arxiv.org/pdf/1903.05628.pdf

    Also from https://arxiv.org/pdf/1903.05628.pdf

    It is also likely that our GAN suffers from MODE collapse and has an inability to generate diversity
s
    Added static vegetation inputs as a predictor
    """

    def __init__(self, discriminator, generator, latent_dim,
                 discriminator_extra_steps=3, gp_weight=10.0, ad_loss_factor=1e-3, latent_loss=5e-2, orog=None, he=None,
                 vegt=None, unet=None, train_gan=True, train_unet=True, loss_multiplier_tmax=3,
                 loss_multiplier_sfcwind=2.5, land_loss_weight =3):
        super(WGAN_Cascaded_Multi_v3, self).__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.ad_loss_factor = ad_loss_factor
        self.latent_loss = latent_loss
        self.orog = orog
        self.he = he
        self.vegt = vegt
        self.unet = unet
        self.train_gan = train_gan
        self.train_unet = train_unet
        self.loss_multiplier_tmax = loss_multiplier_tmax
        self.loss_multiplier_sfcwind = loss_multiplier_sfcwind
        self.land_loss_weight = land_loss_weight

    def compile(self, d_optimizer, g_optimizer,
                d_loss_fn, g_loss_fn, u_loss_fn, u_optimizer):
        super(WGAN_Cascaded_Multi_v3, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.u_loss_fn = u_loss_fn
        self.u_optimizer = u_optimizer

    @staticmethod
    def gradient_penalty(discriminator, batch_size, real_images, fake_images, average, orog_vector,
                         unet_preds):
        """
        need to modify
        """
        #[img_input, img_input2, img_input3, img_input4, img_input5, img_input6,img_input7]
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = discriminator([interpolated, average, orog_vector, unet_preds],
                                 training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @staticmethod
    def expand_conditional_inputs(X, batch_size):
        expanded_image = tf.expand_dims(X, axis=0)  # Shape: (1, 172, 179)

        # Repeat the image to match the desired batch size
        expanded_image = tf.repeat(expanded_image, repeats=batch_size, axis=0)  # Shape: (batch_size, 172, 179)

        # Create a new axis (1) on the last axis
        expanded_image = tf.expand_dims(expanded_image, axis=-1)
        return expanded_image
    @staticmethod
    def process_real_images(real_images_obj):
        output_vars, averages = real_images_obj  # Unpack the input

        # Extract relevant variables from the output_vars dictionary
        real_images = [
            output_vars['pr'],
            output_vars['tasmin'],
            output_vars['tasmax'],
            output_vars['sfcWind'],
            output_vars['sfcWindmax'],
        ]

        real_images_future = [
            output_vars['pr_future'],
            output_vars['tasmin_future'],
            output_vars['tasmax_future'],
            output_vars['sfcWind_future'],
            output_vars['sfcWindmax_future']
            
        ]

        # Extract average and average_future
        average = averages["X"]
        average_future = averages["X_future"]
        
        
        time_of_year_hist = averages["time_of_year_hist"]
        time_of_year_future = averages["time_of_year_future"]
        
        spatial_means_hist = averages["spatial_means_hist"]
        spatial_means_future = averages["spatial_means_future"]
        
        spatial_stds_hist = averages["spatial_stds_hist"]
        spatial_stds_future = averages["spatial_stds_future"]

        # Combine variables into single tensors
        real_images = tf.concat([tf.expand_dims(img, axis=-1) for img in real_images], axis=-1)
        real_images_future = tf.concat([tf.expand_dims(img, axis=-1) for img in real_images_future], axis=-1)

        # Combine all GCMs into one single batch timestep
        real_images = tf.concat([real_images[:, :, :, i, :] for i in range(real_images.shape[3])], axis=0)
        real_images_future = tf.concat([real_images_future[:, :, :, i, :] for i in range(real_images_future.shape[3])],
                                       axis=0)
        average = tf.concat([average[:, :, :, i, :] for i in range(average.shape[3])], axis=0)
        average_future = tf.concat([average_future[:, :, :, i, :] for i in range(average_future.shape[3])], axis=0)
        
        time_of_year_hist = tf.concat([time_of_year_hist[:, i] for i in range(time_of_year_hist.shape[1])], axis=0)
        time_of_year_future = tf.concat([time_of_year_future[:, i] for i in range(time_of_year_future.shape[1])], axis=0)
        
        spatial_means_hist = tf.concat([spatial_means_hist[:, i] for i in range(spatial_means_hist.shape[1])], axis=0)
        spatial_means_future = tf.concat([spatial_means_future[:, i] for i in range(spatial_means_future.shape[1])], axis=0)
        
        spatial_stds_hist = tf.concat([spatial_stds_hist[:, i] for i in range(spatial_stds_hist.shape[1])], axis=0)
        spatial_stds_future = tf.concat([spatial_stds_future[:, i] for i in range(spatial_stds_future.shape[1])], axis=0)
        
        spatial_stds_combined = tf.concat([spatial_stds_hist, spatial_stds_future], axis =0)
        spatial_means_combined = tf.concat([spatial_means_hist, spatial_means_future], axis =0)
        time_of_year_combined = tf.concat([time_of_year_hist, time_of_year_future], axis =0)
        
        average_combined = tf.concat([average, average_future], axis =0)
        real_images_combined = tf.concat([real_images, real_images_future], axis =0)
        return real_images_combined, average_combined, spatial_means_combined, spatial_stds_combined, time_of_year_combined


    def unet_pass(self, average_combined, orog_vector, real_images_combined,spatial_means_combined, spatial_stds_combined, time_of_year_combined):
        # Generate predictions for both current and future conditions
        predictions = self.unet([average_combined, orog_vector, time_of_year_combined, spatial_means_combined,
                                 spatial_stds_combined], training=True)
        rainfall_unet, tasmin_unet, tasmax_unet, sfcwind_unet, sfcwindmax_unet = predictions
        # Compute losses
        def compute_loss(real, pred):
            return self.u_loss_fn(real, pred)
        orog_vec_mask = tf.cast(tf.squeeze(orog_vector)>0.001, 'float32')
        loss_rain_ocean = compute_loss(tf.squeeze(real_images_combined[:, :, :, 0:1]) * (1-orog_vec_mask), tf.squeeze(rainfall_unet) * (1-orog_vec_mask))
        
        loss_sfcwind_ocean = compute_loss(tf.squeeze(real_images_combined[:, :, :, 3:4]) * (1-orog_vec_mask), tf.squeeze(sfcwind_unet) * (1-orog_vec_mask))
        loss_sfcwindmax_ocean = compute_loss(tf.squeeze(real_images_combined[:, :, :, 4:5]) * (1-orog_vec_mask), tf.squeeze(sfcwindmax_unet) * (1-orog_vec_mask))
        loss_tasmin_ocean = compute_loss(tf.squeeze(real_images_combined[:, :, :, 1:2]) * (1-orog_vec_mask), tf.squeeze(tasmin_unet)* (1-orog_vec_mask))
        loss_tasmax_ocean = compute_loss(tf.squeeze(real_images_combined[:, :, :, 2:3]) * (1-orog_vec_mask), tf.squeeze(tasmax_unet)* (1-orog_vec_mask))
        
        loss_rain_land = compute_loss(tf.squeeze(real_images_combined[:, :, :, 0:1]) * orog_vec_mask, tf.squeeze(rainfall_unet) * orog_vec_mask)
        loss_sfcwind_land = compute_loss(tf.squeeze(real_images_combined[:, :, :, 3:4]) * orog_vec_mask, tf.squeeze(sfcwind_unet) * orog_vec_mask)
        loss_sfcwindmax_land = compute_loss(tf.squeeze(real_images_combined[:, :, :, 4:5]) * orog_vec_mask, tf.squeeze(sfcwindmax_unet) * orog_vec_mask)
        loss_tasmin_land = compute_loss(tf.squeeze(real_images_combined[:, :, :, 1:2]) * orog_vec_mask, tf.squeeze(tasmin_unet)* orog_vec_mask)
        loss_tasmax_land = compute_loss(tf.squeeze(real_images_combined[:, :, :, 2:3]) * orog_vec_mask, tf.squeeze(tasmax_unet)* orog_vec_mask)
        return {
            "loss_rain": (loss_rain_ocean + 4* loss_rain_land)/5.0,
            "loss_sfcwind": (loss_sfcwind_ocean + 4* loss_sfcwind_land)/5.0,
            "loss_tasmin": (loss_tasmin_ocean + 4 * loss_tasmin_land)/5.0, 
            "loss_tasmax": (loss_tasmax_ocean + 4 * loss_tasmax_land)/5.0,
            "loss_sfcwindmax": (loss_sfcwindmax_ocean + 4*loss_sfcwindmax_land)/5.0}
    def gan_pass(self, random_latent_vectors, random_latent_vectors1, average_combined, orog_vector,
                 real_images,spatial_means_combined, spatial_stds_combined,
                 time_of_year_combined):
        
        unet_predictions = self.unet([average_combined, orog_vector,
                                  time_of_year_combined,spatial_means_combined,
                                 spatial_stds_combined], training=True)
        rainfall_unet, tasmin_unet,tasmax_unet, sfcwind_unet, sfcwindmax_unet = unet_predictions
        unet_predictions = [rainfall_unet, tasmin_unet, tasmax_unet, sfcwind_unet, sfcwindmax_unet]
        generator_preds = self.generator([random_latent_vectors, random_latent_vectors1, average_combined, orog_vector,rainfall_unet,
                                          sfcwind_unet, tasmin_unet,tasmax_unet, sfcwindmax_unet, time_of_year_combined,
                                         spatial_means_combined, spatial_stds_combined], training=True)
        rainfall_gan, tasmin_gan, tasmax_gan, sfcwind_gan, sfcwindmax_gan = generator_preds
        
        
        orog_vec_mask = tf.cast(tf.squeeze(orog_vector) > 0.001, 'float32')

        # Ocean losses (1 - orog_vec_mask)
        loss_rain_ocean_gan = self.u_loss_fn(tf.squeeze((real_images[:, :, :, 0:1] - rainfall_unet)) * (1 - orog_vec_mask), 
                                             tf.squeeze(rainfall_gan) * (1 - orog_vec_mask))
        loss_sfcwind_ocean_gan = self.u_loss_fn(tf.squeeze((real_images[:, :, :, 3:4] - sfcwind_unet)) * (1 - orog_vec_mask), tf.squeeze(sfcwind_gan) * (1 - orog_vec_mask))
        loss_sfcwindmax_ocean_gan = self.u_loss_fn(tf.squeeze((real_images[:, :, :, 4:5] - sfcwindmax_unet)) * (1 - orog_vec_mask), tf.squeeze(sfcwindmax_gan) * (1 - orog_vec_mask))
        loss_tasmax_ocean_gan = self.u_loss_fn(tf.squeeze((real_images[:, :, :, 2:3] - tasmax_unet)) * (1 - orog_vec_mask), tf.squeeze(tasmax_gan) * (1 - orog_vec_mask))
        loss_tasmin_ocean_gan = self.u_loss_fn(tf.squeeze((real_images[:, :, :, 1:2] - tasmin_unet)) * (1 - orog_vec_mask), tf.squeeze(tasmin_gan) * (1 - orog_vec_mask))

        # Land losses (orog_vec_mask)
        loss_rain_land_gan = self.u_loss_fn(tf.squeeze((real_images[:, :, :, 0:1] - rainfall_unet)) * orog_vec_mask, tf.squeeze(rainfall_gan) * orog_vec_mask)
        loss_sfcwind_land_gan = self.u_loss_fn(tf.squeeze((real_images[:, :, :, 3:4] - sfcwind_unet)) * orog_vec_mask, tf.squeeze(sfcwind_gan) * orog_vec_mask)
        loss_sfcwindmax_land_gan = self.u_loss_fn(tf.squeeze((real_images[:, :, :, 4:5] - sfcwindmax_unet)) * orog_vec_mask, tf.squeeze(sfcwindmax_gan) * orog_vec_mask)
        loss_tasmax_land_gan = self.u_loss_fn(tf.squeeze((real_images[:, :, :, 2:3] - tasmax_unet)) * orog_vec_mask, tf.squeeze(tasmax_gan) * orog_vec_mask)
        loss_tasmin_land_gan = self.u_loss_fn(tf.squeeze((real_images[:, :, :, 1:2] - tasmin_unet)) * orog_vec_mask, tf.squeeze(tasmin_gan) * orog_vec_mask)



        loss_rain_gan = (5*loss_rain_land_gan + loss_rain_ocean_gan)/6.0#self.u_loss_fn(real_images[:, :, :, 0:1] - rainfall_unet, rainfall_gan)
        loss_sfcwind_gan = (5*loss_sfcwind_land_gan + loss_sfcwind_ocean_gan)/6.0#elf.u_loss_fn(real_images[:, :, :, 1:2] - sfcwind_unet, sfcwind_gan)
        loss_sfcwindmax_gan = (5*loss_sfcwindmax_land_gan + loss_sfcwindmax_ocean_gan)/6.0#self.u_loss_fn(real_images[:, :, :, 2:3] - sfcwind_unet, sfcwindmax_gan)
 
        loss_tasmax_gan = (5*loss_tasmax_land_gan + 1* loss_tasmax_ocean_gan)/6.0#self.u_loss_fn(real_images[:, :, :, 3:4] - tasmin_unet, tasmax_gan)
        loss_tasmin_gan = (loss_tasmin_ocean_gan + 5*loss_tasmin_land_gan)/6.0#self.u_loss_fn(real_images[:, :, :, 4:5] - tasmin_unet, tasmin_gan)
        
        
        total_loss_mse = loss_rain_gan + self.loss_multiplier_sfcwind * (loss_sfcwind_gan + loss_sfcwindmax_gan) \
                         + self.loss_multiplier_tmax * (loss_tasmin_gan + loss_tasmax_gan)

        # Intensity Constraint

        maximum_intensity_rain = tf.math.reduce_max(
            real_images[:, :, :, 0:1], axis=[-1, -2, -3])
        maximum_intensity_predicted = tf.math.reduce_max(rainfall_gan + rainfall_unet,
                                                         axis=[-1, -2, -3])

        int_rain = self.u_loss_fn(maximum_intensity_rain, maximum_intensity_predicted)

        maximum_intensity_sfcwind = tf.math.reduce_max(
            real_images[:, :, :, 3:4], axis=[-1, -2, -3])
        maximum_intensity_predicted_sfcwind = tf.math.reduce_max(sfcwind_unet + sfcwind_gan,
                                                                 axis=[-1, -2, -3])
        int_sfcwind = self.u_loss_fn(maximum_intensity_sfcwind,
                                     maximum_intensity_predicted_sfcwind)

        maximum_intensity_sfcwindmax = tf.math.reduce_max(
            real_images[:, :, :, 4:5], axis=[-1, -2, -3])
        maximum_intensity_predicted_sfcwindmax = tf.math.reduce_max(sfcwindmax_gan+sfcwindmax_unet,
                                                                    axis=[-1, -2, -3])
        int_sfcwindmax = self.u_loss_fn(maximum_intensity_sfcwindmax, maximum_intensity_predicted_sfcwindmax)

        maximum_intensity_tasmax = tf.math.reduce_max(
            real_images[:, :, :, 2:3], axis=[-1, -2, -3])
        maximum_intensity_predicted_tasmax = tf.math.reduce_max(tasmax_gan+tasmax_unet,
                                                                axis=[-1, -2, -3])
        int_tasmax = self.u_loss_fn(maximum_intensity_tasmax, maximum_intensity_predicted_tasmax)

        maximum_intensity_tasmin = tf.math.reduce_max(
            real_images[:, :, :, 1:2], axis=[-1, -2, -3])
        maximum_intensity_predicted_tasmin = tf.math.reduce_max(tasmin_unet + tasmin_gan,
                                                                axis=[-1, -2, -3])
        int_tasmin = self.u_loss_fn(maximum_intensity_tasmin, maximum_intensity_predicted_tasmin)
        total_loss_constraint = int_rain + int_sfcwind + int_sfcwindmax + int_tasmax + int_tasmin
        # to avoid any issues with adding contributions
        adv_losses = []        
        for n_gans_adv in range(len(self.discriminator)):
            fake_logits = self.discriminator[n_gans_adv](
                [generator_preds[n_gans_adv], average_combined, orog_vector, unet_predictions[n_gans_adv]],
                training=True)

            # add a maximum penality for each variable
            adv_loss_individual = self.ad_loss_factor * self.g_loss_fn(fake_logits)
            adv_losses.append(adv_loss_individual)
        # Calculate the generator loss

        g_loss = (adv_losses[0] + adv_losses[1] + adv_losses[2] + adv_losses[3] + adv_losses[
            4])/5.0 + 1/5.0 * total_loss_mse + 1/25.0 * total_loss_constraint  # + self.latent_loss * latent_loss
        return g_loss, total_loss_mse


    def train_step(self, real_images):
        real_images_combined, average_combined, spatial_means_combined, spatial_stds_combined, time_of_year_combined = self.process_real_images(real_images)
        batch_size = tf.shape(real_images_combined)[0]  # this should now be N_GCM times the average
        orog_vector = self.expand_conditional_inputs(self.orog, batch_size)
        config = {}

        if self.train_unet:
            with tf.GradientTape() as tape:
                random_latent_vectors = tf.random.normal(
                    shape=(batch_size,) + self.latent_dim[0]
                )
                losses = self.unet_pass(average_combined, orog_vector, real_images_combined,
                                        spatial_means_combined, spatial_stds_combined, time_of_year_combined)
                loss_weights = self.loss_multiplier_sfcwind + self.loss_multiplier_tmax + 1

                total_loss_unet = (1/ loss_weights) *  (losses["loss_rain"] + self.loss_multiplier_sfcwind/2 * (losses["loss_sfcwind"] + losses["loss_sfcwindmax"]) + \
                             self.loss_multiplier_tmax/2 * (losses["loss_tasmin"] + losses["loss_tasmax"])) 
            u_gradient = tape.gradient(total_loss_unet, self.unet.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.u_optimizer.apply_gradients(zip(u_gradient, self.unet.trainable_variables))
            config = losses
        if self.train_gan:
            for n_gans in range(len(self.discriminator)):

                # Get the latent vector
                random_latent_vectors = tf.random.normal(
                    shape=(batch_size,) + self.latent_dim[0]
                )
                random_latent_vectors1 = tf.random.normal(
                    shape=(batch_size,) + self.latent_dim[1]
                )

                # have added two versions of the latent vectors


                predictions = self.unet([average_combined, orog_vector,time_of_year_combined, spatial_means_combined,
                                 spatial_stds_combined], training=True)
                rainfall_unet, tasmin_unet,tasmax_unet, sfcwind_unet, sfcwindmax_unet = predictions
                
                #creating a prediction list
                predictions = [rainfall_unet, tasmin_unet, tasmax_unet, sfcwind_unet, sfcwindmax_unet]

                generator_preds_historical = self.generator([random_latent_vectors, random_latent_vectors1,
                                                             average_combined, orog_vector,
                                                             rainfall_unet, sfcwind_unet, tasmin_unet,tasmax_unet, sfcwindmax_unet,
                                                             time_of_year_combined, spatial_means_combined, 
                                                            spatial_stds_combined], training=True)
                # here we introduce a gan for each individual variable
                for i in range(self.d_steps):
                    with tf.GradientTape() as tape:

                        fake_logits_historical = self.discriminator[n_gans](
                            [generator_preds_historical[n_gans], average_combined, orog_vector, predictions[n_gans]],
                            training=True)
                        # Get the logits for the real images
                        # modified this line to now predict the residuals of the solution

                        real_logits_historical = self.discriminator[n_gans](
                            [real_images_combined[:, :, :, n_gans:n_gans + 1] - predictions[n_gans], average_combined, orog_vector,
                             predictions[n_gans]],
                            training=True)
                        gp_hist = self.gradient_penalty(self.discriminator[n_gans], batch_size,
                                               real_images_combined[:, :, :, n_gans:n_gans + 1] - predictions[n_gans],
                                               generator_preds_historical[n_gans],
                                               average_combined, orog_vector, predictions[n_gans])

                        d_cost_hist = self.d_loss_fn(real_img=real_logits_historical, fake_img=fake_logits_historical)


                        # Add the gradient penalty to the original discriminator loss
                        d_loss = d_cost_hist + gp_hist * self.gp_weight

                    # Get the gradients w.r.t the discriminator loss
                    d_gradient = tape.gradient(d_loss, self.discriminator[n_gans].trainable_variables)
                    # Update the weights of the discriminator using the discriminator optimizer
                    self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator[n_gans].trainable_variables))

            with tf.GradientTape() as tape:
                """
                Introducing the Maximum and Average Penalty in the Loss function for each variable 
                """
                historical_loss, total_loss_mse = self.gan_pass(random_latent_vectors, random_latent_vectors1, average_combined, orog_vector,
                 real_images_combined,spatial_means_combined, spatial_stds_combined,
                 time_of_year_combined)
            total_loss = historical_loss
            # Get the gradients w.r.t the generator loss
            gen_gradient = tape.gradient(total_loss, self.generator.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

            config = {"d_loss": d_loss, "g_loss": total_loss, "unet_loss":total_loss_mse}

        return config

    

class RainGAN(keras.Model):
    """
four variables then we predict a separate rAINFll model
    """

    def __init__(self, discriminator, generator, latent_dim,
                 discriminator_extra_steps=3, gp_weight=10.0, ad_loss_factor=1e-3, latent_loss=5e-2, orog=None, he=None,
                 vegt=None, unet=None, train_gan=True, train_unet=True, land_loss_weight =5, loss_weight =1):
        super(RainGAN, self).__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.ad_loss_factor = ad_loss_factor
        self.latent_loss = latent_loss
        self.orog = orog
        self.he = he
        self.vegt = vegt
        self.unet = unet
        self.train_gan = train_gan
        self.train_unet = train_unet
        self.land_loss_weight = land_loss_weight
        self.loss_weight = loss_weight

    def compile(self, d_optimizer, g_optimizer,
                d_loss_fn, g_loss_fn, u_loss_fn, u_optimizer):
        super(RainGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.u_loss_fn = u_loss_fn
        self.u_optimizer = u_optimizer

    @staticmethod
    def gradient_penalty(discriminator, batch_size, real_images, fake_images, average, orog_vector,
                         unet_preds):
        """
        need to modify
        """
        #[img_input, img_input2, img_input3, img_input4, img_input5, img_input6,img_input7]
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = discriminator([interpolated, average, orog_vector, unet_preds],
                                 training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @staticmethod
    def expand_conditional_inputs(X, batch_size):
        expanded_image = tf.expand_dims(X, axis=0)  # Shape: (1, 172, 179)

        # Repeat the image to match the desired batch size
        expanded_image = tf.repeat(expanded_image, repeats=batch_size, axis=0)  # Shape: (batch_size, 172, 179)

        # Create a new axis (1) on the last axis
        expanded_image = tf.expand_dims(expanded_image, axis=-1)
        return expanded_image
    @staticmethod
    def process_real_images(real_images_obj):
        output_vars, averages = real_images_obj  # Unpack the input

        # Extract relevant variables from the output_vars dictionary
        real_images = [output_vars['pr'],
        ]

        real_images_future = [output_vars['pr_future'],
            
        ]

        # Extract average and average_future
        average = averages["X"]
        average_future = averages["X_future"]
        
        
        time_of_year_hist = averages["time_of_year_hist"]
        time_of_year_future = averages["time_of_year_future"]
        
        spatial_means_hist = averages["spatial_means_hist"]
        spatial_means_future = averages["spatial_means_future"]
        
        spatial_stds_hist = averages["spatial_stds_hist"]
        spatial_stds_future = averages["spatial_stds_future"]

        # Combine variables into single tensors
        real_images = tf.concat([tf.expand_dims(img, axis=-1) for img in real_images], axis=-1)
        real_images_future = tf.concat([tf.expand_dims(img, axis=-1) for img in real_images_future], axis=-1)

        # Combine all GCMs into one single batch timestep
        real_images = tf.concat([real_images[:, :, :, i, :] for i in range(real_images.shape[3])], axis=0)
        real_images_future = tf.concat([real_images_future[:, :, :, i, :] for i in range(real_images_future.shape[3])],
                                       axis=0)
        average = tf.concat([average[:, :, :, i, :] for i in range(average.shape[3])], axis=0)
        average_future = tf.concat([average_future[:, :, :, i, :] for i in range(average_future.shape[3])], axis=0)
        
        time_of_year_hist = tf.concat([time_of_year_hist[:, i] for i in range(time_of_year_hist.shape[1])], axis=0)
        time_of_year_future = tf.concat([time_of_year_future[:, i] for i in range(time_of_year_future.shape[1])], axis=0)
        
        spatial_means_hist = tf.concat([spatial_means_hist[:, i] for i in range(spatial_means_hist.shape[1])], axis=0)
        spatial_means_future = tf.concat([spatial_means_future[:, i] for i in range(spatial_means_future.shape[1])], axis=0)
        
        spatial_stds_hist = tf.concat([spatial_stds_hist[:, i] for i in range(spatial_stds_hist.shape[1])], axis=0)
        spatial_stds_future = tf.concat([spatial_stds_future[:, i] for i in range(spatial_stds_future.shape[1])], axis=0)
        
        spatial_stds_combined = tf.concat([spatial_stds_hist, spatial_stds_future], axis =0)
        spatial_means_combined = tf.concat([spatial_means_hist, spatial_means_future], axis =0)
        time_of_year_combined = tf.concat([time_of_year_hist, time_of_year_future], axis =0)
        
        average_combined = tf.concat([average, average_future], axis =0)
        real_images_combined = tf.concat([real_images, real_images_future], axis =0)
        return real_images_combined, average_combined, spatial_means_combined, spatial_stds_combined, time_of_year_combined
    
    def unet_pass(self, average_combined, orog_vector, real_images_combined,spatial_means_combined, spatial_stds_combined, time_of_year_combined):
        # Generate predictions for both current and future conditions
        predictions = self.unet([average_combined, orog_vector, time_of_year_combined, spatial_means_combined,
                                 spatial_stds_combined], training=True)
        rainfall_unet = predictions
        # Compute losses
        def compute_loss(real, pred):
            return self.u_loss_fn(real, pred)
        orog_vec_mask = tf.cast(tf.squeeze(orog_vector)>0.001, 'float32')
        loss_rain_ocean = compute_loss(tf.squeeze(real_images_combined[:, :, :, 0:1]) * (1-orog_vec_mask), tf.squeeze(rainfall_unet) * (1-orog_vec_mask))
        
        
        loss_rain_land = compute_loss(tf.squeeze(real_images_combined[:, :, :, 0:1]) * orog_vec_mask, tf.squeeze(rainfall_unet) * orog_vec_mask)
        return {
            "loss_rain": (loss_rain_ocean + 8* loss_rain_land)/9.0}

    def gan_pass(self, random_latent_vectors, random_latent_vectors1, average_combined, orog_vector,
                 real_images,spatial_means_combined, spatial_stds_combined,
                 time_of_year_combined):
    
        unet_predictions = self.unet([average_combined, orog_vector,
                                  time_of_year_combined,spatial_means_combined,
                                 spatial_stds_combined], training=True)
        rainfall_unet = unet_predictions

        generator_preds = self.generator([random_latent_vectors, random_latent_vectors1,
                                          average_combined, orog_vector,rainfall_unet,time_of_year_combined,
                                         spatial_means_combined, spatial_stds_combined], training=True)
        
                        # Get the latent vector
        random_latent_vectors = tf.random.normal(
                shape=(batch_size,) + self.latent_dim[0]
            )
        random_latent_vectors1 = tf.random.normal(
                shape=(batch_size,) + self.latent_dim[1]
            )
        
        rainfall_gan = generator_preds
        
        orog_vec_mask = tf.cast(tf.squeeze(orog_vector) > 0.001, 'float32')
        loss_rain_ocean_gan = self.u_loss_fn(tf.squeeze((real_images[:, :, :, 0:1])) * (1 - orog_vec_mask), tf.squeeze(rainfall_gan+rainfall_unet) * (1 - orog_vec_mask))

        loss_rain_land_gan = self.u_loss_fn(tf.squeeze((real_images[:, :, :, 0:1])) * orog_vec_mask, tf.squeeze(rainfall_gan+rainfall_unet) * orog_vec_mask)

        loss_rain_gan = (5* loss_rain_land_gan + loss_rain_ocean_gan)/6.0#elf.u_loss_fn(real_images[:, :, :, 1:2] - sfcwind_unet, sfcwind_gan)

        average_intensity = tf.math.reduce_mean(
                real_images, axis=[-1, -4])
        average_intensity_predicted = tf.math.reduce_mean(rainfall_unet + rainfall_gan,
                                                              axis=[-1, -4])

        average_intensity_error = tf.reduce_mean(
               tf.abs(average_intensity - average_intensity_predicted) ** 2)
        
        total_loss_mse = loss_rain_gan
        # Intensity Constraint

        maximum_intensity_rain = tf.math.reduce_max(
            real_images[:, :, :, 0:1], axis=[-1, -2, -3])
        maximum_intensity_predicted_rain = tf.math.reduce_max(rainfall_unet + rainfall_gan,
                                                                 axis=[-1, -2, -3])
        int_rain = self.u_loss_fn(maximum_intensity_rain,
                                     maximum_intensity_predicted_rain)
        total_loss_constraint = int_rain
        fake_logits = self.discriminator(
                [rainfall_gan+rainfall_unet, average_combined, orog_vector,rainfall_unet],
                training=True)

            # add a maximum penality for each variable
        adv_loss_individual = self.ad_loss_factor * self.g_loss_fn(fake_logits)
        # Calculate the generator loss

        g_loss = adv_loss_individual + self.loss_weight * total_loss_mse + 1* total_loss_constraint + 1 * average_intensity_error
        return g_loss, total_loss_mse
    

    def train_step(self, real_images):
        real_images_combined, average_combined, spatial_means_combined, spatial_stds_combined, time_of_year_combined = self.process_real_images(real_images)
        batch_size = tf.shape(real_images_combined)[0]  # this should now be N_GCM times the average
        orog_vector = self.expand_conditional_inputs(self.orog, batch_size)
        
        
        if self.train_unet:
            with tf.GradientTape() as tape:
                random_latent_vectors = tf.random.normal(
                    shape=(batch_size,) + self.latent_dim[0]
                )
                losses = self.unet_pass(average_combined, orog_vector, real_images_combined,
                                        spatial_means_combined, spatial_stds_combined, time_of_year_combined)

                total_loss_unet = losses["loss_rain"]
            u_gradient = tape.gradient(total_loss_unet, self.unet.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.u_optimizer.apply_gradients(zip(u_gradient, self.unet.trainable_variables))
            config = losses
        
        
        
        if self.train_gan:
                # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size,) + self.latent_dim[0]
            )
            random_latent_vectors1 = tf.random.normal(
                shape=(batch_size,) + self.latent_dim[1]
            )

            predictions = self.unet([average_combined, orog_vector,time_of_year_combined, spatial_means_combined,
                             spatial_stds_combined], training=True)
            rain_unet = predictions
            generator_preds_historical = self.generator([random_latent_vectors, random_latent_vectors1,
                                                         average_combined, orog_vector, rain_unet,
                                                         time_of_year_combined, spatial_means_combined, 
                                                        spatial_stds_combined], training=True)
            # here we introduce a gan for each individual variable
            for i in range(self.d_steps):
                with tf.GradientTape() as tape:
                    fake_logits_historical = self.discriminator(
                        [generator_preds_historical+rain_unet, average_combined, orog_vector, rain_unet],
                        training=True)
                    # Get the logits for the real images
                    # modified this line to now predict the residuals of the solution

                    real_logits_historical = self.discriminator(
                        [real_images_combined[:, :, :, 0:1],average_combined, orog_vector, rain_unet],
                        training=True)
                    gp_hist = self.gradient_penalty(self.discriminator, batch_size,
                                           real_images_combined[:, :, :, 0:1],
                                           generator_preds_historical+rain_unet,
                                           average_combined, orog_vector, rain_unet)

                    d_cost_hist = self.d_loss_fn(real_img=real_logits_historical, fake_img=fake_logits_historical)
                    # Add the gradient penalty to the original discriminator loss
                    d_loss = d_cost_hist + gp_hist * self.gp_weight

                # Get the gradients w.r.t the discriminator loss
                d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
                # Update the weights of the discriminator using the discriminator optimizer
                self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

            with tf.GradientTape() as tape:
                """
                Introducing the Maximum and Average Penalty in the Loss function for each variable 
                """
                historical_loss, total_loss_mse = self.gan_pass(random_latent_vectors, random_latent_vectors1, average_combined, orog_vector,
                 real_images_combined,spatial_means_combined, spatial_stds_combined,
                 time_of_year_combined)
            total_loss = historical_loss
            # Get the gradients w.r.t the generator loss
            gen_gradient = tape.gradient(total_loss, self.generator.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

            config = {"d_loss": d_loss, "g_loss": total_loss, "unet_loss":total_loss_mse}

        return config





def gamma_loss(y_true, y_pred, eps=3e-1):
    occurence = y_pred[:, :, :, -1]
    y_true = y_true[:, :, :, 0]
    shape_param = K.exp(y_pred[:, :, :, 0])
    scale_param = K.exp(y_pred[:, :, :, 1])
    bool_rain = tf.cast(y_true > 0.01, 'float32')
    eps = tf.cast(eps, 'float32')
    loss1 = ((1 - bool_rain) * tf.math.log(1 - occurence + eps) + bool_rain * (
            K.log(occurence + eps) + (shape_param - 1) * K.log(y_true + eps) -
            shape_param * tf.math.log(scale_param + eps) - tf.math.lgamma(shape_param) - y_true / (
                    scale_param + eps)))
    # bool_rain = K.flatten(bool_rain)
    # occurence = K.flatten(occurence)
    output_loss = -1 * (K.mean(loss1))
    return output_loss


def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


def predict(model, x_test, y_test, batch_size=32, key='Rain_bc', pred_name='simple_dense', loss='gamma', thres=0.25):
    """
    This is a function that converts a prediction to a netcdf so that it can be plotted easily
    model: tensorflow model
    x_test: input data ( e.g.. (26, 23, 5) where 26 pixels in the latitude, 23 in longitude and 5 channels)
    y_test: y_test data, please note that this should be a netcdf! not a numpy array
    loss: "gamma" or mse
    """
    data = y_test.to_dataset()
    preds = model.predict(x_test, verbose=1, batch_size=batch_size)
    if loss == "gamma":
        scale = np.exp(preds[:, :, :, 0])
        shape = np.exp(preds[:, :, :, 1])
        prob = preds[:, :, :, -1]
        rainfall = (prob > thres) * scale * shape
    else:
        rainfall = preds
    data[key].values = rainfall
    return data.rename({key: pred_name})


class GeneratorCheckpoint(Callback):
    def __init__(self, generator, filepath, period):
        super().__init__()
        self.generator = generator
        self.filepath = filepath
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            self.generator.save(f"{self.filepath}_epoch_{epoch + 1}.h5")


class DiscriminatorCheckpoint(Callback):
    def __init__(self, discriminator, filepath, period):
        super().__init__()
        self.discriminator = discriminator
        self.filepath = filepath
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            self.discriminator.save(f"{self.filepath}_epoch_{epoch + 1}.h5")


# changed activation function to hyperbolic tangent


def prepare_training_data(config, X, y, means, stds):
    """
    Normalizes the X training data, and stacks the features into a single dimension
    config: json file that contains a dictionary of the experimental files used in training
    X: training data, which is pre-loaded. Note this file is already in the config file, but has been loaded in another script
    mean:: normalize relative to a mean
    std: normalize relative to an std
    """

    list_of_vars = config["var_names"]
    # normalize data
    X_norm = (X[list_of_vars] - means.mean(["lat","lon"])) / stds.mean(["lat","lon"])

    stacked_X = xr.concat([X_norm[varname] for varname in list_of_vars], dim="channel")
    # stack features
    stacked_X['channel'] = (('channel'), list_of_vars)
    stacked_X = stacked_X  # .transpose("time", "lat", "lon", "channel")
    times = stacked_X.time.to_index().intersection(y.time.to_index())
    # this part should be fine
    stacked_X = stacked_X.sel(time=times)
    y = y.sel(time=times)
    return stacked_X, y


def prepare_static_fields(config):
    topography_data = xr.open_dataset(config["static_predictors"])
    vegt = topography_data.vegt
    orog = topography_data.orog
    he = topography_data.he
    print(orog.max(), he.max(), vegt.max())

    # normazation to the range [0,1]
    vegt = (vegt - vegt.min()) / (vegt.max() - vegt.min())
    orog = (orog - orog.min()) / (orog.max() - orog.min())
    he = (he - he.min()) / (he.max() - he.min())
    return vegt, orog, he


# LOADING the mean values
def preprocess_input_data(config):
    vegt, orog, he = prepare_static_fields(config)
    means = xr.open_dataset(config["mean"])
    stds = xr.open_dataset(config["std"])

    X = xr.open_dataset(config["train_x"])  # .sel(time = slice("2016", None))
    X['time'] = pd.to_datetime(X.time.dt.strftime("%Y-%m-%d"))

    y = xr.open_dataset(config["train_y"], chunks={"time": 5000})
    y['time'] = pd.to_datetime(y.time.dt.strftime("%Y-%m-%d"))  # .sel(time = slice("2016", None))
    try:
        y = y.drop("lat_bnds")
        y = y.drop("lon_bnds")
        y = y.drop("time_bnds")

    except:
        pass
    # preare the training data
    stacked_X, y = prepare_training_data(config, X, y, means, stds)

    return stacked_X, y, vegt, orog, he



def prepare_training_data_v2(config, X, y, means, stds):
    """
    Normalizes the X training data, and stacks the features into a single dimension
    config: json file that contains a dictionary of the experimental files used in training
    X: training data, which is pre-loaded. Note this file is already in the config file, but has been loaded in another script
    mean:: normalize relative to a mean
    std: normalize relative to an std
    """

    list_of_vars = config["var_names"]
    # normalize data
    spatial_means = X[list_of_vars].mean(["lat","lon"])
    spatial_stds = X[list_of_vars].std(["lat","lon"])
    X_norm = (X[list_of_vars] - spatial_means) / spatial_stds
    stacked_X = xr.concat([X_norm[varname] for varname in list_of_vars], dim="channel")
    means_means = xr.open_dataset(config["input_means_means"])
    means_stds = xr.open_dataset(config["input_means_stds"])
    spatial_means = (spatial_means - means_means)/means_stds
    
    spatial_means = xr.concat([spatial_means[i] for i in list_of_vars], dim ="channel")
    # stds
    stds_means = xr.open_dataset(config["input_stds_means"])
    stds_stds = xr.open_dataset(config["input_stds_stds"])
    spatial_stds = (spatial_stds - stds_means)/stds_stds
    spatial_stds = xr.concat([spatial_stds[i] for i in list_of_vars], dim ="channel")
    # stack features
    stacked_X['channel'] = (('channel'), list_of_vars)
    spatial_stds['channel'] = (('channel'), list_of_vars)
    spatial_means['channel'] = (('channel'), list_of_vars)
    stacked_X = stacked_X  # .transpose("time", "lat", "lon", "channel")
    times = stacked_X.time.to_index().intersection(y.time.to_index())
    # this part should be fine
    stacked_X = stacked_X.sel(time=times)
    y = y.sel(time=times)
    return stacked_X, y, spatial_means, spatial_stds

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic


def low_pass_filter(arr, cutoff=0.2, taper=False):
    """
    Apply a low-pass filter to the (lat, lon, channel) dimensions of an array.

    Parameters:
    - arr: Input array of shape (time, lat, lon, channel)
    - cutoff: Fraction of the highest frequencies to remove (0 to 0.5)
    - taper: If True, apply a smooth Gaussian taper instead of hard cutoff.

    Returns:
    - Filtered array in the spatial domain.
    """
    time_dim, lat_dim, lon_dim = arr.shape
    arr_fft = np.fft.fft2(arr, axes=(1,2))
    freq_lat = np.fft.fftfreq(lat_dim)
    freq_lon = np.fft.fftfreq(lon_dim)
    #freq_chan = np.fft.fftfreq(chan_dim)

    # Create a frequency mask
    freq_grid = np.sqrt(
        freq_lat[:, None]**2 + 
        freq_lon[None, :]**2  
        
    )


    filter_mask = (freq_grid < cutoff).astype(float)

    # Apply filter
    arr_fft_filtered = arr_fft * filter_mask[None, :, :]

    # Inverse transform
    arr_filtered = np.fft.ifft2(arr_fft_filtered, axes=(1,2)).real
    return arr_filtered


# LOADING the mean values
def preprocess_input_data_v2(config, spectral_filtering = False, spectral_threshold =0.48):
    vegt, orog, he = prepare_static_fields(config)
    means = xr.open_dataset(config["mean"])
    stds = xr.open_dataset(config["std"])

    X = xr.open_dataset(config["train_x"])  # .sel(time = slice("2016", None))
    X['time'] = pd.to_datetime(X.time.dt.strftime("%Y-%m-%d"))
    list_of_vars = config["var_names"]
    X = X.transpose("time","lat","lon", "GCM")
    
    if spectral_filtering:
        X_new = X[list_of_vars].copy() * np.nan
        for varname in list_of_vars:
            X_new[varname].data = low_pass_filter(X[varname].transpose("time","lat","lon", "GCM").values, cutoff = spectral_threshold)
            print(f"Spectrally modified variable: {varname}")
        X = X_new  

    y = xr.open_dataset(config["train_y"], chunks={"time": 5000})
    y['time'] = pd.to_datetime(y.time.dt.strftime("%Y-%m-%d"))  # .sel(time = slice("2016", None))
    try:
        y = y.drop("lat_bnds")
        y = y.drop("lon_bnds")
        y = y.drop("time_bnds")

    except:
        pass
    # preare the training data
    stacked_X, y, spatial_means, spatial_stds = prepare_training_data_v2(config, X, y, means, stds)

    return stacked_X, y, vegt, orog, he, spatial_means, spatial_stds


def prepare_training_data_v3(config, X, y, means, stds):
    """
    Normalizes the X training data, and stacks the features into a single dimension
    config: json file that contains a dictionary of the experimental files used in training
    X: training data, which is pre-loaded. Note this file is already in the config file, but has been loaded in another script
    mean:: normalize relative to a mean
    std: normalize relative to an std
    """

    list_of_vars = config["var_names"]
    # normalize data
    spatial_means = X[list_of_vars].mean(["lat","lon"])
    spatial_stds = X[list_of_vars].std(["lat","lon"])
    X_norm = (X[list_of_vars] - means.mean(["lat","lon"])) / stds.mean(["lat","lon"])
    stacked_X = xr.concat([X_norm[varname] for varname in list_of_vars], dim="channel")
    means_means = xr.open_dataset(config["input_means_means"])
    means_stds = xr.open_dataset(config["input_means_stds"])
    spatial_means = (spatial_means - means_means)/means_stds
    
    spatial_means = xr.concat([spatial_means[i] for i in list_of_vars], dim ="channel")
    # stds
    stds_means = xr.open_dataset(config["input_stds_means"])
    stds_stds = xr.open_dataset(config["input_stds_stds"])
    spatial_stds = (spatial_stds - stds_means)/stds_stds
    spatial_stds = xr.concat([spatial_stds[i] for i in list_of_vars], dim ="channel")
    # stack features
    stacked_X['channel'] = (('channel'), list_of_vars)
    spatial_stds['channel'] = (('channel'), list_of_vars)
    spatial_means['channel'] = (('channel'), list_of_vars)
    stacked_X = stacked_X  # .transpose("time", "lat", "lon", "channel")
    times = stacked_X.time.to_index().intersection(y.time.to_index())
    # this part should be fine
    stacked_X = stacked_X.sel(time=times)
    y = y.sel(time=times)
    return stacked_X, y, spatial_means, spatial_stds


# # LOADING the mean values
# def preprocess_input_data_v2(config):
#     vegt, orog, he = prepare_static_fields(config)
#     means = xr.open_dataset(config["mean"])
#     stds = xr.open_dataset(config["std"])

#     X = xr.open_dataset(config["train_x"])  # .sel(time = slice("2016", None))
#     X['time'] = pd.to_datetime(X.time.dt.strftime("%Y-%m-%d"))

#     y = xr.open_dataset(config["train_y"], chunks={"time": 5000})
#     y['time'] = pd.to_datetime(y.time.dt.strftime("%Y-%m-%d"))  # .sel(time = slice("2016", None))
#     try:
#         y = y.drop("lat_bnds")
#         y = y.drop("lon_bnds")
#         y = y.drop("time_bnds")

#     except:
#         pass
#     # preare the training data
#     stacked_X, y, spatial_means, spatial_stds = prepare_training_data_v2(config, X, y, means, stds)

#     return stacked_X, y, vegt, orog, he, spatial_means, spatial_stds


# LOADING the mean values
def preprocess_input_data_v3(config):
    vegt, orog, he = prepare_static_fields(config)
    means = xr.open_dataset(config["mean"])
    stds = xr.open_dataset(config["std"])

    X = xr.open_dataset(config["train_x"])  # .sel(time = slice("2016", None))
    X['time'] = pd.to_datetime(X.time.dt.strftime("%Y-%m-%d"))

    y = xr.open_dataset(config["train_y"], chunks={"time": 5000})
    y['time'] = pd.to_datetime(y.time.dt.strftime("%Y-%m-%d"))  # .sel(time = slice("2016", None))
    try:
        y = y.drop("lat_bnds")
        y = y.drop("lon_bnds")
        y = y.drop("time_bnds")

    except:
        pass
    # preare the training data
    stacked_X, y, spatial_means, spatial_stds = prepare_training_data_v3(config, X, y, means, stds)

    return stacked_X, y, vegt, orog, he, spatial_means, spatial_stds
