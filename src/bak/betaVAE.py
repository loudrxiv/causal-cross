import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import InputLayer, Conv1D, Conv2D, Flatten, Dense, Reshape, Conv1DTranspose, Conv2DTranspose, Flatten, Reshape, Input

class Reparameterize(tf.keras.layers.Layer):
    """
    Custom layer.
     
    Reparameterization trick, sample random latent vectors Z from 
    the latent Gaussian distribution which has the following parameters 

    mean = Z_mu
    std = exp(0.5 * Z_logvar)
    """
    def call(self, inputs):
        Z_mu, Z_logvar = inputs
        epsilon = tf.random.normal(tf.shape(Z_mu))
        sigma = tf.math.exp(0.5 * Z_logvar)
        return Z_mu + sigma * epsilon


class BetaVAE:
    def __init__(self, pwm, input_shape, latent_dim=32, loss_type="mse", learning_rate=0.0005):
        self.latent_dim = latent_dim
        self.C = 0
        self.gamma = 100
        self.pwm = pwm

        # create encoder
        encoder_input = Input(shape=input_shape)

        # Convolve with PWM
        X = Conv1D(filters = self.pwm.shape[2],
                   kernel_size = self.pwm.shape[0],
                   use_bias = False,
                   trainable = False,
                   padding = "same",
                   input_shape = [None, 4])(encoder_input) #,
                   #weights = self.pwm)(encoder_input)
        
        X = Conv1D(32, 4, strides=2, padding="same", activation="relu", name=None)(X)
        X = Conv1D(32, 4, strides=2, padding="same", activation="relu", name=None)(X)
        X = Conv1D(32, 4, strides=2, padding="same", activation="relu", name=None)(X)
        X = Conv1D(32, 4, strides=2, padding="same", activation="relu", name=None)(X)
        X = Flatten()(X)
        X = Dense(256, activation="relu")(X)
        X = Dense(256,  activation="relu")(X)

        Z_mu = Dense(self.latent_dim)(X)
        Z_logvar = Dense(self.latent_dim, activation="relu")(X)
        Z = Reparameterize()([Z_mu, Z_logvar])

        # create decoder
        output_activation = None
        decoder_input = Input(shape=(self.latent_dim,))
        X = Dense(256,  activation="relu")(decoder_input)
        X = Dense(256,  activation="relu")(X)
        X = Dense(512,  activation="relu")(X)
        X = Reshape((4, 128))(X)
        X = Conv1DTranspose(32, 4, strides=2, padding="same", activation="relu", name=None)(X)
        X = Conv1DTranspose(32, 4, strides=2, padding="same", activation="relu", name=None)(X)
        X = Conv1DTranspose(32, 4, strides=2, padding="same", activation="relu", name=None)(X)
        X = Flatten()(X)
        X = Dense(250*4,  activation="relu")(X)
        X = Reshape((250, 4))(X)
        decoder_output = Conv1DTranspose(4, 4, strides=2, padding="same", activation=output_activation)(X)

        # create models
        self.encoder = Model(encoder_input, [Z_mu, Z_logvar, Z])
        self.decoder = Model(decoder_input, decoder_output)
        self.vae = Model(encoder_input, self.decoder(Z))
        self.vae.compile(optimizer='adam', loss=loss, metrics=[reconstruction_loss, kl_divergence])


# define vae losses
def reconstruction_loss(X, X_pred):
    if loss_type == "bce":
        bce = tf.losses.BinaryCrossentropy() 
        return bce(X, X_pred) * np.prod(input_shape)
    elif loss_type == "mse":
        mse = tf.losses.MeanSquaredError()
        return mse(X, X_pred) * np.prod(input_shape)
    else:
        raise ValueError("Unknown reconstruction loss type. Try 'bce' or 'mse'")

def kl_divergence(X, X_pred):
    self.C += (1/1440) # TODO use correct scalar
    self.C = min(self.C, 35) # TODO make variable
    kl = -0.5 * tf.reduce_mean(1 + Z_logvar - Z_mu**2 - tf.math.exp(Z_logvar))
    return self.gamma * tf.math.abs(kl - self.C)

def loss(X, X_pred):
    return reconstruction_loss(X, X_pred) + kl_divergence(X, X_pred)