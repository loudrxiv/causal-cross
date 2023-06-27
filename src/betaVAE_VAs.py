from _utils import *

class BetaVAE(tf.keras.Model):
  def __init__(self, pwm, latent_dim = 1, input_dims=[None, 4], kernel_size=3, strides=2, prefix='vae'):
    super(BetaVAE, self).__init__()
    self.prefix = prefix
    self.latent_dim = latent_dim
    self.input_dims = input_dims
    self.kernel_size = kernel_size
    self.strides = strides
    self.output_dims = None
    self.pwm = pwm

    self.encoder = tf.keras.Sequential(
      layers = [

        tf.keras.layers.InputLayer(input_shape = self.input_dims),

        tf.keras.layers.Conv1D(
          filters = self.pwm.shape[2],
          kernel_size = self.pwm.shape[0],
          use_bias = False,
          trainable = False, # freeze the pwm
          padding = "same",
          name = "pwm_conv",
          input_shape = self.input_dims,
          weights = [self.pwm]),

        tf.keras.layers.Conv1D(
          filters = 64, 
          kernel_size = self.kernel_size,
          strides = self.strides, 
          activation = 'relu'),

        tf.keras.layers.GlobalMaxPooling1D(),
        #tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(2 * self.latent_dim)
      ],
      name = "encoder"
    )
  
    self.output_dims = compute_output_dims(
      input_dims=self.input_dims[-1],
      kernel_size=self.kernel_size,
      strides=self.strides
    )
    self.output_dims = compute_output_dims(
      input_dims=self.output_dims,
      kernel_size=self.kernel_size,
      strides=self.strides
    )

    self.decoder = tf.keras.Sequential(
      layers = [

        tf.keras.layers.Dense(units=tf.reduce_prod(self.output_dims) * 125 * 4, activation='relu'),

        tf.keras.layers.Reshape(target_shape=(125, 4)),

        tf.keras.layers.Conv1DTranspose(
          filters=64, kernel_size=self.kernel_size, strides=2, 
          padding='same', activation='relu'),

        tf.keras.layers.Conv1DTranspose(
          filters=32, kernel_size=self.kernel_size, strides=2,
          padding='same', activation='relu'),

        tf.keras.layers.Conv1DTranspose(
          filters=self.input_dims[-1], kernel_size=self.kernel_size, strides=1,
          padding='same')

    ],
      name = "decoder"
    )

  def elbo(self, batch, **kwargs):
    beta = kwargs['beta'] if 'beta' in kwargs else 1.0
    mean_z, logvar_z, z_sample, x_pred = self.forward(batch)
    
    logpx_z = compute_log_bernouli_pdf(x_pred, batch)
    
    logpx_z = tf.reduce_sum(logpx_z, axis=[1, 2])
    #logpx_z = tf.reduce_sum(logpx_z, axis=[1, 2, 3])
    
    kl_divergence = tf.reduce_sum(compute_kl_divergence_standard_prior(mean_z, logvar_z), axis=1)

    elbo = tf.reduce_mean(logpx_z - beta * kl_divergence)

    return elbo, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)

  def train_step(self, batch, optimizers, **kwargs):
    with tf.GradientTape() as tape:
      elbo, logpx_z, kl_divergence = self.elbo(batch, **kwargs)
      gradients = tape.gradient(-1 * elbo, self.trainable_variables)
      optimizers['primary'].apply_gradients(zip(gradients, self.trainable_variables))
        
      return elbo, logpx_z, kl_divergence

  def forward(self, batch, apply_sigmoid=False):
    mean_z, logvar_z = self.encode(batch)
    z_sample = self.reparameterize(mean_z, logvar_z)
    x_pred = self.decode(z_sample, apply_sigmoid=apply_sigmoid)
  
    return mean_z, logvar_z, z_sample, x_pred

  def encode(self, batch):
    
    mean_z, logvar_z = tf.split(self.encoder(batch), num_or_size_splits=2, axis=-1)
    #mean_z, logvar_z = tf.split(self.encoder(batch['x']), num_or_size_splits=2, axis=-1)

    return mean_z, logvar_z

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

  def generate(self, z=None, num_generated_images=15, **kwargs):
    if z is None:
      z = tf.random.normal(shape=(num_generated_images, self.latent_dim))
    return self.decode(z, apply_sigmoid=True)

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape) # each distribution has its own epsilon
    return eps * tf.exp(logvar * .5) + mean

  def average_kl_divergence(self, batch):
    mean_z, logvar_z = self.encode(batch)
    return tf.reduce_mean(compute_kl_divergence_standard_prior(mean_z, logvar_z), axis=0)