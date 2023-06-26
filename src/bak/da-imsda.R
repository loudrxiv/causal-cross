source("seqadapt/_utils.R")

# load filters to size the input correctly
pwm <- gen_datamotifs()

batches <- dim(pwm)[3] 
rows    <- dim(pwm)[1]
columns <- dim(pwm)[2]

#---- Instantiate the model

#tf$keras$mixed_precision$set_global_policy("mixed_float16") # to save on memory
tf$keras$mixed_precision$set_global_policy("float32")

X_input = layer_input(shape = list(NULL, 4), name = "X_input")
D_input = layer_input(shape = list(NULL, 1), name = "D_input")

# MODEL ONE -------------------------------------------------------------------------------------------------------

# D_input |> D_prior
D_prior = keras_model_sequential(name = "D_prior")
D_prior$add(layer_dense(units=128, activation=tf$keras$activations$gelu))
D_prior$add(layer_dense(units=64, activation=tf$keras$activations$gelu))
D_prior$add(layer_dense(units=2, activation="linear"))
D_prior$add(layer_reshape(target_shape=list(2,1)))

p_ngD = D_input |> D_prior()

# X_input |> encoder_conv
encoder_conv = keras_model_sequential(name = "encoder_conv")
encoder_conv$add(layer_conv_1d(filters=batches, kernel_size=rows, use_bias=FALSE,
                               trainable=FALSE, padding="same", input_shape=list(NULL, 4),
                               weights=list(pwm)))
encoder_conv$add(layer_max_pooling_1d(pool_size=4)) #" 4bp resolution, 34 bp locality
encoder_conv$add(layer_batch_normalization())
encoder_conv$add(layer_conv_1d(filters=128, kernel_size=5, padding = "same", activation=tf$keras$activations$gelu))
encoder_conv$add(layer_max_pooling_1d(pool_size=5)) #" 8 bp resolution, 40 bp locality
encoder_conv$add(layer_batch_normalization())
encoder_conv$add(layer_conv_1d(filters=64, kernel_size=5, padding = "same", activation=tf$keras$activations$gelu))
encoder_conv$add(layer_max_pooling_1d(pool_size=5)) #" 8 bp resolution, 40 bp locality
encoder_conv$add(layer_batch_normalization())

# encoder_conv |> encoder_nc
encoder_nc = keras_model_sequential(name = "encoder_nc")
encoder_nc$add(layer_conv_1d(filters=1, kernel_size=5, padding = "same", activation="linear"))
encoder_nc$add(layer_max_pooling_1d(pool_size=5)) #" 8 bp resolution, 40 bp locality
encoder_nc$add(layer_batch_normalization())

# encoder_conv |> encoder_ns
encoder_ns = keras_model_sequential(name = "encoder_ns")
encoder_ns$add(layer_conv_1d(filters=1, kernel_size=5, padding = "same", activation="linear"))
encoder_ns$add(layer_max_pooling_1d(pool_size=5)) #" 8 bp resolution, 40 bp locality
encoder_ns$add(layer_batch_normalization())

# Get the two outputs from the encoder (nc and ns) and concat the domain index
q_ncgDX = k_concatenate(list((X_input |> encoder_conv() |> encoder_nc()), D_input), axis=2L)
q_nsgDX = k_concatenate(list((X_input |> encoder_conv() |> encoder_ns()), D_input), axis=2L)
q_ngDX  = tf$multiply(q_ncgDX, q_nsgDX)

# Opt to learn the mean and the logvar with conv_1d layers, we keep the dimension consistent here
ncsrc_mean   = q_ncgDX |> layer_conv_1d(filters=1, kernel_size=1, padding="same", use_bias=TRUE)
ncsrc_logvar = q_ncgDX |> layer_conv_1d(filters=1, kernel_size=1, padding="same", use_bias=TRUE)
nctar_mean   = q_ncgDX |> layer_conv_1d(filters=1, kernel_size=1, padding="same", use_bias=TRUE)           
nctar_logvar = q_ncgDX |> layer_conv_1d(filters=1, kernel_size=1, padding="same", use_bias=TRUE)
ns_mean      = q_nsgDX |> layer_conv_1d(filters=1, kernel_size=1, padding="same", use_bias=TRUE)                                                                                                                                                                                                         
ns_logvar    = q_nsgDX |> layer_conv_1d(filters=1, kernel_size=1, padding="same", use_bias=TRUE)

# Actually sample 
eps           = k_random_normal(shape = list(tf$shape(ns_mean)[1], tf$shape(ns_mean)[2], tf$shape(ns_mean)[3]))
nc_sample_src = ncsrc_mean + tf$math$exp(0.5 * ncsrc_logvar) * k_cast_to_floatx(eps)
nc_sample_tar = nctar_mean + tf$math$exp(0.5 * nctar_logvar) * k_cast_to_floatx(eps)
ns_sample     = ns_mean    + tf$math$exp(0.5 * ns_logvar)    * k_cast_to_floatx(eps)

n_sample      = k_concatenate(c(nc_sample_src, nc_sample_tar, ns_sample))

# n_sample |> decoder
decoder = keras_model_sequential(name = "decoder")
decoder$add(layer_input(shape=list(2, 3), name="n_sample"))
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters=256, kernel_size=4, strides=2, padding='same', activation=tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters=128, kernel_size=4, strides=2, padding='same', activation=tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters=64, kernel_size=4, strides=2, padding='same', activation=tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters=32, kernel_size=4, strides=2, padding='same', activation=tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters=16, kernel_size=4, strides=2, padding='same', activation=tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters=4, kernel_size=4, strides=2, padding='same', activation=tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters=4, kernel_size=4, strides=4, padding='same', activation=tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_flatten(input_shape=list(512, 4)))
decoder$add(layer_dense(units=500*4))
decoder$add(layer_reshape(target_shape=list(500, 4)))
decoder$add(layer_conv_1d_transpose(filters = 4, kernel_size=1, padding='same', activation='sigmoid'))

reconstruction = n_sample |> decoder()

# Define vae model
vae_ilcc = keras_model(inputs=list(X_input, D_input), outputs=reconstruction, name="vae_ilcc")
#vae_ilcc$add_loss(vae_loss(kl_loss(q_ngDX, p_ngD), recon_loss(X_input, reconstruction)))

# MODEL ONE -------------------------------------------------------------------------------------------------------

# MODEL TWO -------------------------------------------------------------------------------------------------------

transfer_block = keras_model_sequential(name = "transfer_block")
transfer_block$add(layer_max_pooling_1d(pool_size=2))
transfer_block$add(layer_dense(units=64, activation="relu"))
transfer_block$add(layer_dense(units=32, activation="relu"))
transfer_block$add(layer_dense(units=1 , activation="relu"))

z_src <- nc_sample_src |> transfer_block()

# z_mean_src   = z_src |> layer_conv_1d(filters = 32,
#                               kernel_size = 1L,
#                               padding = "same",
#                               use_bias = TRUE)
# z_logVar_src  = z_src |> layer_conv_1d(filters = 32,
#                               kernel_size = 1L,
#                               padding = "same",
#                               use_bias = TRUE)
# eps           = tf$random$normal(shape = c(tf$shape(z_mean_src)[1], tf$shape(z_mean_src)[2], tf$shape(z_mean_src)[3]), seed = 173649)
# z_sample_src  =  z_mean_src + tf$math$exp(0.5 * z_logVar_src) * tf$cast(eps, dtype=tf$float16) 

classify_block =  keras_model_sequential(name = 'classify_block')
classify_block$add(layer_dense(units=1, activation = 'sigmoid'))

y_pred = z_src |> classify_block()

da_ilcc = keras_model(inputs=list(X_input, D_input), outputs=list(reconstruction, y_pred), name="da_ilcc")
#da_ilcc$add_loss(total_loss(kl_loss(q_ngDX, p_ngD), recon_loss(X_input, reconstruction), mi_loss(z_src, y_pred)))
da_ilcc$add_loss(vae_loss(kl_loss(q_ngDX, p_ngD), recon_loss(X_input, reconstruction)))

# MODEL TWO -------------------------------------------------------------------------------------------------------

# Save Model
save_model_tf(da_ilcc, "/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/models/da_ilcc/da_ilcc")
save_model_hdf5(da_ilcc, "/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/models/da_ilcc/da_ilcc.h5")