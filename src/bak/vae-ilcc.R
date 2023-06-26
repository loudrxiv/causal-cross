source("_default_params.R")
source("_functions.R")

BATCH_SIZE = 400L
VALBATCH_SIZE = 10000L

# DATA ---------------------------------------------------------------

# PWSM -----
hocomoco_filters <- readRDS(HOCOMOCO) |> lapply(as.matrix)
filter_check(hocomoco_filters)
filter_tensor <- hocomoco_filters |> unlist() |> array(dim = c(nrow(hocomoco_filters[[1]]),4,length(hocomoco_filters)))

rows    = as.integer(dim(filter_tensor)[1])
cols    = as.integer(dim(filter_tensor)[2])
batches = as.integer(dim(filter_tensor)[3])
# PWSM -----

# SEQ -----
X_gr_src = GRangesList("src_bindingtrainpos"=import(src_bindingtrainposfile),
                       "src_bindingtrainneg"=import(src_bindingtrainnegfile),
                       "src_bindingtest"=import(sourcetestfile),
                       "src_bindingval"=import(sourcevalfile))

X_gr_tar = GRangesList("tar_bindingtrainpos"=import(tar_bindingtrainposfile),
                       "tar_bindingtrainneg"=import(tar_bindingtrainnegfile),
                       "tar_bindingtest"=import(targettestfile),
                       "tar_bindingval"=import(targetvalfile))

X_data_src      = lapply(X_gr_src, function(x) Biostrings::getSeq(Mmusculus, x))
X_labels_src    = lapply(X_gr_src, function(x) mcols(x)$name)
D_labels_src    = lapply(X_gr_src, function(x) rep(1, length(x)))
X_data_tar      = lapply(X_gr_tar, function(x) Biostrings::getSeq(Hsapiens, x))
X_labels_tar    = lapply(X_gr_tar, function(x) mcols(x)$name)
D_labels_tar    = lapply(X_gr_tar, function(x) rep(0, length(x)))
# SEQ -----
                         
# TRAIN -----
X_train = c(X_data_src$src_bindingtrainpos,
            X_data_src$src_bindingtrainneg,
            X_data_tar$tar_bindingtrainpos,
            X_data_tar$tar_bindingtrainneg)
y_trainbinding  = c(X_labels_src$src_bindingtrainpos, 
                    X_labels_src$src_bindingtrainneg,
                    X_labels_tar$tar_bindingtrainpos,
                    X_labels_tar$tar_bindingtrainneg)
y_traindomain   = c(D_labels_src$src_bindingtrainpos,
                    D_labels_src$src_bindingtrainneg,
                    D_labels_tar$tar_bindingtrainpos,
                    D_labels_tar$tar_bindingtrainneg)

nNs       = Biostrings::vcountPattern('N', X_train)
xwidth    = width(X_train)

X_train        = X_train[xwidth==500][nNs==0] |> make_ints()
y_trainbinding = y_trainbinding[xwidth==500][nNs==0]
y_traindomain  = y_traindomain[xwidth==500][nNs==0]
                         
train_list = list("input"=list("X_input"=X_train, 
                               "D_input"=tf$expand_dims(tf$constant(y_traindomain), axis=1L)),
                  "y_pred"=tf$cast(tf$expand_dims(tf$constant(as.integer(y_trainbinding), dtype=tf$int32), axis=1L), tf$float32))

mds_train = tensor_slices_dataset(train_list) |>
dataset_map(function(x) list(list("X_input"=tf$one_hot(x$input$X_input, 4L), "D_input"=x$input$D_input), tf$one_hot(x$input$X_input, 4L)), num_parallel_calls = tf$data$AUTOTUNE)|>
dataset_shuffle( buffer_size = 10000L, reshuffle_each_iteration=TRUE) |>
dataset_batch(BATCH_SIZE, drop_remainder = TRUE) |>
dataset_repeat() |>
dataset_prefetch_to_device(device = '/gpu:0', buffer_size = tf$data$AUTOTUNE)
# TRAIN -----

# VAL -----
X_val = c(X_data_src$src_bindingval,
          X_data_tar$tar_bindingval)

y_valbinding  = c(X_labels_src$src_bindingval, 
                  X_labels_tar$tar_bindingval)

y_valdomain   = c(D_labels_src$src_bindingval,
                  D_labels_tar$tar_bindingval)

nNs    = Biostrings::vcountPattern('N',X_val)
xwidth = width(X_val)

X_val = tf$constant(X_val[width(X_val)==500][nNs==0] |> make_ints())
y_valbinding = tf$constant(as.integer(y_valbinding[xwidth==500][nNs==0]))
y_valdomain  = tf$constant(as.integer(y_valdomain[xwidth==500][nNs==0]))
            
val_list = list("input"=list("X_input"=X_val, 
                               "D_input"=tf$expand_dims(tf$constant(y_valdomain), axis=1L)),
                  "y_pred"=tf$cast(tf$expand_dims(tf$constant(as.integer(y_valbinding), dtype=tf$int32), axis=1L), tf$float32))

mds_val = tensor_slices_dataset(val_list) |>
dataset_map(function(x) list(list("X_input"=tf$one_hot(x$input$X_input, 4L), "D_input"=x$input$D_input), tf$one_hot(x$input$X_input, 4L)), num_parallel_calls = tf$data$AUTOTUNE)|>
dataset_shuffle( buffer_size = 10000L, reshuffle_each_iteration=TRUE) |>
dataset_batch(VALBATCH_SIZE, drop_remainder = TRUE) |>
dataset_repeat() |>
dataset_prefetch_to_device(device = '/gpu:0', buffer_size = tf$data$AUTOTUNE)
# VAL -----
            
# ARCHITECTURE -----
# MODEL 1 ------------------------------------------------------
X_input = layer_input(shape = list(NULL, 4L), name = "X_input")
D_input = layer_input(shape = list(NULL, 1L), name = "D_input")

# Domain Prior (D_input) -----
domain_prior = keras_model_sequential(name = "domain_prior")
domain_prior$add(layer_dense(units = 64L, activation = tf$keras$activations$gelu))
domain_prior$add(layer_dense(units = 32L, activation = tf$keras$activations$gelu))
domain_prior$add(layer_dense(units = (32L*64L), activation="linear"))
domain_prior$add(layer_reshape(target_shape = c(32L, 64L)))

# Domain Concatenation -----
domain_concat = keras_model_sequential(name = "domain_concat")
domain_concat$add(layer_dense(units = 64L, activation = tf$keras$activations$gelu))
domain_concat$add(layer_dense(units = (1L*64L)))
domain_concat$add(layer_reshape(target_shape = c(1L,64L)))

# Convolutional layer with PWM -----
conv = keras_model_sequential(name = "conv")
conv1d = layer_conv_1d(filters = batches, 
                       kernel_size = rows,
                       use_bias = FALSE, 
                       trainable = FALSE, 
                       padding = "same",
                       name = "conv1d",
                       input_shape = list(NULL,4L))
conv$add(conv1d)
get_layer(conv, name="conv1d")$set_weights(weights = list(filter_tensor))

# Encoder (X_input) -----
encoder = keras_model_sequential( name = "encoder")
encoder$add(layer_max_pooling_1d(pool_size = 4L)) #" 4bp resolution, 34 bp locality
encoder$add(layer_batch_normalization())
encoder$add(layer_conv_1d(filters = 256L, kernel_size = 5L, padding = "same", activation = tf$keras$activations$gelu))
encoder$add(layer_max_pooling_1d(pool_size = 2L)) #" 8 bp resolution, 40 bp locality
encoder$add(layer_batch_normalization())
encoder$add(layer_conv_1d(filters = 64L, kernel_size = 5L, padding = "same", activation = 'linear') )
encoder$add(layer_max_pooling_1d(pool_size = 2L)) #" 16 bp resolution, ?? bp locality
encoder$add(layer_batch_normalization())

# Compare the learned Domain Prior with the feature embedding of X (with concat'ed D) -----

q_ngX    = X_input |> conv() |> encoder()
p_ngD    = D_input |> domain_prior()
X_concat = D_input |> domain_concat()
q_ngDX   = tf$concat(c(X_concat, q_ngX), axis = 1L)

kl_ilcc = keras_model(inputs=c(X_input, D_input), outputs=c(q_ngDX, p_ngD), name = "kl_ilcc")
kl_ilcc$add_loss(kl_loss(kl_ilcc$outputs[[1]], kl_ilcc$outputs[[2]]) * 0.001)
# MODEL 1 ------------------------------------------------------

# MODEL 2 ------------------------------------------------------
# NcNs_embedding -----
NcNs_embedding = keras_model_sequential(name = "NcNs_embedding")
NcNs_embedding$add(layer_flatten(input_shape = c(32L, 64L)))
NcNs_embedding$add(layer_dense(units = 256L, activation = tf$keras$activations$gelu))
NcNs_embedding$add(layer_dense(units = 128L, activation = tf$keras$activations$gelu))
NcNs_embedding$add(layer_dense(units = (32L*64L), activation="linear"))
NcNs_embedding$add(layer_reshape(target_shape = c(32L, 64L)))

# Get the general and specific noise terms from n_gDX -----

# Learn n_c and n_s
q_ncgDX = NcNs_embedding(kl_ilcc$outputs[[1]])
q_nsgDX = NcNs_embedding(kl_ilcc$outputs[[1]])

# Then sample from n_c and n_s
nc_mean_src   = q_ncgDX |> layer_conv_1d(filters = 32,                                                                                                                                                                                                   
                              kernel_size = 1L,                                                                                                                                                                                               
                              padding = "same",                                                                                                                                                                                               
                              use_bias = TRUE)                
nc_logVar_src = q_ncgDX |> layer_conv_1d(filters = 32,                                                                                                                                                                                                  
                              kernel_size = 1L,                                                                                                                                                                                               
                              padding = "same",                                                                                                                                                                                               
                              use_bias = TRUE)
nc_mean_tar   = q_ncgDX |> layer_conv_1d(filters = 32,                                                                                                                                                                                                   
                              kernel_size = 1L,                                                                                                                                                                                               
                              padding = "same",                                                                                                                                                                                               
                              use_bias = TRUE)                
nc_logVar_tar = q_ncgDX |> layer_conv_1d(filters = 32,                                                                                                                                                                                                  
                              kernel_size = 1L,                                                                                                                                                                                               
                              padding = "same",                                                                                                                                                                                               
                              use_bias = TRUE)
ns_mean   = q_nsgDX |> layer_conv_1d(filters = 32,                                                                                                                                                                                                   
                              kernel_size = 1L,                                                                                                                                                                                               
                              padding = "same",                                                                                                                                                                                               
                              use_bias = TRUE)                                                                                                                                                                                                               
ns_logVar = q_nsgDX |> layer_conv_1d(filters = 32,                                                                                                                                                                                                  
                              kernel_size = 1L,                                                                                                                                                                                               
                              padding = "same",                                                                                                                                                                                               
                              use_bias = TRUE)

b_dim    = tf$shape(ns_mean)[1] # batch dimension
x_dim    = tf$shape(ns_mean)[2] # sequence length
y_dim    = tf$shape(ns_mean)[3] # latent dimension

eps           = tf$random$normal(shape = c(b_dim, x_dim, y_dim), seed = 173649)                                                                                                                                                                 
nc_sample_src = nc_mean_src + tf$math$exp(0.5 * nc_logVar_src) * eps  
nc_sample_tar = nc_mean_tar + tf$math$exp(0.5 * nc_logVar_tar) * eps
ns_sample     = ns_mean + tf$math$exp(0.5 * ns_logVar) * eps
n_sample      = tf$concat(c(nc_sample_src, nc_sample_tar, ns_sample), axis=2L)

# Decoder -----
decoder = keras_model_sequential(name = "decoder")
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters = 64L, stride = 2L, kernel_size = 8L, padding='same', activation = tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters = 64L, stride = 2L, kernel_size = 8L, padding='same', activation = tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters = 64L,  stride = 2L, kernel_size = 8L, padding='same', activation = tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters = 64L,  stride = 2L, kernel_size = 8L, padding='same', activation = tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters = 64L, stride = 1L, kernel_size = 4L,  padding='same', activation = tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_conv_1d_transpose(filters = 64L, stride = 1L, kernel_size = 2L,  padding='same', activation = tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_flatten(input_shape = c(512L, 4L)))
decoder$add(layer_dense(units = (500L*4L), activation = tf$keras$activations$gelu))
decoder$add(layer_reshape(target_shape = c(500L, 4L)))
decoder$add(layer_conv_1d_transpose(filters = 4L,  kernel_size = 1L,   padding='same', activation = 'sigmoid'))

# Run through decoder and get reconstruction -----
reconstruction = n_sample |> decoder()

vae_ilcc = keras_model(inputs=list(kl_ilcc$input[[1]], kl_ilcc$input[[2]]), outputs=reconstruction, name="vae_ilcc")
vae_ilcc$add_loss(recon_loss(vae_ilcc$input[[1]], vae_ilcc$outputs[[1]]))
# MODEL 2 ------------------------------------------------------

# MODEL 3 ------------------------------------------------------

# Transfer Block -----
transfer_block = keras_model_sequential( name = "transfer_block")
transfer_block$add(layer_flatten(input_shape = c(32L, 32L)))
transfer_block$add(layer_dense(units = 256L, activation = tf$keras$activations$gelu))
transfer_block$add(layer_dense(units = 128L, activation = tf$keras$activations$gelu))
transfer_block$add(layer_dense(units = 64L , activation = tf$keras$activations$gelu))
transfer_block$add(layer_dense(units = 32L , activation = tf$keras$activations$gelu))
transfer_block$add(layer_dense(units = 16L , activation = tf$keras$activations$gelu))
transfer_block$add(layer_dense(units = 1L , activation = 'sigmoid'))
transfer_block$add(layer_reshape(target_shape = list(1L)))

y_pred = nc_sample_src |> transfer_block()

# Classifer Model -----
classifier = keras_model(inputs=list(kl_ilcc$input[[1]], kl_ilcc$input[[2]]), outputs=y_pred, name="classifier")
classifier$add_loss(mututal_loss(classifier$outputs[[1]]) * 1e-8)

# Whole Model -----
da_ilcc = keras_model(inputs=c(X_input, D_input), outputs=c(vae_ilcc$outputs[[1]], classifier$outputs[[1]]), name="da_ilcc")

# MODEL 3 ------------------------------------------------------
# ARCHITECUTRE -----
            
            
# MODEL TRAINING -----
vae_ilcc = vae_ilcc |> compile(loss=almost_total_loss,
                             optimizer=keras$optimizers$Adamax(learning_rate=0.001))

model_checkpoints=callback_model_checkpoint(VAE_ILCC_ROOT, save_best_only = FALSE, save_weights_only = FALSE, save_freq = "epoch") 
early_stopping=tf$keras$callbacks$EarlyStopping(monitor="val_loss", patience = 7, restore_best_weights = TRUE, mode = "max")
naan_terminate=callback_terminate_on_naan()

history <- vae_ilcc |> fit(mds_train,
                           epochs=10,
                           steps_per_epoch=(dim(X_train)[1] / BATCH_SIZE),
                           validation_data=mds_val,
                           validation_steps=(dim(X_val)[1] / VALBATCH_SIZE),
                           callbacks=c(model_checkpoints, early_stopping, naan_terminate))

saveRDS(history, file=paste0(MODEL_ROOT, "vae_ilcc/history/history_", format(Sys.time(), "%Y-%m-%d-%H:%M"), ".rds"))
# MODEL TRAINING -----
