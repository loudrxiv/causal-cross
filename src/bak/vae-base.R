#" Defaults set here
source("_default_params.R")
source("_functions.R")

#" Generate PWSM
hocomoco_filters <- readRDS(HOCOMOCO) |> lapply(as.matrix)

filter_check(hocomoco_filters)

filter_tensor <- hocomoco_filters |> 
  unlist() |>  
  array(dim = c(nrow(hocomoco_filters[[1]]),4,length(hocomoco_filters)))

#" Tile sequences from genome
mt = GenomicRanges::tileGenome(BSgenome.Hsapiens.UCSC.hg38::Hsapiens |> 
                                 seqinfo()                           |> 
                                 keepStandardChromosomes(), tilewidth=16*1024, cut.last.tile.in.chrom = TRUE)

#" Format sequences for input
sqs        = Biostrings::getSeq(Hsapiens, mt) 
names(sqs) = mt |> as.character()
nNs        = Biostrings::vcountPattern('N',sqs)
sqs        = sqs[nNs == 0]
sqs        = sqs[width(sqs) == 16*1024]

#" Model creation
rows    = as.integer(dim(filter_tensor)[1])
cols    = as.integer(dim(filter_tensor)[2])
batches = as.integer(dim(filter_tensor)[3])

input    = layer_input(shape = list(NULL,4), name = "input")

conv1d_1 = layer_conv_1d(filters = batches, 
                         kernel_size = rows,
                         use_bias = FALSE,
                         name = "conv1d_1", 
                         trainable = FALSE, 
                         padding = "same",
                         input_shape = list(NULL,4))

conv = keras_model_sequential(name = "conv")

conv$add(conv1d_1)

#" Add PWSM weights to model
get_layer(conv, name="conv1d_1")$set_weights(weights = list(filter_tensor))

conv_input = conv(input)

#" Encoder
encoder = keras_model_sequential( name = "encoder")
encoder$add(layer_max_pooling_1d(pool_size = 4L)) #" 4bp resolution, 34 bp locality
encoder$add(layer_batch_normalization())
encoder$add(layer_conv_1d(filters = 256, kernel_size = 5, padding = "same", activation = tf$keras$activations$gelu))
encoder$add(layer_max_pooling_1d(pool_size = 2L)) #" 8 bp resolution, 40 bp locality
encoder$add(layer_batch_normalization())
encoder$add(layer_conv_1d(filters = 64, kernel_size = 5, padding = "same", activation = 'linear') )
encoder$add(layer_max_pooling_1d(pool_size = 2L)) #" 16 bp resolution, ?? bp locality
encoder$add(layer_batch_normalization())

#" Latent embedding
z = encoder(conv_input)

#" Note: this will need to be combined in other models (ilcc); you need a mean and variance for both nc and ns`
z_mean   = z |> layer_conv_1d(filters = 32,
                              kernel_size = 1L,
                              padding = "same",
                              use_bias = TRUE
                              )
z_logVar  = z |> layer_conv_1d(filters = 32,
                              kernel_size = 1L,
                              padding = "same",
                              use_bias = TRUE
                              )

embedding = keras_model(input, z_mean, name = "embedding")
 
b_dim    = tf$shape(z_mean)[1] #" batch dimension
x_dim    = tf$shape(z_mean)[2] #" sequence length
y_dim    = tf$shape(z_mean)[3] #" latent dimension

eps      = tf$random$normal(shape = c(b_dim, x_dim, y_dim), seed = 173649)
z_sample =  z_mean + tf$math$exp(0.5 * z_logVar) * eps

#" Decoder
decoder = keras_model_sequential(name = "decoder")
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters = 64, stride = 2, kernel_size = 8, padding='same', activation = tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters = 64, stride = 2, kernel_size = 8, padding='same', activation = tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters = 64,  stride = 2, kernel_size = 8, padding='same', activation = tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters = 64,  stride = 2, kernel_size = 8, padding='same', activation = tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_dropout(rate=0.01))
decoder$add(layer_conv_1d_transpose(filters = 64, stride = 1, kernel_size = 4,  padding='same', activation = tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_conv_1d_transpose(filters = 64, stride = 1, kernel_size = 2,  padding='same', activation = tf$keras$activations$gelu))
decoder$add(layer_batch_normalization())
decoder$add(layer_conv_1d_transpose(filters = 4,  kernel_size = 1,   padding='same', activation = 'sigmoid'))

reconstruction = decoder(z_sample)
vae_base = keras_model(input, reconstruction, name="vae_base")

#" KLD loss
vae_base$add_loss(kl_loss(z_mean, z_logVar) * 0.001)

vae_base = vae_base |> compile(loss="binary_crossentropy",
                       optimizer = keras$optimizers$Adamax(),
                       metrics    = list(tf$keras$metrics$BinaryAccuracy(name="acc"),
                                         tf$keras$metrics$AUC(name="auroc"),
                                         tf$keras$metrics$AUC(name="auprc", curve="PR")))

early_stopping = tf$keras$callbacks$EarlyStopping(monitor  = 'val_auprc',
                                                  patience =  7,
                                                  restore_best_weights = TRUE,
                                                  mode = 'max')
#" Shuffle data for better learning
set.seed(24213)
tst_size = length(sqs) %/% 5
val_size = (length(sqs) * 0.8) %/% 5
shuff    = sample(seq_len(sqs |> length()))

#" Split into train & test sets
ssqs     = sqs[shuff]
ds_test  = ssqs[1:tst_size]        |> make_ints()
ds_train = sqs[-(1:tst_size)]
ds_val   = ds_train[1:val_size]    |> make_ints()
ds_train = ds_train[-(1:val_size)] |> make_ints()

#" Utilize tfdatasets for efficiency
mds_train  =   tensor_slices_dataset(ds_train)       |>
  dataset_map(function(x) tf$one_hot(x,4L),
              num_parallel_calls = tf$data$AUTOTUNE) |>
  dataset_map(function(x) list(x,x),
              num_parallel_calls = tf$data$AUTOTUNE) |>
  dataset_shuffle( buffer_size = 10000L,
                   reshuffle_each_iteration=TRUE)    |>
  dataset_batch(64, drop_remainder = TRUE)           |>
  dataset_repeat()                                   |>
  dataset_prefetch_to_device(device = '/gpu:0',
                             buffer_size = tf$data$AUTOTUNE)

mds_val  =     tensor_slices_dataset(ds_val)         |>
  dataset_map(function(x) tf$one_hot(x,4L),
              num_parallel_calls = tf$data$AUTOTUNE) |>
  dataset_map(function(x) list(x,x),
              num_parallel_calls = tf$data$AUTOTUNE) |>
  dataset_shuffle( buffer_size = 10000L,
                   reshuffle_each_iteration=TRUE)    |>
  dataset_batch(64, drop_remainder = TRUE)           |>
  dataset_repeat()                                   |>
  dataset_prefetch_to_device(device = '/gpu:0',
                             buffer_size = tf$data$AUTOTUNE)

history <- vae_base |> fit(mds_train,
                           epochs=base_params$epochs,
                           steps_per_epoch = 50,
                           class_weights = c("0"=1/3, "1"=1),
                           validation_data=mds_val,
                           validation_steps=10,
                           callbacks = list(early_stopping))

#" save models
save_model_tf(vae_base,  filepath = paste0(MODEL_ROOT, "vae_base"))
save_model_tf(embedding, filepath = paste0(MODEL_ROOT, "embedding"))
save_model_tf(decoder,   filepath = paste0(MODEL_ROOT, "decoder"))

#" plot the loss
plot_figure("loss", data.frame(history$metrics$loss, history$metrics$val_loss))
plot_figure("acc", data.frame(history$metrics$acc, history$metrics$val_acc))
plot_figure("auprc", data.frame(history$metrics$auprc, history$metrics$val_auprc))
plot_figure("auroc", data.frame(history$metrics$auroc, history$metrics$val_auroc))
