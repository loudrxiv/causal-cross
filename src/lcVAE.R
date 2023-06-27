source("_utils.R")

# Can we see the GPU?
if( length(tf$config$list_physical_devices("GPU")) < 1) {
stop("No GPU found")
} else {
message("GPU found")
}

# load filters to size the input correctly
pwm <- k_constant(gen_datamotifs())

load("/net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation/data/train.RData")

# X_tmp  <- X_train[1:BATCH_SIZE, 1:500] |> make_ints()
# X_test <- k_one_hot(X_train[1:BATCH_SIZE, 1:500] |> make_ints(), 4L)
# y_btmp <- k_expand_dims(k_constant(as.integer(y_trainbinding[1:BATCH_SIZE])))
# y_dtmp <- k_expand_dims(k_expand_dims(k_constant(y_traindomain[1:BATCH_SIZE])))

train_list = list("x_input"=X_train)
#train_list = list("X_input"=X_tmp, "D_input"=k_expand_dims(k_expand_dims(k_constant(y_traindomain))), "y_true"=k_cast_to_floatx(k_expand_dims(k_expand_dims(k_constant(as.integer(y_trainbinding))))))

mds_train = tensor_slices_dataset(train_list)  |>
dataset_shuffle_and_repeat(buffer_size=10000L) |>
dataset_map_and_batch(function(x) list(k_one_hot(x$x_input, 4L), k_one_hot(x$x_input, 4L)),
                    BATCH_SIZE,
                    drop_remainder = TRUE,
                    num_parallel_calls = tf$data$AUTOTUNE) |>
dataset_prefetch_to_device(device = '/gpu:0', buffer_size = tf$data$AUTOTUNE)

model       <- lcVAE(pwm)
model_input <- model$encoder$input

outs <- model$forward(model_input)

mean_z   <- outs[[1]]
logvar_z <- outs[[2]]
z_sample <- outs[[3]]
x_pred   <- outs[[4]]

vae <- keras_model(
    inputs = model_input,
    outputs = outs,
    name = "vae"
)
vae$add_loss(model$elbo(vae$input, beta = 4.0))

#---- Instantiate the model

#save_model_tf(model$vae, "/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/models/da_ilcc/bvae")
#save_model_hdf5(model$vae, "/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/models/da_ilcc/bvae.h5")

#---- Train the model

# Filenames
dir_base  <- paste0(
    "/net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation/models/test_model/",
     "run_",
     format(Sys.time(),
     "%Y-%m-%d-%H:%M:%S")
)

fn_history <- paste0(
    dir_base,
    "/history.RDS"
)

# Create directories
dir.create(dir_base)

lr_schedule <- learning_rate_schedule_exponential_decay(
    initial_learning_rate = 0.001,
    decay_rate = 0.01,
    decay_steps = (dim(X_train)[1] / BATCH_SIZE)
)

model_checkpoints <- callback_model_checkpoint(
    dir_base,
    save_best_only = FALSE,
    save_weights_only = FALSE,
    save_freq = "epoch"
) 

early_stopping <- callback_early_stopping(
    monitor = "loss",
    patience = 3,
    verbose = 1,
    mode = "min",
    restore_best_weights = TRUE
)

naan_terminate <- callback_terminate_on_naan()

vae <- vae |> compile(
    loss = NULL,
    metrics = NULL,
    optimizer = keras$optimizers$Adamax(learning_rate = 0.001)
)

spe <- ceiling(dim(X_train)[1] / 2**9)

history <- vae |> fit(
    mds_train,
    epochs = 30,
    steps_per_epoch = spe,
    callbacks = c(model_checkpoints, early_stopping, naan_terminate)
)

saveRDS(history, file = fn_history)
