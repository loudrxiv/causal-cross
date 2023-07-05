source(paste0(getwd(), "/src/_utils.R"))

# Can we see the GPU?
if (length(tf$config$list_physical_devices("GPU")) < 1) {
stop("No GPU found")
} else {
message("GPU found")
}

# load filters to size the input correctly
pwm <- k_constant(gen_datamotifs())

load("/net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation/data/train.RData")

tmp   <- X_train[1:BATCH_SIZE, 1:500] |> make_ints()
X_tmp <- k_one_hot(X_train[1:BATCH_SIZE, 1:500] |> make_ints(), 4L)
#b_tmp <- k_expand_dims(k_constant(as.integer(y_trainbinding[1:BATCH_SIZE])))
#d_tmp <- k_expand_dims(k_expand_dims(k_constant(y_traindomain[1:BATCH_SIZE])))

train_list = list("x_input"=X_train) # list("X_input"=X_tmp, "D_input"=k_expand_dims(k_expand_dims(k_constant(y_traindomain))), "y_true"=k_cast_to_floatx(k_expand_dims(k_expand_dims(k_constant(as.integer(y_trainbinding))))))

mds_train <- tensor_slices_dataset(train_list) |>

dataset_shuffle(
    buffer_size = 10000,
    reshuffle_each_iteration = TRUE
) |> # dataset_shuffle_and_repeat(buffer_size=10000L) |>

dataset_map_and_batch(
    function(x) list(k_one_hot(x$x_input, 4), k_one_hot(x$x_input, 4)),
    BATCH_SIZE,
    drop_remainder = TRUE,
    num_parallel_calls = tf$data$AUTOTUNE
) |>

dataset_prefetch_to_device(
    device = "/gpu:0",
    buffer_size = tf$data$AUTOTUNE
)

model       <- lcVAE(pwm)
model_input <- model$encoder$input
outs <- model$forward(model_input)

# ys = spline_flow().forward(xs)
# ys_inv = spline_flow().inverse(ys)  # ys_inv ~= xs

mean_z   <- outs[[1]]
logvar_z <- outs[[2]]
zhat_s   <- spline_flow()$forward(logvar_z)
z_c_s    <- k_concatenate(c(mean_z, zhat_s))
z_sample <- outs[[3]]
x_pred   <- outs[[4]]

vae <- keras_model(
    inputs = model_input,
    outputs = outs,
    name = "vae"
)
nsf <- keras_model(
    inputs = model_input,
    outputs = z_c_s,
    name = "nsf"
)
vae$add_loss(model$elbo(vae$input, beta = 4.0))

#---- Compile and fit the model

# Filenames
dir_base  <- paste0(
    "/net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation/models/test_model/",
     "run_",
     format(
        Sys.time(),
        "%Y-%m-%d-%H:%M:%S"
        )
)

fn_history <- paste0(
    dir_base,
    "/history.RDS"
)

fn_log <- paste0(
    dir_base,
    "/log.csv"
)

# Create directories & train loggers
dir.create(dir_base)

# wandb config
wandb$tensorboard$patch(root_logdir = paste0(dir_base, "/train"))
wandb$init(
    project = "causal-domain-adaptation",
    dir = dir_base,
    group = "Improving Architecture",
    name = "Full Encoder and Decoder + FTRL",
    notes = "",
    #sync_tensorboard = TRUE
)

vae <- vae |> compile(
    loss = NULL,
    metrics = NULL,
    optimizer =  optimizer_adamax(learning_rate = 0.001)
)

history <- vae |> fit(
    mds_train,
    epochs = 500,
    #steps_per_epoch = ceiling(dim(X_train)[1] / 2**9),
    callbacks = c(
        callback_model_checkpoint(
            dir_base, # paste0(dir_base, "/model_{epoch}")
            monitor = "loss",
            verbose = 1,
            save_best_only = TRUE,
            mode = "min",
            save_freq = "epoch"),
        callback_early_stopping(
            monitor = "loss",
            patience = 7,
            verbose = 1,
            mode = "min",
            restore_best_weights = TRUE),
        callback_terminate_on_naan(),
        callback_csv_logger(filename = fn_log),
        callback_reduce_lr_on_plateau(
            monitor = "loss",
            patience = 3,
            verbose = 1,
            mode = "min"),
        callback_tensorboard(
            log_dir = dir_base,
            histogram_freq = 1,
            write_graph = TRUE,
            write_grads = TRUE,
            write_images = TRUE,
            embeddings_freq = 1,
            update_freq = "epoch")
    )
)

wandb$finish()

save_model_tf(vae, paste0(dir_base, "/model_dir"))
save_model_hdf5(vae, paste0(dir_base, "/model.h5"))
saveRDS(history, file = fn_history)