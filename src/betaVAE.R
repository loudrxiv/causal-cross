source("_utils.R")

main <- function() {

    # Can we see the GPU?
    if( length(tf$config$list_physical_devices("GPU")) < 1) {
    stop("No GPU found")
    } else {
    message("GPU found")
    }

    # load filters to size the input correctly
    pwm <- k_constant(gen_datamotifs())

    load("/net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation/data/train.RData")

    X_tmp  <- k_one_hot(X_train[1:BATCH_SIZE, 1:500] |> make_ints(), 4L)
    y_btmp <- k_expand_dims(k_constant(as.integer(y_trainbinding[1:BATCH_SIZE])))
    y_dtmp <- k_expand_dims(k_expand_dims(k_constant(y_traindomain[1:BATCH_SIZE])))

    model <- BetaVAE(
        latent_dim = 32L,
        input_dims = c(NULL, 4L), # c(NULL, 500L, 4L)
        input_shape = ,
        latent_dim=32L,
        loss_type="bce"
    )

    # model$vae$summary()

    # #---- Instantiate the model

    # save_model_tf(model$vae, "/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/models/da_ilcc/bvae")
    # save_model_hdf5(model$vae, "/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/models/da_ilcc/bvae.h5")

    # #---- Train the model

    # mds_X = tensor_slices_dataset(c(X_tmp, X_tmp)) |> dataset_batch(BATCH_SIZE)

    # history <- model$vae |> fit(
	# 			mds_X,
	# 			epochs = 15
    # )

}

if( !interactive() ) {
    main()
}
