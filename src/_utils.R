#----- Call the libraries I need

library(reticulate)
library(tensorflow)
library(tfdatasets)
library(BSgenome.Hsapiens.UCSC.hg38)
library(BSgenome.Mmusculus.UCSC.mm10)
library(rtracklayer)
library(GenomicRanges)
library(ggplot2)
library(keras)
library(pROC)
library(tfruns)
library(glue)

#" reticulate imports
reticulate::use_condaenv("tf-R", conda="/net/talisker/home/benos/mae117/mambaforge/condabin/mamba")
reticulate::source_python("src/betaVAE_VAs.py")
#tfpr   <- reticulate::import("tensorflow_probability")
#wandb  <- reticulate::import("wandb")

#----- Set up some 'constants'

BATCH_SIZE = 400L
VALBATCH_SIZE = 10000L

HOCOMOCO = "/net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation/data/hocomoco/human/hoc_pwms_hg_16.rdat"

#" General Sets
SPECIES = list("mm10", "hg38")
TFS = list("CTCF", "CEBPA", "Hnf4a", "RXRA")

GENOMES = list("mm10"="/net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation/data/mm10/mm10_no_alt_analysis_set_ENCODE.fasta",
  "hg38"="/net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation/data/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")

ROOT = "/net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation"
DATA_ROOT = paste0(ROOT,"/data/")
FIGURE_ROOT = paste0(ROOT,"/figures/")
MODEL_ROOT = paste0(ROOT,"/models/")

#" These files are created by the script 1_make_training_and_testing_data/1_runall_setup_model_data.sh
VAL_FILENAME = "chr1_random_1m.bed"
TRAIN_POS_FILENAME = "chr3toY_pos_shuf.bed"
TRAIN_NEG_FILENAME = "chr3toY_neg_shuf_run1_1E.bed"
TEST_FILENAME = "chr2.bed"

#" Where models will be saved during/after training
MODEL_ROOT = paste0(ROOT,"/models/")
DA_ILCC_ROOT = paste0(MODEL_ROOT, "da_ilcc/")
VAE_ILCC_ROOT = paste0(MODEL_ROOT, "vae_ilcc/")
KL_ILCC_ROOT = paste0(MODEL_ROOT, "kl_ilcc/")
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

base_params = list("batchsize" = 400,  # number of examples seen every batch during training
                "seqlen" = 500,       # the input sequence length that will be expected by the model
                "convfilters" = 240,   # number of filters in the convolutional layer
                "filtersize" = 20,     # the size of the convolutional filters
                "strides" = 15,        # the max-pooling layer's stride
                "pool_size" = 15,      # the max-pooling layer's pooling size
                "lstmnodes" = 32,      # "width" of the LSTM layer
                "dl1nodes" = 1024,     # neurons in the first dense layer (after LSTM)
                "dl2nodes" = 512,      # neurons in the second dense layer (before output)
                "dropout" = 0.5,       # fraction of neurons in the first dense layer to randomly dropout
                "valbatchsize" = 10000,
                "epochs" = 200)


ilcc_params = list("batchsize" = 400,  # number of examples seen every batch during training
                "seqlen" = 500,       # the input sequence length that will be expected by the model
                "convfilters" = 240,   # number of filters in the convolutional layer
                "filtersize" = 20,     # the size of the convolutional filters
                "strides" = 15,        # the max-pooling layer's stride
                "pool_size" = 15,      # the max-pooling layer's pooling size
                "lstmnodes" = 32,      # "width" of the LSTM layer
                "dl1nodes" = 1024,     # neurons in the first dense layer (after LSTM)
                "dl2nodes" = 512,      # neurons in the second dense layer (before output)
                "dropout" = 0.5,       # fraction of neurons in the first dense layer to randomly dropout
                "valbatchsize" = 10000,
                "epochs" = 15)

#" Set changing variables
tfs            = TFS[4]

source_species = SPECIES[1]
target_species = SPECIES[2]

source_root = paste0(DATA_ROOT,source_species,"/",tfs,"/")
target_root = paste0(DATA_ROOT,target_species,"/",tfs,"/")

src_bindingtrainposfile = paste0(source_root, TRAIN_POS_FILENAME)
src_bindingtrainnegfile = paste0(source_root, TRAIN_NEG_FILENAME)
tar_bindingtrainposfile = paste0(target_root, TRAIN_POS_FILENAME)
tar_bindingtrainnegfile = paste0(target_root, TRAIN_NEG_FILENAME)

sourcevalfile = paste0(source_root, VAL_FILENAME)
targetvalfile = paste0(target_root, VAL_FILENAME)

sourcetestfile = paste0(source_root, TEST_FILENAME)
targettestfile = paste0(target_root, TEST_FILENAME)

#----- Functions

# Read in the PWSM
gen_datamotifs <- function(path = "/net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation/data", safety = FALSE, save = FALSE) {
        
    pwm_filters <- readRDS(HOCOMOCO) |> lapply(as.matrix)                                                                                                                                                                             
                                                                                                                                                                                                                                        
    if( safety == TRUE) {

        message("Checking the size of the PWM")

        if(any(lapply(filters, is.matrix) |> unlist() == FALSE))            
        stop("filters invalid\n")

        if(any(lapply(filters, ncol) |> unlist() != ncol(filters[[1]]))) 
        stop("filters invalid\n")

        if(any(lapply(filters, nrow) |> unlist() != nrow(filters[[1]]))) 
        stop("filters invalid\n") 
    }
                                                                                                                                                                                                                                        
    filter_tensor <- pwm_filters |>                                                                                                                                                                                                   
    unlist() |>                                                                                                                                                                                                                          
    array(dim = c(nrow(pwm_filters[[1]]),4,length(pwm_filters)))

    if( save == TRUE ) {
        message("Saving motif data to data/PWM.RData")
        save(filter_tensor, file = glue(path, "/PWM.RData"))
    }

    return(filter_tensor)

}

# Generate and offer to save the training data

gen_datatrain <- function(path = "/net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation/data", save = FALSE) {

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
    y_traindomain  = y_traindomain[xwidth==500][nNs==0]
    y_trainbinding = y_trainbinding[xwidth==500][nNs==0]

    if( save == TRUE ) {
        message("Saving training data to data/train.RData")
        save(X_train, y_trainbinding, y_traindomain, file = glue(path, "/train.RData"))
    }
    
    return( c(X_train, y_traindomain, y_trainbinding) )

}

# Generate and offer to save the testing data

gen_datatest <- function(path = "/net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation/data", save = FALSE) {

  X_gr_src = GRangesList("src_bindingtrainpos"=import(src_bindingtrainposfile),                                                                                                                                                   
                      "src_bindingtrainneg"=import(src_bindingtrainnegfile),
                      "src_bindingtest"=import(sourcetestfile),
                      "src_bindingval"=import(sourcevalfile))

  X_gr_tar = GRangesList("tar_bindingtrainpos"=import(tar_bindingtrainposfile),                                                                                                                                                   
                      "tar_bindingtrainneg"=import(tar_bindingtrainnegfile),
                      "tar_bindingtest"=import(targettestfile),
                      "tar_bindingval"=import(targetvalfile))

  X_data_src      <- lapply(X_gr_src, function(x) Biostrings::getSeq(Mmusculus, x))                                                                                                                                                       
  X_labels_src    <- lapply(X_gr_src, function(x) mcols(x)$name)                                                                                                                                                                          
  D_labels_src    <- lapply(X_gr_src, function(x) rep(1, length(x)))
                          
  X_data_tar      <- lapply(X_gr_tar, function(x) Biostrings::getSeq(Hsapiens, x))                                                                                                                                                        
  X_labels_tar    <- lapply(X_gr_tar, function(x) mcols(x)$name)                                                                                                                                                                          
  D_labels_tar    <- lapply(X_gr_tar, function(x) rep(0, length(x)))

  X_test        <- c(X_data_src$src_bindingtest, X_data_tar$tar_bindingtest)
  y_testbinding <- c(X_labels_src$src_bindingtest, X_labels_tar$tar_bindingtest)
  y_testdomain  <- c(D_labels_src$src_bindingtest, D_labels_tar$tar_bindingtest)

  nNs    = Biostrings::vcountPattern('N', X_test)
  xwidth = width(X_test)

  X_test        = X_test[xwidth==500][nNs==0] |> make_ints()
  y_testdomain  = y_testdomain[xwidth==500][nNs==0]
  y_testbinding = y_testbinding[xwidth==500][nNs==0]

  if( save == TRUE ) {
      message("Saving test data to data/test.RData")
      save(X_test, y_testdomain, y_testbinding, file = glue(path, "/test.RData"))
  }

  return( c(X_test, y_testdomain, y_testbinding) )

}

# Generate and offer to save the validation data

gen_dataval <- function(path = "/net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation/data", save = FALSE) {

  X_gr_src = GRangesList("src_bindingtrainpos"=import(src_bindingtrainposfile),                                                                                                                                                   
                      "src_bindingtrainneg"=import(src_bindingtrainnegfile),
                      "src_bindingtest"=import(sourcetestfile),
                      "src_bindingval"=import(sourcevalfile))

  X_gr_tar = GRangesList("tar_bindingtrainpos"=import(tar_bindingtrainposfile),                                                                                                                                                   
                      "tar_bindingtrainneg"=import(tar_bindingtrainnegfile),
                      "tar_bindingtest"=import(targettestfile),
                      "tar_bindingval"=import(targetvalfile))

  X_data_src      <- lapply(X_gr_src, function(x) Biostrings::getSeq(Mmusculus, x))                                                                                                                                                       
  X_labels_src    <- lapply(X_gr_src, function(x) mcols(x)$name)                                                                                                                                                                          
  D_labels_src    <- lapply(X_gr_src, function(x) rep(1, length(x)))

  X_data_tar      <- lapply(X_gr_tar, function(x) Biostrings::getSeq(Hsapiens, x))                                                                                                                                                        
  X_labels_tar    <- lapply(X_gr_tar, function(x) mcols(x)$name)                                                                                                                                                                          
  D_labels_tar    <- lapply(X_gr_tar, function(x) rep(0, length(x)))

  X_val        <- c(X_data_src$src_bindingval, X_data_tar$tar_bindingval)
  y_valbinding <- c(X_labels_src$src_bindingval, X_labels_tar$tar_bindingval)
  y_valdomain  <- c(D_labels_src$src_bindingval, D_labels_tar$tar_bindingval)

  nNs    = Biostrings::vcountPattern('N', X_val)
  xwidth = width(X_val)

  X_val        = X_val[xwidth==500][nNs==0] |> make_ints()
  y_valdomain  = y_valdomain[xwidth==500][nNs==0]
  y_valbinding = y_valbinding[xwidth==500][nNs==0]

  if( save == TRUE ) {
      message("Saving val data to data/val.RData")
      save(X_val, y_valdomain, y_valbinding, file = glue(path, "/val.RData"))
  }

  return( c(X_val, y_valdomain, y_valbinding) )

}

recon_loss <- function(y_true, y_pred) { return(tf$reduce_mean(loss_binary_crossentropy(y_true, y_pred,label_smoothing = k_epsilon()))) }
kl_loss    <- function(z_mean, z_logVar) { return(tf$reduce_mean(loss_kullback_leibler_divergence(z_mean, z_logVar))) }
vae_loss   <- function(kl_loss, recon_loss) { return(  1E-4 * ((4 * kl_loss) + recon_loss) ) }

mi_loss    <- function(z_src, y_pred) { 

    # Get the marginal PDF of both inputs and their joint
    marginal_y  <- tf$reduce_mean(y_pred, axis=1L)
    marginal_zc <- tf$reduce_mean(z_src, axis=1L)
    joint_yzc   <- tf$linalg$matmul(marginal_y, tf$transpose(marginal_zc))

    # Get the entropy
    marginal_y_entropy = -tf$reduce_sum(marginal_y * tf$math$log(marginal_y + k_epsilon()))
    marginal_zc_entropy = -tf$reduce_sum(marginal_zc * tf$math$log(marginal_zc + k_epsilon()))
    joint_yzc_entropy = -tf$reduce_sum(joint_yzc * tf$math$log(joint_yzc + k_epsilon()))

    # Get the mutual informaton 
    mi = marginal_y_entropy + marginal_zc_entropy - joint_yzc_entropy

    # Check for NaNs and other invalid values
    mi = tf$debugging$check_numerics(mi, "mi is NaN or Inf")
    
    return(mi) 
    
}

total_loss <- function(kl_loss, recon_loss, mi_loss) { return(  1E-4 * (recon_loss - (4 * kl_loss)) + (0.1 * mi_loss) ) }

make_ints = function(seqs){
  tmp      <- as.matrix(seqs)
  seqs_int <- apply(tmp, 2,function(x) as.factor(x) |> as.integer() - 1L)
  return(seqs_int)
}

make_1he = function(seqs){
  seq_ints <- make_ints(seqs)
  seqs_1he  <- tf$one_hot(seqs_int,4L) |> as.array()
  return(seqs_1he) 
}

plot_figure = function(name, data) {

  name=paste0(FIGURE_ROOT, paste0(name, "_plot.pdf"))
  
  pdf(file=name)
  
  matplot(data,
          metrics=NULL,
          method="base",
          theme_bw=ggplot2::theme_grey(),
          type ="o",
          xlab="epochs",
          ylab="name",
          main="Reconstruction (hg38)",
          pch=c(19, 17),
          col=c("blue","red"),
          bg="lightblue",
          bty="n"
  )
  grid()
  legend("topright", c("tr. loss", "val. loss"), pch=c(19, 17), col=c("blue", "red"))
  dev.off() 
}
