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

reticulate::use_condaenv("tf-R", conda="/net/talisker/home/benos/mae117/mambaforge/condabin/mamba")
reticulate::source_python(paste0(getwd(), "/src/lcVAE.py"))
reticulate::source_python(paste0(getwd(), "/src/nsfs.py"))

#" reticulate imports
wandb <- reticulate::import("wandb")
tfpy  <- reticulate::import("tensorflow")
tfpr  <- reticulate::import("tensorflow_probability")
tfp   <- tfpr$distributions
tfb   <- tfpr$bijectors

#----- Set up some variables & 'constants'

ROOT = "/net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation"
TFS       <- c("CTCF")
SPECIES   <- c("mm10", "hg38")
RUNS      <- c(1)

# data files generated from shaun's scripts
VAL_FILENAME        <- "chr1_random_1m.bed"
TEST_FILENAME       <- "chr2.bed"
TRAIN_POS_FILENAME  <- "chr3toY_pos_shuf.bed"
TRAIN_NEG_FILENAME  <- "chr3toY_neg_shuf_runX_ZE.bed"

params <- list(
  "batchsize" = 512,       # number of examples seen every batch during training
  #"seqlen" = 500,         # the input sequence length that will be expected by the model
  #"convfilters" = 240,    # number of filters in the convolutional layer
  #"filtersize" = 20,      # the size of the convolutional filters
  #"strides" = 15,         # the max-pooling layer's stride
  #"pool_size" = 15,       # the max-pooling layer's pooling size
  #"lstmnodes" = 32,       # "width" of the LSTM layer
  #"dl1nodes" = 1024,      # neurons in the first dense layer (after LSTM)
  #"dl2nodes" = 512,       # neurons in the second dense layer (before output)
  #"dropout" = 0.5,        # fraction of neurons in the first dense layer to randomly dropout
  "valbatchsize" = 10000   # Talking about the testing data
  #"epochs" = 15
)

#----- Functions

# Read in the PWSM
gen_datamotifs <- function(path = ROOT, safety = FALSE, save = FALSE) {

  hocomoco_loc      <- paste0(path, "/data/hocomoco/human/hoc_pwms_hg_16.rdat")
  hocomoco_filters  <- readRDS(hocomoco_loc) |> lapply(as.matrix)

  if (safety == TRUE) {

      message("Checking the size of the PWM...\n")

      if ( any(lapply(hocomoco_filters, is.matrix) |> unlist() == FALSE) ) { stop("filters invalid\n") }
      if ( any(lapply(hocomoco_filters, ncol) |> unlist() != ncol(hocomoco_filters[[1]])) ) { stop("filters invalid\n") }
      if ( any(lapply(hocomoco_filters, nrow) |> unlist() != nrow(hocomoco_filters[[1]])) ) { stop("filters invalid\n") }

  }
                                                                                                                                                                                                                                        
    filter_tensor <- hocomoco_filters |>                                                                                                                                                                                                   
    unlist() |>                                                                                                                                                                                                                          
    array(dim = c(nrow(hocomoco_filters[[1]]),4,length(hocomoco_filters)))

    if ( save == TRUE ) {
        message("Saving motif data to data/PWM.RData")
        save(filter_tensor, file = paste0(path, "/PWM.RData"))
    }

    return(filter_tensor)

}

# Generate and offer to save the training data

# so essentially what I want to do is go through this step by step: 
# 1. there is 1 pos train file per species (272661 seqs) I want to combine the 15 negative examples (total 272661) into a block and then combine those two toghethe
# 2. do this for the other species and create the labels
# 3. the validation data requires no effort, but make sure the species labels are correct
# 4. chr2 is the test and the 'validation' data we use here

gen_datatrain <- function(path = ROOT, save = FALSE) {

    # Create the files names for the negative bound sequences
    construct_names <- function(root = ROOT, tfs = TFs, runs = RUNS) {
      names <- c()
      for (tf in tfs) {
        for (spec in species) {
          for (run in runs) {
            names <- c(names, paste0(tf, "_", spec, "_run", run))
          }
        }
      }

      return(names)

    }

     rep(paste0(ROOT, "/data/"), 15)

    # X_gr_src = GRangesList("src_bindingtrainpos"=import(src_bindingtrainposfile),                                                                                                                                                   
    #                     "src_bindingtrainneg"=import(src_bindingtrainnegfile),
    #                     "src_bindingtest"=import(sourcetestfile),
    #                     "src_bindingval"=import(sourcevalfile))

    # X_gr_tar = GRangesList("tar_bindingtrainpos"=import(tar_bindingtrainposfile),                                                                                                                                                   
    #                     "tar_bindingtrainneg"=import(tar_bindingtrainnegfile),
    #                     "tar_bindingtest"=import(targettestfile),
    #                     "tar_bindingval"=import(targetvalfile))

    # X_data_src      = lapply(X_gr_src, function(x) Biostrings::getSeq(Mmusculus, x))                                                                                                                                                       
    # X_labels_src    = lapply(X_gr_src, function(x) mcols(x)$name)                                                                                                                                                                          
    # D_labels_src    = lapply(X_gr_src, function(x) rep(1, length(x)))
                            
    # X_data_tar      = lapply(X_gr_tar, function(x) Biostrings::getSeq(Hsapiens, x))                                                                                                                                                        
    # X_labels_tar    = lapply(X_gr_tar, function(x) mcols(x)$name)                                                                                                                                                                          
    # D_labels_tar    = lapply(X_gr_tar, function(x) rep(0, length(x)))
                            
    # X_train = c(X_data_src$src_bindingtrainpos,
    #             X_data_src$src_bindingtrainneg,
    #             X_data_tar$tar_bindingtrainpos,
    #             X_data_tar$tar_bindingtrainneg)

    # y_trainbinding  = c(X_labels_src$src_bindingtrainpos, 
    #                     X_labels_src$src_bindingtrainneg,
    #                     X_labels_tar$tar_bindingtrainpos,
    #                     X_labels_tar$tar_bindingtrainneg)

    # y_traindomain   = c(D_labels_src$src_bindingtrainpos,
    #                     D_labels_src$src_bindingtrainneg,
    #                     D_labels_tar$tar_bindingtrainpos,
    #                     D_labels_tar$tar_bindingtrainneg)

    # nNs       = Biostrings::vcountPattern('N', X_train)
    # xwidth    = width(X_train)

    # X_train        = X_train[xwidth==500][nNs==0] |> make_ints()
    # y_traindomain  = y_traindomain[xwidth==500][nNs==0]
    # y_trainbinding = y_trainbinding[xwidth==500][nNs==0]

    # if( save == TRUE ) {
    #     message("Saving training data to data/train.RData")
    #     save(X_train, y_trainbinding, y_traindomain, file = glue(path, "/train.RData"))
    # }
    
    # return( c(X_train, y_traindomain, y_trainbinding) )

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