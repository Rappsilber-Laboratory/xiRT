"""Module for constants in the xirt package."""
from xirt import __version__

learning_params = f"""
# Learning options generated with xiRT v. {__version__}

# the preprocessing options define how the sequences are encoded / filtered. Usually, default values
# are fine.
# If transfer learning is intended, the label encoder and max_length parameter need to be adapted.

preprocessing:
    # label encoder, str or none. If str, use a previously trained label encoder to translate
    # amino acids to specific integers. If you are using xiRT on a single data file set to None
    # default None
    le: None
    
    # max sequence length, integer. Filter all sequences longer than this number. Disable by setting
    # it to -1
    # default -1
    max_length: -1
    
    # for crosslinks only, bool: encode crosslinked residues as different residues than their 
    # unmodified counter parts
    # e.g. a crosslinked K, will be encoded as clK in modX format.
    # default True
    cl_residue: True
    
    # filter, str. string filter that must be contained in the description for a CSM to be included
    # default ""
    filter: ""

# these options are crucial for the setting up xiRT with the correct training mode. Stay strong! 
# It's easier than it seems right now. 
# Check the readthedocs documentation if you need more info / examples.
train:
  # float value, defines cutoff to filter the input CSMs, e.g. all CSMs with a lower fdr are 
  # used for training
  # default 0.01
  fdr: 0.01
  
  # int, the number of crossvalidation folds to be done. 1=nocv, 3=minimal value, recommended
  # alternatives with higher run time:5 or 10.
  # default 1
  ncv: 1
  
  # bool, if True the training data is used to fit a new neural network model after the 
  # cross-validation step, this model is used for the prediction of RTs for all peptides > 
  # the given FDR value.
  # refit=False: use best CV predictor; b) refit=True: retrain on all CSMs < 0.01 FDR.
  # default False
  refit: False
  
  # str, important that defines the training mode (important!)
  # "train", train on entire data set: use
  # "crossvalidation", perform crossvalidation on the input data (train multiple classifiers)
  # "predict", do NOT train on the supplied CSMs but simply predict with an already trained model
  # default "train"
  mode: "train"
  
  # str, augment the input data by swapping sequences (peptide1, peptide2). Marginal gains in
  # predicition were observed here.
  # Can usually, be left as False. If you are dealing with very small data sets, this option 
  # might also help.
  # default False
  augment: False
  
  # str, multiple sequence types are supported: "linear", "crosslink", "pseudolinear" (concatenate
  # peptide1 and peptide2 sequences)
  # default "crosslink"
  sequence_type: "crosslink"
  
  # str (file location), this option can be set with any of the above described options.
  # if a valid weight set is supplied, the network is initalized with the given weights
  # default "None"
  pretrained_weights: "None"
  
  # str (file location), similarly to the option above, a pretrained model can be supplied. 
  # this is necessary when (extreme) transfer-learning applications are intended (e.g. different 
  # number of fractions for e.g. SCX)
  # this requires adjustments of the network architecture
  # default: "None"
  pretrained_model: "None"
  
  # float, defines the fraction of test data (e.g. a small fraction of the training folds that is
  # used for validation
  # default 0.10
  test_frac: 0.10
  
  # float, used for downsampling the input data (e.g. to create learning curves). Can usually left as 1.
  # default 1
  sample_frac: 1
  
  # int, seed value for the sampling described above
  # default 21
  sample_state: 21
"""

xirt_params = f"""
# xiRT options generated with xiRT v. {__version__}
# options for the recurrent layer used in xiRT
# can usually be used with default values, except for type
LSTM:
  # activation parameters, leave as default unless you know what you are doing
  activation: tanh
  activity_regularization: l2
  activityregularizer_value: 0.001
  
  # option that activates the bidirectional layer to the used LSTM layer
  bidirectional: true
  
  # kernal regularization, leave as default
  kernel_regularization: l2
  kernelregularizer_value: 0.001
  lstm_bn: true
  
  # central layer parameters
  # increasing the values here will drastically increase runtime but might also improve results
  # usually, 1 and GRU (for CPUs) or CuDNNGRU (for GPUs) will deliver good performance
  nlayers: 1
  type: GRU
  units: 50
 
# dense parameters are used for the individual task subnetworks (e.g. RP, SCX, ...)
dense:
  # activation functions in the layers between the embedding and prediction layer
  # recommended to leave on defaults for most applications
  activation:
  - relu
  - relu
  - relu
  
  # boolean indicator if batch_normalization shoulde be used, leave on default
  # recommended to leave on defaults for most applications
  dense_bn:
  - true
  - true
  - true
  
  # dropout rate to use
  # recommended to leave on defaults for most applications
  dropout:
  - 0.1
  - 0.1
  - 0.1
  
  # regularization methods to use on the kernels, leave on defaults
  kernel_regularizer:
  - l2
  - l2
  - l2
  regularization:
  - true
  - true
  - true
  regularizer_value:
  - 0.001
  - 0.001
  - 0.001
  # size of the individual layers, defaults deliver good results. Changes here might need adjustments 
  # on dropout rates and other hyper-parameters
  neurons:
  - 300
  - 150
  - 50
  
  # int, number of layers to use. Note that all other parameters in the 'dense' section
  # must be adapted to the new number used in this variable
  nlayers: 3

# dimension of the embedding output
embedding:
  length: 50
 
# parameters influencing the learning 
learning:
  # numbers of samples to pass during a single iteration
  batch_size: 512
  # number of epochs to train
  epochs: 50
  # other tested/reasonable values for learning rate: 0.003, 0.001
  learningrate: 0.01
  verbose: 1
  # default optimizer, most tensorflow optimizers are implemented as well
  optimizer: adam
  
#!!!!!!!!!!!!!!!!!! most important parameters!!!!!!!!!!!!!!!
output:
  # task-parameters. Here the prefix hsax and rp are used to build and parameterize the
  # respective sub-networks (this prefix must also match the "predictions" section. 
  # each task needs to contain the sufixes: activation, column, dimension, loss, metric and weight.

  # They must be carefully adapted for each prediction task.
  # recommended to use sigmoid for fractions (SCX/hSAX) if ordinal regression method should be used
  hsax-activation: sigmoid
  # column where the fraction RT is in the CSV input ("xx_ordinal" xx_
  hsax-column: hsax_ordinal
  # the number of unique / distinct values (e.g. fractions)
  hsax-dimension: 10
  # must be binary_crossentropy for sigmoid activations
  hsax-loss: binary_crossentropy
  # must be mse
  hsax-metrics: mse
  # weight parameter to combine the loss of this task to any other defined task
  hsax-weight: 50
  
  # use linear for regression tasks (revesed phase)
  rp-activation: linear
  rp-column: rp
  # dimension is always 1 for regression
  rp-dimension: 1
  # loss and metrics should not be changed from mse
  rp-loss: mse
  rp-metrics: mse
  # again, a weight parameter that might need tuning for multi-task settings
  rp-weight: 1

# siames parameters
siamese:
  # set to True for crosslinks (default)
  use: True
  # define how to combine the outputs of the siamese layers, most tensorflow options are supported.
  # default value should be fine
  merge_type: add
  # add predictions for single peptides based on the crosslink model (default)
  single_predictions: True
callbacks:
  # for debugging and model storage
  # define which callbacks to use.
  # default values are fine here and should not be changed
  # options define the meta data that is written throughout the training process. The results can 
  # be find in the callback in the specified outdir
  check_point: True
  log_csv: True
  # early stopping callback
  early_stopping: True
  early_stopping_patience: 15
  tensor_board: False
  progressbar: True
  # reduce learning rate callback
  reduce_lr: True
  reduce_lr_factor: 0.5
  reduce_lr_patience: 15
predictions:
  # define the prediction tasks unambiguously as they appear in the output file; need to match
  # column labels defined in output
  # "continues" is reserved for regression problems e.g. reversed-phase chromatography here
  continues:
    - rp
  # fractions are reserved for classification or ordinal regression problems e.g. 
  # fractionation method that led to discrete fractions
  # use [] if no fraction prediction is desired
  fractions: # simply write fractions: [] if no fraction prediction is desired
    - hsax
"""


readme = f"""

xiRT ReadMe:
------------
This folder contains the results from running xiRT v. {__version__}. The following
descriptions summarize the most important output data and formats.

Important files:
----------------
1. xirt_logger.log
This file summarizes and logs the training procedure. It contains the parameters used to run xiRT
but also short summary values from the training process.

2. processed_psms.csv
CSMs or PSMs from the input data but with the additional and temporary columns used by xiRT.

3. error_features.csv
Contains the predictions and errors (observed - predicted) for each PSM / CSM.

4. error_features_interactions.csv
Contains further features that are derived from the errors (e.g. products, sums, absolute values)

5. figures for quality control
- cv_epochs_loss.svg / cv_epochs_metrics.svg - plots the training performance over time.
- cv_summary_strip_loss.svg / cv_summary_strip_metric.svg - plots summarizing the CV fold results
- error_characteristics.svg - plots prediction errors between TT/TD/DD identifications
- qc_cv_01/02/dd - x vs. y plots of predictions and observations for the tasks (pred fold)
- qc_cv_-1 - x vs. y plots of predictions and observations for the data with >1% FDR

Optional files:
---------------
6. epoch_history.csv
Training performance over time (epochs).

7. model_summary.csv
Summarizes the model performance in more depth (metrics, training splits, input files, etc.)

8. callbacks folder
In-depth results from the model training process, e.g. trained weights and model architectures.
Also contains the encoder and data used in python as pickle objects.

Please visit the documentation to get more details on the output data:
https://xirt.readthedocs.io/en/latest/results.html
"""