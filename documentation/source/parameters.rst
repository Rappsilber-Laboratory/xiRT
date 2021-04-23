.. _linking-parameters:

Parameters
==========

xiRT needs two sets of parameters that are supplied via two YAML files. The *xiRT parameters*
contain the settings that define the network architecture and learning tasks. With different / new
types of chromatography or other separation settings, the learning behavior is influenced and hence
needs adjustement. The *learning parameters* are used to define the learning data (e.g. filtered to
a desired confidence limit) and some higher-level learning behaviour. For instance, settings for 
loading pretrained models and cross-validation are controlled.


xiRT-Parameters
***************
The xiRT-Parameters can be divided into several categories that either reflect the individual
layers of the network or some higher level parameters. Since the input file structure is very
dynamic, the xiRT configuration needs to be handled with care. For example, the RT information
in the input data is encoded in the *predictions* section. Here, the column names of the RT
data needs to be defined. Accordingly, the learning options in the *output* section must be
adapted. Each prediction task needs the parameters x-activation, x-column, x-dimension,
x-loss, x-metrics and x-weight, where "x" represents the seperation method of interest.

Please see here for an example YAML file including comments (form xiRT v. 1.0.32)::

    LSTM:
      activation: tanh      # activiation function
      activity_regularization: l2       # regularization to use
      activityregularizer_value: 0.001  # lambda value
      bidirectional: true               # if RNN-cell should work bidirectional
      kernel_regularization: l2         # kernel regularization method
      kernelregularizer_value: 0.001    # lambda value
      lstm_bn: true                     # use batch normalization
      nlayers: 1                        # number of layers
      type: GRU                         # RNN type of layer to use: GRU, LSTM and CuDNNGRU, CuDNNGRU
      units: 50                         # number of units in the RNN cell
    dense:          # parameters for the dense layers
      activation:   # type of activiations to use for the layers (for each layer)
      - relu    # activiation function
      - relu
      - relu
      dense_bn: # use batch normalization
      - true
      - true
      - true
      dropout:  # dropout usage rate
      - 0.1
      - 0.1
      - 0.1
      kernel_regularizer:   # regularizer for the kernel
      - l2
      - l2
      - l2
      neurons:  # number of neurons per layer
      - 300
      - 150
      - 75
      nlayers: 3    # number of layers, this number must be matched by the parameters
      regularization:   # use regularization
      - true
      - true
      - true
      regularizer_value:    # lambda values
      - 0.001
      - 0.001
      - 0.001
    embedding:      # parameters for the embedding layer
      length: 50    # embedding vector dimension
    learning:       # learning phase parameters
      batch_size: 128   # observations to use per batch
      epochs: 75        # maximal epochs to train
      learningrate: 0.001   # initial learning rate
      verbose: 1        # verbose training information
    output:     # important learning parameters
      callback-path: data/results/callbacks/       # network architectures and weights will be stored here
      # the following parameters need to be defined for each chromatography variable
      hsax-activation: sigmoid  # activiation function, use linear for regression
      hsax-column: hsax_ordinal # output column name
      hsax-dimension: 10    # equals number of fractions
      hsax-loss: binary_crossentropy    # loss function, must be adapted for regression / classification
      hsax-metrics: mse     # report the following metric
      hsax-weight: 50       # weight to be used in the loss function
      rp-activation: linear
      rp-column: rp
      rp-dimension: 1
      rp-loss: mse
      rp-metrics: mse
      rp-weight: 1
      scx-activation: sigmoid
      scx-column: scx_ordinal
      scx-dimension: 9
      scx-loss: binary_crossentropy
      scx-metrics: mse
      scx-weight: 50
    siamese:        # parameters for the siamese part
      use: True         # use siamese
      merge_type: add   # how to combine individual network params after the Siamese network
      single_predictions: True  # use also single peptide predictions
    callbacks:                  # callbacks to use
      check_point: True
      log_csv: True
      early_stopping: True
      early_stopping_patience: 15
      tensor_board: False
      progressbar: True
      reduce_lr: True
      reduce_lr_factor: 0.5
      reduce_lr_patience: 15
    predictions:
        # parameters that define how the input variables are treated
        # "continues" means that linear (regression) activation functions are used for the learning.
        # if this should be done, the above parameters must also be adapted (weight, loss, metric, etc)
      continues:
        - rp
      fractions: # simply write fractions: [] if no fraction prediction is desired
        # if (discrete) fraction numbers should be used for the learning, this needs to be
        # indicated here
        # For fractions, either ordinal regression or classification can be used in the
        # fractions setting (regression is possible too).
        - scx
        - hsax

Apart from the very important neural network architecture definitions, the target variable encoding
is also defined in the YAML.

Learning-Parameters
*******************

Parameters that govern the separation of training and testing data for the learning.

Here is an example YAML file with comments (form xiRT v. 1.0.32)::

    # preprocessing options:
    # le: str, label encoder location. Only needed for transfer learning, or usage of pretrained
    # max_length: float, max length of sequences
    # cl_residue: bool, if True crosslinked residues are decoded as Kcl or in modX format clK
    preprocessing:
        le: None
        max_length: -1 # -1
        cl_residue: True


    # fdr: float, a FDR cutoff for peptide matches to be included in the training process
    # ncv: int, number of CV folds to perform to avoid training/prediction on the same data
    # mode: str, must be one of: train, crossvalidation, predict
    # train and transfer share the same options that are necessary to run xiML, here is a brief rundown:
    # augment: bool, if data augmentation should be performed
    # sequence_type: str, must be linear, crosslink, pseudolinear. crosslink uses the siamese network
    # pretrained_weights: "None", str location of neural network weights. Only embedding/RNN weights
    #   are loaded. pretrained weights can be used with all modes, essentially resembling a transfer
    #   learning set-up
    # sample_frac: float, (0, 1) used for downsampling the input data (e.g. for learning curves).
    #   Usually, left to 1 if all data should be used for training
    # sample_state: int, random state to be used for shuffling the data. Important for recreating
    #   results.
    # refit: bool, if True the classifier is refit on all the data below the FDR cutoff to predict
    # the RT times for all peptide matches above the FDR cutoff. If false, the already trained CV
    # classifier with the lowest validation loss is chosen
    train:
      fdr: 0.01
      ncv: 3
      mode: "crossvalidation" # other modes are: train / crossvalidation / predict
      augment: False
      sequence_type: "crosslink"
      pretrained_weights: "None"
      test_frac: 0.10
      sample_frac: 1
      sample_state: 21
      refit: False

Generally, it is better to supply more high-quality data than more data. Sometimes considerable
drops in performance can be observed when 5% instead of 1% input data is used. However, there is
no general rule of thumb and this needs to be optimized per run / experiment.

Hyperparameter-Optimization
***************************

Neural Networks are very sensitive to their hyperparameters. To automate the daunting task
of finding the right hyperparameters two
`utils <https://github.com/Rappsilber-Laboratory/xiRT/tree/master/utils>`_ are shipped with xiRT.
1) a convenience function that generates YAML files from a *grid YAML* file. 2) a snakemake workflow
that can be used to run xiRT with each parameter combination.

The grid will be generated based on all entries where not a single value is passed but a list of
values. This can lead to an enormous search space, so step-wise optimization is sometimes the
only viable option.
