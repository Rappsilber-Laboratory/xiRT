Parameters
===============

xiRT needs to two set of parameter files that are supplied via YAML files. The *xiRT parameters*
contain the settings that define the network architecture and learning tasks. With different / new
types of chromatography settings this is where the learning behavior is influenced. The *leanring
parameters* are used to define the learning data (which FDR) and some higher-level learning
behaviour. For instance, loading pretrained models and crossvalidation settings are controlled.


xiRT-Parameters
***************
The xiRT-Parameters can be divided into several categories that either reflect the individual
layers of the network or some higher level parameters.

Here is an example YAML file with comments (form xiRT v. 1.0.32)::

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
      merge_type: add   # how to combined individual network params after the Siamese network
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
        # continues means that linear (regressin) activation functions are used for the learning.
        # if this should be done the above parameters must also be adapted (weight, loss, metric, etc)
      continues:
        - rp
      fractions: # simply write fractions: [] if no fraction prediction is desired
        # if fractions (discrete) numbers should be used for the learning than this needs to be
        # indicated here
        # For fractions, either ordinal regression or classification can be used in the
        # fractions setting (regression is possible too).
        - scx
        - hsax

Apart from the very important neural network architecture definitions the target variable encoding
is also done in the YAML.

Learning-Parameters
*******************

Parameters that govern the separation of training and testing data for the learning.