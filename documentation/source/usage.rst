General Usage
=============
The command line interface (CLI) requires three inputs:

1) input spectra matches file (CSMs or PSMs)
2) a `YAML <https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html>`_ file to configure the neural network architecture
3) another YAML file to configure the general training / prediction behaviour, called setup-config

Configs are available via `github <https://github.com/Rappsilber-Laboratory/xiRT/tree/master/default_parameters>`_).
Alternatively, up-to-date configs can be generated using the xiRT package itself:

.. code-block:: console

    > xirt -p learning_params.yaml
    >
    > xirt -s xirt_params.yaml

To use xiRT these options are combined as shown below:

.. code-block:: console

    >xirt -i CSMs.csv -o out_dir -x xirt_params.yaml -l learning_params.yaml

To adapt xiRT's parameters, a yaml config file needs to be prepared. The configuration file 
determines network parameters (number of neurons, layers, regularization) but also defines the 
prediction task (classification / regression / ordinal regression). In general, the
default values do not need to be changed for standard use cases. Depending
on the decoding of the target variable, the output layers may need to be adapted.
For a usual RP prediction, regression is essentially the only viable option. For SCX / hSAX (e.g. general classification
from fractionation experiments) the prediction task can be formulated as classification,
regression or ordinal regression. For the usage of regression for fractionation, we recommend to use
fraction-specific eluent concentrations as prediction target variable (raw fraction numbers are also possible).
Please see some examples below to learn more about the different
parameterizations.

Quick start
'''''''''''

The GitHub repository contains a few example files. Download the following files from  `HERE <https://github.com/Rappsilber-Laboratory/xiRT/tree/master/sample_data>`_:

- DSS_xisearch_fdr_CSM50percent_minimal.csv
- xirt_params_rp.yaml
- learning_params_training_cv.yaml

This set of files can now be used to perform a RP (only) prediction on crosslink data.
To run xiRT on the data call the main function as follows after successfull installation:

.. code-block:: console

    > xirt -i DSS_xisearch_fdr_CSM50percent_minimal.csv -o xirt_results/ -x xirt_params_rp.yaml -l learning_params_training_cv.yaml



Examples
========

This section covers a few use case examples. Please check the :ref:`Parameters <linking-parameters>` section to gain
a better understanding for each of the variables.


Reversed-phase Prediction
'''''''''''''''''''''''''
While xiRT was developed for multi-dimensional RT prediction, it can also be used for a single
dimension. For this, the xiRT YAML parameter file needs to be adapted as follows::

    output:
      rp-activation: linear
      rp-column: rp
      rp-dimension: 1
      rp-loss: mse
      rp-metrics: mse
      rp-weight: 1

    predictions:
      continues:
        - rp
      # simply write fractions: [] if no fraction prediction is desired
      fractions: []

This configuration assumes that the target column in the input data is named "rp" and that the
scale is continuous (*rp-activation: linear*). If that is the case, the other parameters should
not be changed (dimension, loss, metric, weight).

2D RT Prediction - Ordinal Task
'''''''''''''''''''''''''''''''

Many studies apply a pre-fractionation method (e.g. SEC, SCX) and then measure the obtained fractions.
 For this given experimental setup, the xiRT config could look like this::

    output:
      rp-activation: linear
      rp-column: rp
      rp-dimension: 1
      rp-loss: mse
      rp-metrics: mse
      rp-weight: 1

      scx-activation: sigmoid
      scx-column: scx_ordinal
      scx-dimension: 15
      scx-loss: binary_crossentropy
      scx-metrics: mse
      scx-weight: 50

    predictions:
      continues:
        - rp
      # simply write fractions: [] if no fraction prediction is desired
      fractions: [scx]


In this config, 15 fractions (or pools) were measured. While RP prediction is modeled as regression
problem, the SCX prediction is handled as ordinal regression. This type of regression performs
classification while accounting for the magnitude of the classification errors. E.g. in a regular
classification it does not matter whether an observed PSM from fraction 5, got predicted to
elute in fraction 10 or in fraction 4. The error would only count as *false classification*.
However, in ordinal regression the margin of error is incorporated to the loss function and thus
(theoretically) ordinal regression should perform better than classification. The weight here defines 
how the losses from the two prediction tasks are added to derive the final loss. This parameter
needs to be adapted for differences in scale and type of the output.

2D RT Prediction - Classification Task
''''''''''''''''''''''''''''''''''''''

Despite the theoretical advantage of ordinal regression, classification also delivered good
results during the development of xiRT. Therefore, we kept this as an option.

For this experimental setup, the xiRT config could look like this::

    output:
      rp-activation: linear
      rp-column: rp
      rp-dimension: 1
      rp-loss: mse
      rp-metrics: mse
      rp-weight: 1

      scx-activation: softmax
      scx-column: scx_1hot
      scx-dimension: 15
      scx-loss: categorical_crossentropy
      scx-metrics: accuracy
      scx-weight: 50

    predictions:
      continues:
        - rp
      # simply write fractions: [] if no fraction prediction is desired
      fractions: [scx]

Here we have the same experimental setup as above but the scx prediction task is modeled
as classification. For classification, the activation function, column name and loss function must be
defined as in the example.

Transfer Learning
'''''''''''''''''
xiRT supports multiple types of transfer-learning. For instance,
training the exact same architecture (dimensions, sequence lengths) on a data set (e.g. BS3
crosslinked proteome) and then fine tune the learned weights on the actual data set (e.g. DSS crosslinked protein complex)
is possible.
This requires a simple change in the learning (-l parameter) config. The *pretrained_model*
parameter needs to be adapted for the location of the weights file from the BS3 model.

Additionally, the the underlying model can be changed even more. This might become necessary when the
training was done with e.g. 10 fractions but only 5 got acquired eventually. In this
scenario, the weights cannot be used from the last layers. Therefore, the *pretrained_weights* and
the *pretrained_model* parameter need to be defined in the learning (-l) config.

The files in the repository ("sample_data" and "DSS_transfer_learning_example" folder)
provide examples to achieve the transfer learning. Two calls to xiRT are necessary:

1) Train the reference model without crossvalidation:

.. code-block:: console

    >xirt -i sample_data\DSS_xisearch_fdr_CSM50percent_minimal.csv \
    -x sample_data\xirt_params_3RT_best_ordinal.yaml \
    -l sample_data\learning_params_training_nocv.yaml \
    -o models\3DRT_full_nocv

2) Use the model for transfer-learning:

.. code-block:: console

    >xirt -i sample_data\DSS_xisearch_fdr_CSM50percent_transfer_scx17to23_hsax2to9_minimal.csv \
    -x models/3DRT_full_nocv/callbacks/xirt_params_3RT_best_ordinal_scx17to23_hsax2to9.yaml \
    -l models/3DRT_full_nocv/callbacks/learning_params_training_nocv_scx17to23_hsax2to9.yaml \
    -o models\3DRT_transfer_dimensions

Further extensions
''''''''''''''''''

To further expand the tasks, two steps need to be done. First, the *predictions* section
needs to be adapted such that a list of values, for example, [scx, hsax] is supplied. Further,
each entry in the *predictions* section needs to have a matching set of entries in the *output*
section. Carefully adjust the combination of activation, loss and column parameters as shown above.
xiRT allows to have 3x regression tasks, 1x regression task + 1x classification task, etc.

In principle, the learning and prediction is agnostic to the type of input data. That means
that not only RT can be learned but also other experimentally observed properties. Simply follow
the notation and decoding of the training parameters to add other (non-liquid-chromatography) columns.

Note
''''
It is important to follow the conventions above. Otherwise learning results might vary a lot.

For classification always use the following setup:

.. code-block:: console

    output:
        scx-activation: softmax
        scx-column: scx_1hot
        scx-dimension: 15
        scx-loss: categorical_crossentropy
        scx-metrics: accuracy

For **ordinal regression** always use the following setup:

.. code-block:: console

    output:
        scx-activation: sigmoid
        scx-column: scx_ordinal
        scx-dimension: 15
        scx-loss: binary_crossentropy
        scx-metrics: mse

For **regression** always use the following setup:

.. code-block:: console

    output:
        rp-activation: linear
        rp-column: rp
        rp-dimension: 1
        rp-loss: mse
        rp-metrics: mse
