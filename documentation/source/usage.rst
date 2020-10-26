Usage
=====
The command line interface (CLI) requires three inputs:

1) input PSM/CSM file
2) a `YAML <https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html>`_ file to configure the neural network architecture
3) another YAML file to configure the general training / prediction behaviour, called setup-config

To use xiRT these options are put together as shown below::

> xirt(.exe) -i peptides.csv -o out_dir -x xirt_params.yaml -l learning_params.yaml

To adapt the xiRT parameters a yaml config file needs to be prepared. The configuration file
is used to determine network parameters (number of neurons, layers, regularization) but also for the
definition of the prediction task (classification, regression, ordinal regression). Depending
on the decoding of the target variable the output layers need to be adapted. For standard RP
prediction, regression is essentially the only viable option. For SCX/hSAX (general classification
from fractionation experiments) the prediction task can be formulated as classification,
regression or ordinal regression. For the usage of regression for fractionation it is recommended
that the estimated salt concentrations are used as target variable for the prediction  (raw
fraction numbers are possible too).

Examples
========

This section covers a few use case examples. Please check the :ref:`Parameters <linking-parameters>` section to gain
a better understanding for each of the variables.

Reversed-phase Prediction
*************************
While xiRT was developed for multi-dimensional RT prediction it can also be used for single
domains. For this, the xiRT YAML parameter file needs to be adapted as follows::

    output:
      rp-activation: linear
      rp-column: rp
      rp-dimension: 1
      rp-loss: mse
      rp-metrics: mse
      rp-weight: 1

    predictions:
      continues:
        - RP
      # simply write fractions: [] if no fraction prediction is desired
      fractions: []

This configuration assumes that the target column in the input data is named "RP" and that the
scale is continuous (*rp-activation: linear*). If that is the case, the other parameters should
not be changed (dimension, loss, metric, weight).

2D RT Prediction - Ordinal Task
*******************************

Many studies simply apply one pre-fractionation method (e.g. SEC, SCX) and then acquire the samples.
For this experimental setup, the xiRT config could look like this::

    output:
      rp-activation: linear
      rp-column: rp
      rp-dimension: 1
      rp-loss: mse
      rp-metrics: mse
      rp-weight: 1

      scx-activation: sigmoid
      scx-column: scx-ordinal
      scx-dimension: 15
      scx-loss: binary_crossentropy
      scx-metrics: mse
      scx-weight: 50

    predictions:
      continues:
        - RP
      # simply write fractions: [] if no fraction prediction is desired
      fractions: [scx]


In this config, 15 fractions (or pools) were acquired. While RP prediction is modeled as regression
problem the SCX prediction is handled as ordinal regression. This type of regression performs
classification but the magnitude of the classification errors is taken into account. E.g. in normal
classification it does not make a difference if an observed PSM in fraction 5, got predicted to
elude in fraction 10 or in fraction 4. The error would only count as *false classification*.
However, in ordinal regression the margin of error is incorporated to the loss function and thus
(theoretical) ordinal regression should perform better than classification. The weight defines here
how the losses from the two prediction tasks are added to derive the final loss. This parameter
needs to be adapted for differences in scale and type of the output.

2D RT Prediction - Classification Task
**************************************

Despite the theoretical advantage of ordinal regression, classification also delivered good
results during the development of xiRT. Therefore, the option can still be used.

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
        - RP
      # simply write fractions: [] if no fraction prediction is desired
      fractions: [scx]

Here we have the same experimental setup as above, but the scx prediction task is modeled
as classification. For classification the activation, column and loss must be defined as in the
example.

Transfer Learning
*************************
xiRT supports multiple types of transfer-learning capabilities. For instance,
training the exact same architecture (dimensions, sequence lengths) on a data set (e.g. BS3 crosslinked)
and then fine tune the learned weights on the actual data set (e.g. DSS crosslinked). This requires
a simple change in the learning (-l parameter) config. The *pretrained_model* parameter needs to be adapted
for the location of the weights file from the BS3 model.
Another option is to change the underlying even more. This might be necessary when the training
data was trained with e.g. 10 fractions but only 5 got acquired in the new acquisition. In this
scenario the weights cannot be used from the last layers. Therefore, the *pretrained_weights* and the *pretrained_model*
parameter need to be given in the learning (-l) config.

The files in the repository ("sample_data" and "models" folder) provide examples to achieve the
transfer learning. Two calls to xiRT are necessary:

**Example:**

``# train the full model, dont do crossvalidation!``

``>xirt -i sample_data\DSS_xisearch_fdr_CSM50percent.csv
-x sample_data\xirt_params_3RT_best_ordinal.yaml
>-l sample_data\learning_params_training_nocv.yaml
-o models/3DRT_full_nocv``

``# use the already trained architecture``

``>xirt -i sample_data\DSS_xisearch_fdr_CSM50percent_transfer_scx17to23_hsax2to9.csv
-x sample_data\xirt_params_3RT_best_ordinal_scx17to23_hsax2to9.yaml
-l sample_data\learning_params_training_nocv_scx17to23_hsax2to9.yaml
-o models\3DRT_transfer_dimensions``

Further extensions
******************

To further expand the tasks, 2 steps need to be done. First, the *predictions* section
needs to be adapted such that a list of values, for example, [scx, hsax] is supplied. Further,
each entry in the *predictions* section needs to have a matching set of entries in the *output*
section. Carefully adjust the combination of activation, loss and column parameters as shown above.
xiRT allows to have 3x regression tasks, 1x regression task + 1x classification task, etc.

In principle the learning and prediction is agnostic to the kind of input data. That means
that not only RT can be learned but also other experimentally observed properties. Simply follow
the notation and decoding of the training parameters to add non-liquid-chromatography columns.

Example Usage
*************

The github repository contains a few example files. Download the following files from  `HERE <https://github.com/Rappsilber-Laboratory/xiRT/tree/master/sample_data>`_:

- DSS_xisearch_fdr_CSM50percent.csv
- xirt_params_rp.yaml
- learning_params_training_cv.yaml

This set of files can now be used to perform a RP prediction on crosslink data.
To run xiRT on the data call the main function as follows after successfull installation::

> xirt(.exe) -i DSS_xisearch_fdr_CSM50percent.csv -o xirt_results/ -x xirt_params_rp.yaml -l learning_params_training_cv.yaml


Note
****
It is important to follow the conventions above. Otherwise learning results can vary a lot.

For **classification** always use the following setup:

      scx-activation: softmax
      scx-column: scx_1hot
      scx-dimension: 15
      scx-loss: categorical_crossentropy
      scx-metrics: accuracy

For **ordinal regression** always use the following setup:

      scx-activation: sigmoid
      scx-column: scx_ordinal
      scx-dimension: 15
      scx-loss: binary_crossentropy
      scx-metrics: mse

For **regression** always use the following setup:

      scx-activation: linear
      scx-column: scx
      scx-dimension: 15
      scx-loss: mse
      scx-metrics: mse

