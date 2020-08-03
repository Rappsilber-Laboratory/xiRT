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
definition of the prediction task (classification, regression, ordered regression). Depending
on the decoding of the target variable the output layers need to be adapted. For standard RP
prediction, regression is essentially the only viable option. For SCX/hSAX (general classification
from fractionation experiments) the prediction task can be formulated as classification,
regression or ordered regression. For the usage of regression for fractionation it is recommended
that the estimated salt concentrations are used as target variable for the prediction  (raw
fraction numbers are possible too).