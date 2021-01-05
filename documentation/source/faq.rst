Frequently Asked Questions
==========================

1. What is xiRT?
''''''''''''''''
xiRT is a python package for multi-dimensional RT prediction for linear and cross-linked peptides.

2. How does xiRT work?
''''''''''''''''''''''
xiRT is a deep learning application and uses a Siamese network to encode crosslinked peptides.
xiRT can predict continuous and discrete retention times (e.g. from reversed phase or
fractionation experiments).

3. What are the requirements for xiRT?
''''''''''''''''''''''''''''''''''''''
xiRT requires a running python installation, please follow the installation guide to get xiRT
running. To visualize the neural network pydot and graphviz are also needed.

4. Do I need a GPU?
'''''''''''''''''''
A GPU is not necessary to use xiRT. It speeds things up but xiRT can run on any desktop computer.
Make sure to specify the correct layer in the xirt_params file (e.g. GRU instead of CudNNGRU).

5. What's the run time of xiRT?
'''''''''''''''''''''''''''''''
Depends heavily on the settings (e.g. cross-validation folds, epochs, number input PSMs). For the
example data (3-fold crossvalidation, 17k PSMs, 25 epochs) the analysis finishes within 10 minutes
on a desktop pc.

6. Where can I get help using xiRT?
'''''''''''''''''''''''''''''''''''
Please create an `GitHub issue <https://github.com/Rappsilber-Laboratory/xiRT/issues/new>`_
if we can assist you with your analysis or if anything is unclear.