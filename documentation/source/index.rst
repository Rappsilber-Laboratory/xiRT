Welcome to xiRT's documentation!
================================

.. image:: xiRT_logo.png

xiRT is a versatile python package for multi-dimensional retention time prediction
for linear and crosslinked peptides.

xiRT requires identified peptide sequences with an assigned confidence (FDR) to learn the retention
behavior from multiple dimensions. The high confidence identifications are necessary to reduce
the noise in the data which allows more accurate retention time prediction. However, typically
we want to supply higher FDR (>1%) data to also predict the retention times for peptide spectrum
matches where the search score was not sufficient for passing the FDR cutoff. Post-search validation
algorithms such as `percolator <http://percolator.ms/>`_ can then be used to rescore the given set
of PSMs with the predicted retention times.

**Approach.**

xiRT uses a deep neural network architecture to realize the simultaneous learning for multiple
retention times. In brief, xiRT builds a multi-layer network that can be divided into a Siamese part
and individual task subnetworks. The Siamese part takes the peptide sequences as input and applys
and Embedding and Recurrent function to the input. For linear peptides the output of the
recurrent layer is directly forwarded to the task subnetworks. For crosslinked peptides, each
peptide has it's own input and after the recurrent layer the two outputs are first combined and then
passed towards the individual task networks. In contrast, to typical regression models the input
data (peptide) sequences are not transformed into features but rather the entire peptide
sequence including modifications is used as input.

**Supported Prediction Tasks**

xiRT is versatile in the input and experimental design. An arbitrary number of prefractionation
methods are supported as well as a standard reversed phase RT prediction.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   readme
   installation
   usage
   parameters
   modules
   development


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


