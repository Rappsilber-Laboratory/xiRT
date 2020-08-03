xiRT - Introduction
===================

xiRT is a deep learning tool to predict the retention times(s) of linear and crosslinked peptides
from multiple fractionation dimensions including RP (typically coupled to the mass spectrometer).
xiRT was developed with a combination of SCX / hSAX / RP chromatography. However, xiRT supports
all available chromatography methods.

xiRT requires the columns shown in the table below. Importantly, the xiRT framework requires that
CSM are sorted such that in the Peptide1 - Peptide2, Peptide1 is the longer or lexicographically
larger one for crosslinked RT predictions.

Description
***********

xiRT is meant to be used to generate additional information about CSMs for machine learning-based
rescoring frameworks (similar to percolator). However, xiRT also delivers RT prediction for various
scenarios. Therefore xiRT offers several training / prediction  modes that need to be configured
depending on the use case. At the moment training, prediction, crossvalidation are the supported
modes.
- *training*: trains xiRT on the input CSMs (using 10% for validation) and stores a trained model
- *prediction*: use a pretrained model and predict RTs for the input CSMs
- *crossvalidation*: load/train a model and predict RTs for all data points without using them
in the training process. Requires the training of several models during CV

Note: all modes can be supplemented by using a pretrained model ("transfer learning").