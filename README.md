![logo](documentation/imgs/xiRT_logo.png) 

![release](https://flat.badgen.net/github/release/Rappsilber-Laboratory/xirt)
![tag](https://flat.badgen.net/github/tag/Rappsilber-Laboratory/xirt)
[![GitHub](https://flat.badgen.net/github/license/Rappsilber-Laboratory/xirt)](https://www.apache.org/licenses/LICENSE-2.0)
[![Twitter](https://flat.badgen.net/twitter/follow/rappsilberlab?icon=twitter)](https://twitter.com/RappsilberLab/)
[![Python 3.8.3](https://img.shields.io/badge/python-3.8.3-blue.svg)](https://www.python.org/downloads/release/python-370/)
![PyPI version](https://flat.badgen.net/pypi/v/xiRT)
![coverage](documentation/imgs/coverage.svg)
[![Documentation Status](https://readthedocs.org/projects/xirt/badge/?version=latest)](https://xirt.readthedocs.io/en/latest/?badge=latest)
![pytest](https://github.com/Rappsilber-Laboratory/xiRT/workflows/pytest/badge.svg)

A python package for multi-dimensional retention time prediction for linear and crosslinked 
peptides using a (Siamese) deep neural network architecture.
---

- [Overview](#overview)
- [Description](#Description)
- [Installation](#Installation)

---
## overview

xiRT is a deep learning tool to predict the retention time(s) (RT) of linear and crosslinked 
peptides from multiple fractionation dimensions including reversed phase (RP, typically coupled
 online to the mass spectrometer). xiRT was developed with data from separating a complex 
 crosslinked peptide mixture by a combination of strong cation exchange (SCX) / hydrophilic 
 strong anion exchange (hSAX) / RP chromatography. However, xiRT supports all chromatographic 
 methods.

xiRT requires the columns shown in the table (“input format”) below in the input data. 
Importantly, the xiRT framework requires that crosslinked spectrum matches (CSMs) are defined 
such that for a given Peptide1 - Peptide2, Peptide1 is the longer or lexicographically larger 
one for RT predictions of crosslinked peptides.The sorting is done internally and may result in 
swapped peptide sequences in the output tables.

This readme features a quick tour through xiRT. For more information please visit the 
[documentation](https://xirt.readthedocs.io/en/latest/).


![xiRT Architecture](documentation/imgs/xiRT.PNG)

## Description
xiRT is intended to generate additional information about CSMs for machine learning-based 
rescoring frameworks but the usage can be extended to spectral libraries, targeted acquisitions
 etc. Therefore, xiRT offers several training / prediction modes that need to be configured 
 depending on the use case. At the moment training, prediction, cross-validation are the supported 
 modes.
- *training*: trains xiRT on the input CSMs (using 10% for validation) and stores a trained model
- *prediction*: use a pretrained model and predict RTs for the input CSMs
- *crossvalidation*: load/train a model and predict RTs for all data points without using them
in the training process. Requires the training of several models during cross-validation (CV).

Note: all modes can be supplemented by using a pretrained model ("transfer learning") 
when the quantity of available training data are insufficient to achieve a robust prediction 
performance.

This readme only gives a brief overview about xiRTs functions and parameters. Please refer
to the [documentation](https://xirt.readthedocs.io/en/latest/) for more details and examples.

## Installation and Usage

xiRT is a python package that comes with an executable python file. To run xiRT follow the steps 
below.

### Requirements
xiRT requires a running python installation on windows/mac/linux. All further requirements
are managed during the installation process via pip or conda. xiRT was tested using python >3.7 with
TensorFlow 1.4 and python >3.8 and TensorFlow >2.0. A GPU is not mandatory to run xiRT, however
it can greatly decrease runtime. Further system requirements depend on the data sets to be used.

### Installation
To install xiRT simply run the command below. We recommend using an isolated python environment,
for example by using pipenv **or** conda. Installation should finish within minutes.

Using pipenv:
>pip install pipenv
>pipenv shell
>und
>pip install xirt

To enable CUDA support, using a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) is the easiest solution.  
Conda will take care of the CUDA libraries and other dependencies. Note, xiRT runs either on CPUs
or GPUs. To use a GPU specify CuDNNGRU/CuDNNLSTM as type in the LSTM settings, to use a CPU set the
type to GRU/LSTM.

> conda create --name xirt_env python=3.7
>
>conda activate xirt_env
>
> pip install xirt
>
> conda install tensorflow-gpu

Hint:
pydot and graphviz sometimes cause trouble when they are installed via pip. If on linux,
simply use *sudo apt-get install graphviz*, on windows download the latest graphviz package from 
[here](https://www2.graphviz.org/Packages/stable/windows/), unzip the content of the file and add the
*bin* directory path to the windows PATH variable. These two packages allow the visualization
of the neural network architecture.

#### Usage
The command line interface (CLI) requires three inputs:
1) input PSM/CSM file
2) a [YAML](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html) file to configure the neural network architecture
3) another YAML file to configure the general training / prediction behaviour, called setup-config

To use xiRT these options are put together as shown below:
> xirt(.exe) -i peptides.csv -o out_dir -x xirt_params -l learning_params

To adapt the xiRT parameters a YAML config file needs to be prepared. The configuration file
is used to determine network parameters (number of neurons, layers, regularization) but also for the
definition of the prediction task (classification, regression, ordinal regression). Depending
on the decoding of the target variable, the output layers need to be adapted. For standard RP 
prediction, regression is essentially the only viable option. For SCX/hSAX (general classification
from fractionation experiments) the prediction task can be formulated as classification, 
regression or ordinal regression. For the usage of regression for data from a fractionation it is 
recommended that the estimated salt concentrations are used as target variable for the prediction
(raw fraction numbers are also possible).

Please find a working example / quick-start guide [here](https://xirt.readthedocs.io/en/latest/usage.html#quick-start).

### input format
| short name         | explicit column name | description                                                                    | Example     |
|--------------------|----------------------|--------------------------------------------------------------------------------|-------------|
| peptide sequence 1 | Peptide1             | First peptide sequence for crosslinks                                        | PEPRTIDER   |
| peptide sequence 2 | Peptide2             | Second peptide sequence for crosslinks, or empty                                 | ELRVIS      |
| fasta description 1        | Fasta1             | FASTA header / description of protein 1                                    | SUCD_ECOLI Succinate--CoA ligase [ADP-forming]           |
| fasta description 2        | Fasta2             | FASTA header / description of protein 2                                | SUCC_ECOLI Succinate--CoA ligase [ADP-forming]           |
| link site 1        | LinkPos1             | Crosslink position in the first peptide (0-based)                                    | 3           |
| link site 2        | LinkPos2             | Crosslink position in the second peptide (0-based                                | 2           |
| score              | score                | Single score from the search engine                                            | 17.12       |
| unique id          | PSMID                | A unique index for each entry in the result table                              | 0           |
| TT              | isTT                 | Binary column which is True for any TT identification and False for TD, DD ids | True          |
| fdr                | fdr                  | Estimated false discovery rate                                                 | 0.01        |

The first four columns should be self explanatory, for more information please
check the [sample input](https://github.com/Rappsilber-Laboratory/xiRT/tree/master/sample_data). 
The fifth column ("PSMID") is a unique(!) integer that can be used to retrieve CSMs. In addition, 
depending on the number retention time domains that should be learned/predicted the RT columns 
need to be present. The column names need to match the configuration in the network parameter yaml.
Note that xiRT swaps the sequences such that peptide1 is longer than peptide 2. In order to
keep track of this process all columns that follow the convention <prefix>1 and <prefix>2 are swapped.
Make sure to only have such paired columns and not single columns ending with either "1" or "2".

#### xiRT config
This file determines the network architecture and training behaviour used in xiRT. Please see
the [documentation](https://xirt.readthedocs.io/en/latest/parameters.html#xirt-parameters) for a 
detailed example. For crosslinks the most important parameter sections to adapt are the *output* and
the *predictions* section. Here the parameters must be adapted for the used chromatography
dimensions and modelling choices. See also the provided 
[examples](https://xirt.readthedocs.io/en/latest/usage.html#examples).

#### Setup config
This file determines the input data to be used and gives some training procedure options. Please see
the [documentation](https://xirt.readthedocs.io/en/latest/parameters.html#learning-parameters) for 
a detailed example.

### Contributors
- Sven Giese
- Ludwig Sinn

## Citation
If you consider xiRT helpful for your work please cite our manuscript. *Currently, in preparation.*

## RappsilberLab
The Rappsilber applies and develops crosslinking chemistry methods, workflows and software.
Visit the lab page to learn more about the developed [software](https://www.rappsilberlab.org/software/).

## xiSUITE
1) xiVIEW: Graham, M. J.; Combe, C.; Kolbowski, L.; Rappsilber, J. bioRxiv 2019.
2) xiNET: Combe, C. W.; Fischer, L.; Rappsilber, J. Mol. Cell. Proteomics 2015.
3) xiSPEC: Kolbowski, L.; Combe, C.; Rappsilber, J. Nucleic Acids Res. 2018, 46 (W1), W473–W478.
4) xiSEARCH: Mendes, M. L.; Fischer, L.; Chen, Z. A.; Barbon, M.; O’Reilly, F. J.; Giese, S. H.; Bohlke‐Schneider, M.; Belsom, A.; Dau, T.; Combe, C. W.; Graham, M.; Eisele, M. R.; Baumeister, W.; Speck, C.; Rappsilber, J. Mol. Syst. Biol. 2019, 15 (9), e8994.
