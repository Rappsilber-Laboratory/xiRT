[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
![coverage](./documentation/coverage.svg)

# xiRT

A python package for multi-dimensional retention time prediction for linear and cross-linked 
peptides using a (siamese) deep neural network architecture. Figure 1 shows a simplified 
network architecture.

### overview

xiRT is a deep learning tool to predict the RT of linear and cross-linked peptides. xiRT requires 
columns shown in the table below. Importantly, the xiRT framework requires that CSM are sorted
such that Peptide1 - Peptide2 pairs are defined by peptide length / lexicographically. 

### input format
| short name         | explicit column name | description                                                                    | Example     |
|--------------------|----------------------|--------------------------------------------------------------------------------|-------------|
| peptide sequence 1 | Peptide1             | Alpha peptide sequence for crosslinks                                        | PEPRTIDER   |
| peptide sequence 2 | Peptide2             | Beta peptide sequence for crosslinks, or empty                                 | ELRVIS      |
| link site 1        | LinkPos1             | Crosslink position in the peptide (0-based)                                    | 3           |
| link site 2        | LinkPos2             | Crosslink position in the beta peptide (0-based                                | 2           |
| precursor charge   | Charge               | Precursor charge of the crosslinked peptide                                    | 3           |
| score              | Score                | Single score from the search engine                                            | 17.12       |
| unique id          | CSMID                | A unique index for each entry in the result table                              | 0           |
| decoy              | isTT                 | Binary column which is True for any TT identification and False for TD, DD ids | TT          |
| fdr                | fdr                  | Estimated false discovery rate                                                 | 0.01        |
| fdr level          | fdrGroup             | String identifier for heteromeric and self links (splitted FDR)                | heteromeric |

The first four columns should be self explanatory, if not check the sample input *#TODO*. 
The fifth column ("CSMID") is a unique integer that can be used as index. In addition, 
depending on the number retention time domains that want to be learned/predicted the 
following columns need to be present:

- xirt_RP
- xirt_SCX
- xirt_hSAX

### config file
To adapt the xiRT parameters a yaml config file needs to be prepared.

### Installation & Usage
To install xiRT a pip package is under development. Future release can be installed via:
>pip install xirt

To enable CUDA support, the specific libraries need to be installed manually. The python 
dependencies are covered via the pip installation.

To use xiRT a simple command line script gets installed with pip. To run the predictions ...


