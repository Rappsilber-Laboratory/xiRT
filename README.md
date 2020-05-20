# xiRT

A python package for multi-dimensional retention time prediction for linear and cross-linked 
peptides using deep neural networks.

### overview

xiRT is a deep learning tool to predict the RT of linear and cross-linked peptides.
To do that several steps are performed per default:
- tag duplicates 
- 

### input format
xiRT requires the following columns in a csv-like format.
- peptide sequence 1; "PepSeq1"
- peptide sequence 2; "PepSeq2"
- link site 1; "LinkPos1"
- link site 2; "LinkPos2"
- precursor charge; "Charge"
- score; "Score"
- cmsid; "CSMID"

The first four columns should be self explanatory, if not check the sample input *#TODO*. The fifth column ("CSMID")
is a unique integer that can be used as index. In addition, depending on the number
retention time domains that want to be learned/predicted the following columns
need to be present:

- xirt_RP
- xirt_SCX
- xirt_hSAX



