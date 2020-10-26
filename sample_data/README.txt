Data sets
#########
1) BS3_xisearch_fdr_CSM50percent.txt - BS3 crosslinked E. coli lysate searched with xiFDR
2) DSS_xisearch_fdr_CSM50percent.txt - DSS crosslinked E. coli lysate searched with xiFDR
3) DSS_xisearch_fdr_CSM50percent_transfer_scx17to23_hsax2to10.csv - same as 2 but peptides
identified in the scx fractions 24 and 25 have been removed, as well as the peptides in hsax 1, 10

xiRT-params
###########
1) xirt_grid.yaml - a convenience file that can be used to generate a set of different yaml configurations. This file cannot be used as direct input.
2) xirt_params_3RT.yaml -
3) xirt_params_3RT_best_classification.yaml - Full sample yaml for 3D RT and classification for fractionation tasks.
4) xirt_params_3RT_best_ordinal.yaml - Full sample yaml for 3D RT and ordinal regression for fractionation tasks.
5) xirt_params_3RT_best_ordinal_pseudolinear.yaml - pseudo linear formatting of crosslinked peptides (concatenation of the two individual peptides)
6) xirt_params_3RT_best_ordinal_scx17to23_hsax2to10.yaml - adapted config from 4) for changed dimensions of the fractionation.
7) xirt_params_3RT_best_regression.yaml - Full sample yaml for 3D RT and regression for fractionation tasks.
