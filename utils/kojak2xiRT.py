"""Script to convert kojak result files to be used with xiRT"""
# description = \
# """
# kojak2xiFDR - a convenience script to generate a minimal input for using xiRT from xiFDR data.
# Use python xiFDR2xiRT.py --help to see more information.
# """
# parser = argparse.ArgumentParser(description=description)
# parser.add_argument("-i", "--in-xifdr",
#                     help="Input CSMs with assigned false discovery rate estimates.",
#                     required=True, action="store", dest="in_peptides")
#
# parser.add_argument("-o", "--out-xirt", help="Output CSM format for usage in xiRT.",
#                     required=True, action="store", dest="out_peptides")
#
# minimal_columns = ["rp", "Fasta1", "Fasta2", "score", "Peptide1", "Peptide2", "FDR",
#                    "Run", "fdrGroup", "PSMID", "isTT", "isTD", "isDD",
#                    "LinkPos1", "LinkPos2", "PrecoursorCharge", "Protein1",
#                    "Protein2", "Crosslinker", "Decoy1", "Decoy2", "PepPos1", "PepPos2"]

# parse arguments
#args = parser.parse_args(sys.argv[1:])
#%%
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

directory = r"C:\Users\Sven.Giese\Desktop\ECCP\kojak_sample_files"
directory = r""
files = glob.glob(os.path.join(directory, "*.kojak.txt"))
dfs = []
for file in files:
    print(file)
    df_temp = pd.read_csv(file, skiprows=1, sep="\t")
    df_temp["Run"] = os.path.basename(file)
    df_temp = df_temp[(df_temp["Obs Mass"] != 0) &
                      (df_temp["Peptide #2"] != "-")]
    dfs.append(df_temp)
df_kojak = pd.concat(dfs)
df_kojak.rename(columns={"Charge": "PrecursorCharge",
                         "Score": "score",
                         "Ret Time": "rp",
                         "Linked AA #1": "LinkPos1",
                         "Linked AA #2": "LinkPos2",
                         "Protein #1": "Description1",
                         "Protein #2": "Description2",
                         "Peptide #1": "Peptide1",
                         "Peptide #2": "Peptide2",
                         "Protein #1 Site": "PepPos1",
                         "Protein #2 Site": "PepPos2"}, inplace=True)
print("Finished reading. Start processint dataframe.")
df_kojak["PSMID"] = np.arange(1, len(df_kojak) + 1)
df_kojak["Protein1"] = df_kojak["Description1"]
df_kojak["Protein2"] = df_kojak["Description2"]
df_kojak["Decoy1"] = df_kojak["Description1"].str.startswith("DEC")
df_kojak["Decoy2"] = df_kojak["Description2"].str.startswith("DEC")
df_kojak["isTT"] = df_kojak[["Decoy1", "Decoy2"]].sum(axis=1)
df_kojak["score"] = df_kojak["score"].astype(float) * 1.0
rec = re.compile(r"((?:DEC_sp|sp)\|\w+\|)")
df_kojak["Protein1"] = [";".join(re.findall(rec, i)) for i in df_kojak["Description1"]]
df_kojak["Protein2"] = [";".join(re.findall(rec, i)) for i in df_kojak["Description2"]]
#  transform run name to fraction
frac_annot = lambda name, prefix: int(re.search(r"{}(\d+)".format(prefix), name).groups()[0])

df_kojak["scx"] = df_kojak["Run"].apply(frac_annot, args=("SCX",))
df_kojak["hsax"] = df_kojak["Run"].apply(frac_annot, args=("hSAX",))
df_kojak.to_csv(r"all_kojak.csv", sep=",")

