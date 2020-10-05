"""Useful script to make xiFDR output useable for xiRT."""
import argparse
import re
import sys

import pandas as pd


def rename_columns_crosslinks(psms_df):
    """
    Rename columns in the crosslink result files from xiFDR.

    Args:
        psms_df: df, dataframe with CSMs (xiFDR format required).

    Returns:
        psms_df: df, same df but with changed column names
    """
    psms_df.rename(columns={"file": "Run"}, inplace=True)
    # psms_df.rename(columns={"run": "Run"}, inplace=True)
    psms_df.rename(columns={"Score": "score"}, inplace=True)
    psms_df.rename(columns={"fdr": "FDR"}, inplace=True)
    psms_df.rename(columns={"PepSeq1": "Peptide1"}, inplace=True)
    psms_df.rename(columns={"PepSeq2": "Peptide2"}, inplace=True)
    psms_df.rename(columns={"match score": "score"}, inplace=True)
    psms_df.rename(columns={"Description1": "Fasta1"}, inplace=True)
    psms_df.rename(columns={"Description2": "Fasta2"}, inplace=True)
    psms_df.rename(columns={"MatchScore": "score"}, inplace=True)
    psms_df.rename(columns={"Actual_RT": "rp"}, inplace=True)
    psms_df.rename(columns={"ElutionStart": "rp"}, inplace=True)
    return psms_df
description = """
    xiFDR2xiRT - a convenience script to generate a minimal input for using xiRT from xiFDR data.
    
    Use python xiFDR2xiRT.py --help to see more information.
    """
parser = argparse.ArgumentParser(description=description)
parser.add_argument("-i", "--in-xifdr",
                    help="Input CSMs with assigned false discovery rate estimates.",
                    required=True, action="store", dest="in_peptides")

parser.add_argument("-o", "--out-xirt", help="Output CSM format for usage in xiRT.",
                    required=True, action="store", dest="out_peptides")

minimal_columns = ["rp", "Fasta1", "Fasta2", "score", "Peptide1", "Peptide2", "FDR",
                   "Run", "fdrGroup", "PSMID", "isTT", "isTD", "isDD",
                   "LinkPos1", "LinkPos2", "PrecoursorCharge", "Protein1",
                   "Protein2", "Crosslinker", "Decoy1", "Decoy2", "PepPos1", "PepPos2"]

# parse arguments
args = parser.parse_args(sys.argv[1:])

print("Reading file: {}".format(args.in_peptides))
# read, rename, minimize, sort
df_csms = pd.read_csv(args.in_peptides)
df_csms = rename_columns_crosslinks(df_csms)[minimal_columns]
df_csms = df_csms.loc[:, ~df_csms.columns.duplicated()]
df_csms = df_csms[sorted(df_csms.columns)]

#  transform run name to fraction
frac_annot = lambda name, prefix: int(re.search(r"{}(\d+)".format(prefix), name).groups()[0])
df_csms["scx"] = df_csms["Run"].apply(frac_annot, args=("SCX",))
df_csms["hsax"] = df_csms["Run"].apply(frac_annot, args=("hSAX",))
df_csms.to_csv(args.out_peptides, sep=",")
print("Writing file: {}".format(args.out_peptides))
