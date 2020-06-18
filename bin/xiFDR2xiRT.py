import pandas as pd
import re


def rename_columns_crosslinks(psms_df):

    psms_df.rename(columns={"file": "Run"}, inplace=True)
    #psms_df.rename(columns={"run": "Run"}, inplace=True)
    psms_df.rename(columns={"Score": "score"}, inplace=True)
    psms_df.rename(columns={"fdr": "FDR"}, inplace=True)
    psms_df.rename(columns={"PepSeq1": "Peptide1"}, inplace=True)
    psms_df.rename(columns={"PepSeq2": "Peptide2"}, inplace=True)
    psms_df.rename(columns={"match score": "score"}, inplace=True)
    psms_df.rename(columns={"Description1": "Fasta1"}, inplace=True)
    psms_df.rename(columns={"Description2": "Fasta2"}, inplace=True)
    psms_df.rename(columns={"MatchScore": "score"}, inplace=True)
    psms_df.rename(columns={"Actual_RT": "xirt_RP"}, inplace=True)
    return(psms_df)


def annotate_fraction(name, prefix="SCX"):
    """
    Uses a regular expression to retrieve the fraction number.

    Parameters:
    -----------
    name: str,
            filename of the raw file (with fraction encoding in the name)
    prefix: str,
            prefix for the annotation of the RT

    Note:
    ----
    To successfully extract the fractions a defined format in the run name is required.
    E.g. ]: 'XXXXXXX_SCX20_hSAX06_XXXX' is a valid format.
    """
    return(int(re.search("{}(\d+)".format(prefix), name).groups()[0]))


minimal_columns = ["xirt_RP", "Fasta1", "Fasta2", "score", "Peptide1", "Peptide2", "FDR",
                   "Run", "FDR", "fdrGroup", "PSMID", "isTT", "LinkPos1", "LinkPos2"]
df_csms = pd.read_csv(r"C:\\Users\\Hanjo\\Documents\\xiRT\\tests\\fixtures\\50pCSMFDR_universal.csv")
df_csms = rename_columns_crosslinks(df_csms)[minimal_columns]
df_csms = df_csms.loc[:,~df_csms.columns.duplicated()]
df_csms = df_csms[sorted(df_csms.columns)]

df_csms["xirt_SCX"] = df_csms["Run"].apply(annotate_fraction, args=("SCX",))
df_csms["xirt_hSAX"] = df_csms["Run"].apply(annotate_fraction, args=("hSAX",))
df_csms.to_csv(r"C:\\Users\\Hanjo\\Documents\\xiRT\\tests\\fixtures\\50pCSMFDR_universal_final.csv")
