"""This util module allows the conversion from pLink to be used in xiFDR/xiRT."""
import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyteomics import mgf, parser, fasta
from tqdm import tqdm

pd.set_option('display.max_columns', 20)
frac_annot = lambda name, prefix: int(re.search(r"{}(\d+)".format(prefix), name).groups()[0])


def add_col_c(x):
    """Add enumeration for datafram x to column col_c."""
    x["col_c"] = np.arange(len(x))
    return x


def get_info_duplicates(df_plink_o):
    """Print information on aggregated results."""
    print("pLink2 # spectra by a 'unique' filter")
    print(df_plink_o.shape[0])
    print("Peptide")
    print(df_plink_o.drop_duplicates(["Peptide"]).shape[0])
    print("Peptide-Charge")
    print(df_plink_o.drop_duplicates(["Peptide", "Charge"]).shape[0])
    print("Peptide1-Peptide2")
    print(df_plink_o.drop_duplicates(["Peptide1", "Peptide2"]).shape[0])
    print("Peptide1-Peptide2-Charge")
    print(df_plink_o.drop_duplicates(["Peptide1", "Peptide2", "Charge"]).shape[0])
    print("Peptide1-Peptide2-Positions")
    print(df_plink_o.drop_duplicates(["Peptide1", "Peptide2", "LinkPos1", "LinkPos2"]).shape[0])
    print("Peptide1-Peptide2-Charge+LinkPositions")
    print(df_plink_o.drop_duplicates(
        ["Peptide1", "Peptide2", "Charge", "LinkPos1", "LinkPos2"]).shape[0])


def get_fasta_df(fastaf):
    """
    Get a fasta dataframe (peptide-protein).

    Parameters:
        fastaf: str, location of fasta.
    """
    print('Cleaving the proteins with trypsin...')

    peptides = []
    proteins = []
    # standard regex
    # r'([KR](?=[^P]))|((?<=W)K(?=P))|((?<=M)R(?=P))',
    # allow cleavage before P
    tryp_regex = '([KR])|((?<=W)K(?=P))|((?<=M)R(?=P))'
    with open(fastaf, mode='rt') as gzfile:
        for description, sequence in tqdm(fasta.FASTA(gzfile)):
            # two trypsin rules
            new_peptides1 = parser.cleave(sequence, 'trypsin', missed_cleavages=2)
            new_peptides2 = parser.cleave(sequence, tryp_regex, 2)
            new_peptides = list(set(new_peptides1) | set(new_peptides2))
            peptides.extend(new_peptides)
            proteins.extend([description.split()[0]] * len(new_peptides))

            # for peptides where m is cleaved
            if sequence.startswith("M"):
                new_peptides_m3 = parser.cleave(sequence[1:], 'trypsin', missed_cleavages=2)
                new_peptides_m4 = parser.cleave(sequence[1:], tryp_regex, missed_cleavages=2)
                new_peptides_m = list(set(new_peptides_m3) | set(new_peptides_m4))
                peps_m = set(new_peptides_m) - set(new_peptides)
                peptides.extend(peps_m)
                proteins.extend([description.split()[0]] * len(peps_m))
    df_fasta = pd.DataFrame()
    df_fasta["Peptide"] = peptides
    df_fasta["Proteins"] = proteins
    df_fasta = df_fasta.set_index("Peptide")

    df_fasta = df_fasta.reset_index()
    df_fasta = df_fasta.groupby(["Peptide"], as_index=False).agg({"Proteins": ";".join})
    df_fasta = df_fasta.set_index("Peptide")
    return df_fasta


def get_protein_dic(fastaf):
    """
    Get a <protein>:<sequence> dictionary from a fasta file.

    Parameters:
        fastaf: str, location of fasta
    """
    proteins_dic = {}
    with open(fastaf, mode='rt') as gzfile:
        for description, sequence in tqdm(fasta.FASTA(gzfile)):
            proteins_dic[description.split()[0]] = sequence
            # proteins_dic["|".join(description.split()[0].split("|")[0:2])] = sequence
    return proteins_dic


# adjust for multiple proteins
def get_positions(peptide, proteins, proteins_dic):
    """
    Get a string of peptide positions.

    Parameters:
    peptide: str, peptide
    proteins: str,proteins sep by ;
    proteins_dic: dict, <protein>:<sequence>
    """
    return ";".join(
        [str(proteins_dic[protein_i].index(peptide) + 1) for protein_i in proteins.split(";")])


def mgf2csv(filelist):
    """
    Function that extracts MS2 information from an mzML/mgf file and stores
    the results in a handy dataframe.

    Extracted information for MS2:
        - MS2: scan, rt
        - precursor: mz, int, mass, charge
        - meta: machine, raw file
    """
    data = {"RT": [], "TITLE": []}
    for filename_in in tqdm(files):
        with mgf.read(filename_in) as reader:
            for spectrum in reader:
                data["RT"].append(spectrum["params"]["rtinseconds"].real)
                data["TITLE"].append(spectrum["params"]["title"])
    data = pd.DataFrame(data)
    data.set_index("TITLE")
    return (data)


def create_modseq(df_plink_cl):
    """
    Create a modified peptide sequence string

    """

    # make mod detection easier
    df_plink_cl["Modifications"] = df_plink_cl["Modifications"].fillna("")

    # psm_df = df_plink_cl[df_plink_cl["Modifications"] != ""].head(1000)
    # map common modifications
    mod_dic = {"M": "ox", "C": "cm"}
    modex = re.compile(r"\[(\w)\]\((\d+)\)")
    modsseqs1 = [""] * len(df_plink_cl)
    modsseqs2 = [""] * len(df_plink_cl)

    # compute the mod seqecne column
    idx = -1
    for ii, row in tqdm(df_plink_cl.iterrows(), total=len(df_plink_cl)):
        idx += 1
        # get modifications
        mods = row["Modifications"]

        if mods == "":
            # nod mods
            modsseqs1[idx] = row.Peptide1
            modsseqs2[idx] = row.Peptide2
            continue
        else:
            # modifications list with (aa, pos) tuples
            # go reverse because otherwise the indices dont match after the
            # first insertion
            pepseq1 = row.Peptide1
            pepseq2 = row.Peptide2
            modified1 = False
            modified2 = False
            modes_ar = re.findall(modex, mods)[::-1]
            for mod in modes_ar:
                modpos = int(mod[1])

                # mods on peptide 1
                if modpos <= row["PepLength1"]:
                    pepseq1 = pepseq1[:modpos] + mod_dic[mod[0]] + pepseq1[modpos:]
                    modified1 = True
                else:
                    # -3 for (, ), - in the peptide string?!
                    modpos_adj = modpos - row.PepLength1 - 3
                    # mods on peptide 2
                    pepseq2 = pepseq2[:modpos_adj] + mod_dic[mod[0]] + pepseq2[modpos_adj:]
                    modified2 = True

        # check if there were changes and if not, write the unmodified sequence
        if modified1:
            modsseqs1[idx] = pepseq1
        else:
            modsseqs1[idx] = row.Peptide1

        if modified2:
            modsseqs2[idx] = pepseq2
        else:
            modsseqs2[idx] = row.Peptide2

    df_plink_cl["ModSeq1"] = modsseqs1
    df_plink_cl["ModSeq2"] = modsseqs2


# get peptide1,2,linkpos
def split_peptides(df_plink_cl):
    regex = re.compile(r"(\w+)\((\d+)\)-(\w+)\((\d+)\)")
    df_info = df_plink_cl["Peptide"].str.extract(regex)
    df_info.columns = ["Peptide1", "LinkPos1", "Peptide2", "LinkPos2"]
    df_plink_cl = df_plink_cl.join(df_info)
    return df_plink_cl


def split_proteins(df_plink_cl):
    regex = re.compile(r"(\S+) \((\d+)\)-(\S+) \((\d+)\)")
    df_info = df_plink_cl["Proteins"].str.extract(regex)
    df_info.columns = ["Protein1", "PepPos1", "Protein2", "PepPos2"]
    df_plink_cl = df_plink_cl.join(df_info)
    return df_plink_cl


def decoy_type_annotation(df_plink_cl):
    # Target_Decoy: the identification is target or decoy.
    # 0 for Decoy-Decoy,
    # 1 for Target-Decoy (or Decoy-Target), and
    # 2 for Target-Target.
    TD_dic = {2: "TT", 1: "TD", 0: "DD"}
    df_plink_cl["IDType"] = df_plink_cl["Target_Decoy"].map(TD_dic)
    df_plink_cl["isTT"] = df_plink_cl["IDType"] == "TT"
    df_plink_cl["isTD"] = df_plink_cl["IDType"] == "TD"
    df_plink_cl["isDD"] = df_plink_cl["IDType"] == "DD"


def group_annotation(df_plink_cl):
    # convert within / between
    # Protein_Type: the same as the Protein_Type in spectra level described above,
    # but with 0 for Regular/Common, 1 for Intra-protein, and 2 for Inter-protein.
    type_dict = {1: "within", 2: "between"}
    df_plink_cl["Group"] = [type_dict[t] for t in df_plink_cl["Protein_Type"]]


def get_protein_short(prot):
    """Get short protein identifier to work with xiFDR."""
    return (";".join([i.split("|")[0] for i in prot.replace("sp|", "").split(";")]))


def plot_diag(test):
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4))
    ax1.plot(test["Score"], test["Q-value"])
    ax1.set(xlabel="Score", ylabel="Q-value")

    ax2.plot(test["SVM_Score"], test["Q-value"])
    ax2.set(xlabel="SVM_Score", ylabel="Q-value")

    ax3.plot(test["Refined_Score"], test["Q-value"])
    ax3.set(xlabel="Refined_Score", ylabel="Q-value")

    ax4.scatter(test["SVM_Score"], test["Score"])
    ax4.set(xlabel="SVM_Score", ylabel="Score")
    plt.tight_layout()
    plt.show()

    x = df_self["IDType"].value_counts()
    f, ax = plt.subplots()
    ax.bar([0, 1, 2], x.values)
    ax.set(xticks=[0, 1, 2], xticklabels=x.index.values)
    sns.despine()
    plt.show()


def plink2xifdr(df_plink_input, data, filtered=False):
    print(df_plink_input.shape)
    print("Mapping pLink and RT data...")
    df_plink_input = df_plink_input.merge(data, left_on="Title", right_index=True)
    df_plink_input["rp"] = df_plink_input["RT"] / 60.

    print("Reorganize Peptide/protein storage...")
    df_plink_input = split_peptides(df_plink_input)
    df_plink_input = split_proteins(df_plink_input)
    df_plink_input["PepLength1"] = df_plink_input["Peptide1"].apply(len)
    df_plink_input["PepLength2"] = df_plink_input["Peptide2"].apply(len)

    if filtered:
        # filtered only contains TT
        df_plink_input["Target_Decoy"] = 2
        df_plink_input["Protein_Type"] = [1 if i != j else 2 for i, j in
                                          zip(df_plink_input["Protein1"],
                                              df_plink_input["Protein2"])]
    print("Decoy Annotation..")
    # isTT, isTD, columns
    decoy_type_annotation(df_plink_input)

    print("Group Annotation..")
    # self, between annotation
    group_annotation(df_plink_input)

    # target / decoy proteins
    df_plink_input = df_plink_input.merge(df_fasta, left_on="Peptide1", right_index=True,
                                          suffixes=("", "_1"))
    df_plink_input = df_plink_input.merge(df_fasta, left_on="Peptide2", right_index=True,
                                          suffixes=("", "_2"))

    # some weird list formatting was introduce .. reavel to single entries
    df_plink_input["Protein1"] = np.ravel(df_plink_input.Proteins_1)
    df_plink_input["Protein2"] = np.ravel(df_plink_input.Proteins_2)
    df_plink_input["Protein1_short"] = df_plink_input["Protein1"].apply(get_protein_short)
    df_plink_input["Protein2_short"] = df_plink_input["Protein2"].apply(get_protein_short)

    df_plink_input["Decoy1"] = df_plink_input["Protein1"].str.contains("REVERSE")
    df_plink_input["Decoy2"] = df_plink_input["Protein2"].str.contains("REVERSE")

    # create the modification seqeuences
    create_modseq(df_plink_input)

    # peptide positions
    pep_pos1 = np.array([get_positions(pep, prot, proteins_dic) for prot, pep in
                         zip(df_plink_input["Protein1"], df_plink_input["Peptide1"])])

    pep_pos2 = np.array([get_positions(pep, prot, proteins_dic) for prot, pep in
                         zip(df_plink_input["Protein2"], df_plink_input["Peptide2"])])

    # assign to dataframe, these are peptide positions! not protein link positions
    df_plink_input["PSMID"] = np.arange(0, len(df_plink_input))
    df_plink_input["PepPos1"] = pep_pos1
    df_plink_input["PepPos2"] = pep_pos2
    df_plink_input["Description1"] = df_plink_input.Proteins_1
    df_plink_input["Description2"] = df_plink_input.Proteins_2
    print(df_plink_input.shape)
    return df_plink_input


# %%
# =============================================================================
# config
# =============================================================================
# how to
# 1. adapt the "dir" parameter, thats where the MGFs are located
# 2. specify the plink input (tested with the unfiltered results file)
# 3. specify the FASTA file from pLink2 output
#
# Step 1 is needed to get the mz / RT information to the pLink2 data.
# parsing the >100 files takes a bit of time so the data is saved (line 368)
# by uncommenting the block below after the first execution this step can
# be skipped.


args = {}
# mgf dir
args["dir"] = ""
args["extension"] = "mgf"
# get list of mgfs
files = list(glob.iglob("{}/recal*.{}".format(args["dir"], args["extension"])))

plink_input = r""
plink_fasta = r"*_comb_rever.fasta"
output = plink_input.replace(".csv", "_xifdr.csv")

print("Running with the following params:")
for key, value in args.items():
    print(key, ":", value)

# process input data
df_plink = pd.read_csv(plink_input)
#need mgfs for reannotation
# alternatively
# grep 'TITLE\|RTINSECONDS' B181121_09_HF_FW_IN_130_ECLP_DSS01_SCX19_hSAX02_rep2.mgf | awk '{ORS=NR % 2? " ": "\n";print}'
# get CSV data and write table
data = mgf2csv(filelist=files)
data = data.set_index("TITLE")
data.to_csv(args["dir"] + "_MS2_summary_data_newrecal.csv")
data.to_pickle(args["dir"] + "_MS2_summary_data_newrecal.p")
print("File written to {}".format(args["dir"] + args["name"] + "_MS2_summary_data.csv"))
df_plink_output = plink2xifdr(df_plink, data, filtered=False)
df_plink_output.to_csv(output)
