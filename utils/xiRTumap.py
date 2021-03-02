"""Utility stuff for umap plotting and annotation of the psms. Not well tested."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib import ticker
from pyteomics import fasta
from sklearn.preprocessing import StandardScaler


def annotate_organism(psms_df, fasta_file):
    """
    Annotate the organism on the psms.
    """
    fa_reader = fasta.read(fasta_file)
    uniprot_osmap = {fasta.parse(seqs[0])["id"]: fasta.parse(seqs[0])["OS"] for seqs in
                     fa_reader}
    os1 = []
    os2 = []
    os_both = []
    os_str = []

    for rowid, row in psms_df.iterrows():
        # replace decoy prefix, and make list from string. Multiple proteins are separated by ";"
        proteins = row.Protein1.replace("decoy:", "").split(";")
        # get organism from ID
        organisms1 = np.sort(np.unique([uniprot_osmap[i] for i in proteins]))
        organisms1.sort()
        # check if row was decoy
        if row.Decoy1:
            append_str1 = ";".join(organisms1) + "(D)"
        else:
            append_str1 = ";".join(organisms1) + "(T)"
        os1.append(append_str1)

        proteins = row.Protein2.replace("decoy:", "").split(";")
        organisms2 = np.sort(np.unique([uniprot_osmap[i] for i in proteins]))
        organisms2.sort()
        if row.Decoy2:
            append_str2 = ";".join(organisms2) + "(D)"
        else:
            append_str2 = ";".join(organisms2) + "(T)"
        os2.append(append_str2)
        os_both.append(";".join(np.unique(np.concatenate([[append_str1], [append_str2]]))))

        if ("Escherichia coli" in os_both[-1]) and ("Homo sapiens" in os_both[-1]):
            os_str.append("Mix")

        elif "Escherichia coli" not in os_both[-1]:
            os_str.append("Homo sapiens")

        elif "Homo sapiens" not in os_both[-1]:
            os_str.append("Escherichia coli")
        else:
            os_str.append("Ups...")

    psms_df["OS1"] = os1
    psms_df["OS2"] = os2
    psms_df["OS1+2"] = os_both
    psms_df["OS"] = os_str
    return psms_df


def annotate_decoytype(psms_df):
    """
    Summarieze isTT,isTD,isDD to a single column.
    Returns
    -------

    """
    cols = np.array(["isTT", "isTD", "isDD"])
    names = np.array(["TT", "TD", "DD"])
    psms_df["DecoyType"] = names[np.argmax(psms_df[cols].values, axis=1)]
    return psms_df


def plot_embedding(em_df, subtitle):
    palette = {"TT_Escherichia coli": "#66c2a4", "TD_Escherichia coli": "#e34a33",
               "TT_Mix": "#fc8d59", "TD_Mix": "#fdbb84"}

    print(em_df["grp"].value_counts())
    fs = []
    xlim = (em_df["umap1"].min(), em_df["umap1"].max())
    ylim = (em_df["umap2"].min(), em_df["umap2"].max())
    for grp, grpdf in em_df.groupby("fdrGroup"):
        jax = sns.jointplot(x="umap1", y="umap2", data=grpdf, hue="grp", marker="+", s=100,
                            palette=palette, hue_order=list(palette.keys()))
        jax.ax_marg_x = sns.histplot(x="umap1", data=grpdf, hue="grp", element="step",
                                     ax=jax.ax_marg_x, fill=False, palette=palette)
        jax.ax_marg_y = sns.histplot(y="umap2", data=grpdf, hue="grp", element="step",
                                     ax=jax.ax_marg_y, fill=False, palette=palette)
        jax.ax_joint.set(xlim=xlim, ylim=ylim, xlabel=f"UMAP 1\n{subtitle}\n{grp}", ylabel="UMAP 2")
        jax.ax_joint.yaxis.set_major_locator(ticker.MaxNLocator(4))
        jax.ax_joint.xaxis.set_major_locator(ticker.MaxNLocator(4))
        # jax.ax_joint.axhline(6.5)
        jax.ax_marg_x.legend()
        jax.ax_marg_y.legend()
        jax.ax_joint.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        tppoints = grpdf[(grpdf["fdr"] <= 0.01) & (grpdf["Type"] == "TT")]
        jax.ax_joint.scatter(tppoints["umap1"], tppoints["umap2"], s=1, c="k")
        jax.ax_marg_x.hist(tppoints["umap1"], histtype="step", color="k")
        jax.ax_marg_y.hist(tppoints["umap2"], orientation="horizontal", histtype="step", color="k")
        plt.show()
        fs.append(jax)
    return fs


if __name__ == "__main__":
    # init reducer
    # feature input file
    feats_in = r"C:\Users\hanjo\Desktop\test_rp\results_new\processed_psms.csv"
    # feature input file
    psms_in = r"C:\Users\hanjo\Desktop\test_rp\results_new\error_features_interactions.csv"
    fasta_file = r"F:\Sven\CloudStorage\Dropbox\02_HPI\02_Projects\ECCP\00_ZONODO\fasta\EColi_K12_reviewed_20190828_cbnn_filter.fasta"

    feats_df = pd.read_csv(feats_in)
    psms_df = pd.read_csv(psms_in)
    psms_df = annotate_decoytype(psms_df)
    psms_df = annotate_organism(psms_df, fasta_file)

    # feats_df
    reducer = umap.UMAP()
    scaled_df = StandardScaler().fit_transform(feats_df)
    embedding = reducer.fit_transform(scaled_df)

    # make nice annotation   formatting
    em_df = pd.DataFrame(embedding)
    em_df.index = feats_df.index
    em_df.columns = ["umap1", "umap2"]
    em_df["Type"] = psms_df["DecoyType"].values
    em_df["fdrGroup"] = psms_df.loc[psms_df.index]["fdrGroup"].values
    em_df["OS"] = psms_df.loc[psms_df.index]["OS"].values
    em_df["fdr"] = psms_df.loc[psms_df.index]["fdr"].values
    em_df["grp"] = em_df["Type"] + "_" + em_df["OS"]
    em_df["pass_fdr"] = (em_df["Type"] == "TT") & em_df["fdr"] <= 0.01
    em_df = em_df[em_df["Type"] != "DD"]
    em_df = em_df[~em_df["grp"].str.contains("sapiens")]
