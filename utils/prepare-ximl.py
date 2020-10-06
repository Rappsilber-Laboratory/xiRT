import pandas as pd
import os


path = r"C:\Users\hanjo\PycharmProjects\xiRT\results\xiRT_revision"

# read processed data
psms = pd.read_csv(os.path.join(path, "processed_psms.csv"), index_col=0)
features = pd.read_csv(os.path.join(path, "error_interactions.csv"), index_col=0)
features.columns = "feature_" + features.columns

# remove list columns
delete_columns = ["Seqar_Peptide1", "Seqar_Peptide2", "scx_1hot", "hsax_1hot", "scx_ordinal",
                  "hsax_ordinal", "Unnamed: 0"]
psms.drop(delete_columns, inplace=True, axis=1)
psms_annotated = psms.join(features)
psms_annotated["Fasta1"] = psms_annotated["Fasta1"].str.replace(",", "")
psms_annotated["Fasta2"] = psms_annotated["Fasta2"].str.replace(",", "")

psms_annotated = psms_annotated[~psms_annotated["Fasta1"].str.contains(";")]

psms_annotated.to_csv(os.path.join(path, "_ximl_input.csv"))
psms_annotated.to_excel(os.path.join(path, "_ximl_input.xlsx"))