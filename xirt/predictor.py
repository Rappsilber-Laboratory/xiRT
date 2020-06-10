"""Module to organize predictions from CLMS data."""
import sys

import pandas as pd
import yaml
import numpy as np
from sklearn.model_selection import KFold

from xirt import processing as xp
from xirt import sequences as xs
from xirt import xirtnet


def train_xirt(csms_df, params_xirt):
    csms_df = pd.read_csv("C:\\Users\\Hanjo\\Documents\\xiRT\\tests\\fixtures\\50pCSMFDR_universal_final.csv")
    params_xirt_loc = "C:\\Users\\Hanjo\\Documents\\xiRT\\tests\\fixtures\\xirt_params.yaml"
    params_xirt = yaml.load(open(params_xirt_loc), Loader=yaml.FullLoader)

    training_data = preprocess(csms_df, sequence_type="crosslink", max_length=-1,
                               cl_residue=True, fraction_cols=["xirt_SCX", "xiRT_hSAX"])

class model_data:
    """Class to keep data together."""
    def __init__(self, psms_df, features1, features2=pd.DataFrame(), le=None):
        """
        Class to keep the model data at a single place.

        Args:
            psms_df: df, dataframe with identifications
            features1: df, features for the second peptide
            features2: df, features for the second peptide
            le:
        """
        # init by setting training column to false for all ids
        psms_df["xirt_training"] = False

        self.psms = psms_df
        self.features1 = features1
        self.features2 = features2
        self.le = le


def generate_model(param, input_dim, sequence_type):
    """
    Function to build the network archtiecture.

    If the input is linear or pseudolinear the resulting networks dont feature the Siamese
    architecutre. Only for the crosslinks the Siamese architecture is used.

    Args:
        param: dict, parameters to init the architecture
        input_dim: int, number of columns that are used as input for the network (rnn_features)
        sequence_type: str, one of linear, pseudolinear, crosslink

    Returns:

    """
    if sequence_type in ("linear", "pseudolinear"):
        xiRTNET = xirtnet.build_xirtnet(config=param, input_dim=input_dim,
                                        input_meta = None, single_task = "None")
    else:
        xiRTNET = xirtnet.build_siamese(config=param, input_dim=input_dim,
                                        input_meta=None, single_task="None")
    return xiRTNET


def preprocess(matches_df, sequence_type="crosslink", max_length=-1, cl_residue=True,
               fraction_cols=["xirt_SCX", "xirt_hSAX"]):
    """Prepare peptide identifications to be used with xiRT.

    High-level wrapper performing multiple steps. In processing order this function:
    sets and index, sorts by score, reorders crosslinked peptides for length (alpha longer),
    rewrites sequences to modX, marks duplicates, modifies cl residues, computes RNN features
    and stores everything in the data model.

    Args:
        matches_df: df, dataframe with peptide identifications
        sequence_type: str, either linear or cross-linked indicating the input molecule type.
        max_length: int, maximal length of peptide sequences to consider. Longer sequences will
        be removed
        cl_residue: bool, if true handles cross-link sites as additional modifications

    Returns:
        model_data, processed feature dataframes and label encoder
    """
    # set index
    matches_df.set_index("PSMID", drop=False, inplace=True)

    # sort to keep only highest scoring peptide from duplicated entries
    matches_df = matches_df.sort_values(by="score", ascending=False)

    # generate columns to handle based on input data type
    if sequence_type == "crosslink":
        # change peptide order
        matches_df["Peptide1"], matches_df["Peptide2"], matches_df["PeptidesSwapped"] = \
            xs.reorder_sequences(matches_df)
        seq_in = ["Peptide1", "Peptide2"]

    elif sequence_type == "pseudolinear":
        # TODO
        sys.exit("Sorry, not yet implemented ):. (crosslink, pseudolinears, linear)")

    elif sequence_type == "linear":
        seq_in = ["Peptide1"]
    else:
        sys.exit("sequence type not supported. Must be one of (crosslink, pseudolinears, linear)")

    # dynamically get if linear or crosslinked
    seq_proc = ["Seqar_" + i for i in seq_in]

    # perform the sequence based processing
    matches_df = xp.prepare_seqs(matches_df, seq_cols=seq_in)

    # concat peptide sequences
    matches_df["PepSeq1PepSeq2_str"] = matches_df["Peptide1"] + matches_df["Peptide2"]

    # mark all duplicates
    matches_df["Duplicate"] = matches_df.duplicated(["PepSeq1PepSeq2_str"], keep="first")

    if cl_residue:
        xs.modify_cl_residues(matches_df, seq_in=seq_in)

    features_rnn_seq1, features_rnn_seq2, le = \
        xp.featurize_sequences(matches_df, seq_cols=seq_proc, max_length=max_length)

    if len(fraction_cols) > 0:
        xp.fraction_encoding(matches_df, rt_methods=fraction_cols)

    # keep all data together in a data class
    training_data = model_data(matches_df, features_rnn_seq1, features_rnn_seq2, le=le)
    return training_data

# %%
def develop():
    in_loc = r"C:\\Users\\Hanjo\\Documents\\xiRT\\tests\\fixtures\\50pCSMFDR_universal_final.csv"
    xiRT_loc = r"C:\\Users\\Hanjo\\Documents\\xiRT\\tests\\fixtures\\xirt_params.yaml"
    xiRT_params = yaml.load(open(xiRT_loc), Loader=yaml.FullLoader)
    sequence_type = "crosslink"
    fdr_cutoff = 0.05
    sample_frac = 1
    sample_state = 42
    psms_df = pd.read_csv(in_loc)
    data = preprocess(psms_df, sequence_type=sequence_type, max_length=-1, cl_residue=False,
                      fraction_cols=["xirt_SCX", "xirt_hSAX"])


def train(data, fdr_cutoff=0.05, sample_frac=1, sample_state=42):
    """asdasdasdasdasdasd.

    Args:
        data:
        fdr_cutoff:
        sample_frac:
        sample_state:

    Returns:
    """
    data.psms["xirt_RP"] = data.psms["xirt_RP"] / 60.

    # extract the training data
    psms_train = data.psms[(data.psms["FDR"] <= fdr_cutoff)
                           & (data.psms["isTT"])
                           & (data.psms["Duplicate"] == False)]
    psms_train = psms_train.sample(frac=sample_frac, random_state=sample_state)
    psms_train["depart_training"] = True

    # psms not involved in training
    psms_nontrain = data.psms.loc[data.psms.index.difference(psms_train.index)]

    loss_weights, metrics, multi_task_loss = xirtnet.create_params(xiRT_params)
    xiRTNET = generate_model(xiRT_params, data.features1, sequence_type)
    xirtnet.compile_model(xiRTNET, xiRT_params, loss=multi_task_loss,
                          metric=metrics, loss_weights=loss_weights)
    outname = r"C:\\Users\\Hanjo\\Documents\\xiRT\\tests\\fixtures\\temp\\"
    train_data = [data.features1.loc[psms_train.index],
                  data.features1.loc[psms_train.index]]

    temp_config = xiRT_params["output"]
    temp_config["hsax-column"] = "xirt_hSAX_ordinal"
    temp_config["scx-column"] = "xirt_SCX_ordinal"
    temp_config["rp-column"] = "xirt_RP"

    callbacks = xirtnet.get_callbacks("dummy", check_point=True, log_csv=True, early_stopping=True,
                                      patience=5, prefix_path=outname)

    y_train = xirtnet.prepare_multitask_y(psms_train.loc[psms_train.index], xiRT_params["output"])
    history = xiRTNET.fit(train_data, y_train, batch_size=512, epochs=50, callbacks=callbacks)
    predictions = xiRTNET.predict(train_data)

    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    sns.jointplot(y_train[2], np.ravel(predictions[2]))
    plt.show()


def generate_cv_idx(cv_seed, cv_type, ncv, psms_df, rnn_features):
    # if cv_type == "simple":
    idxs, single_indices, _ = get_folds(rnn_features, ncv, cv_seed)
    single_indices = np.array([psms_df.iloc[i].index.values for i in single_indices])
    # else:
    #     idxs, single_indices, qc = get_folds_minimize_overlap(psms_df, ncv, cv_seed)
    #     single_indices = np.array([np.array(i) for i in single_indices])
    return idxs, single_indices


def get_folds(rnn_features, ncv=5, cv_seed=2020):
    """Return train and test indices for a dataframe.

    :param cv_seed: int, seed for random number generation
    :param ncv: int, number of folds
    :param rnn_features: df, dataframe with features
    :return: (idx, single_indices), The idx array is a simple np array from 0 to ncv-1. The single_
    indices are the 0-based row numbers for the individual folds.
    """
    kf = KFold(n_splits=ncv, random_state=cv_seed)
    kf.get_n_splits(rnn_features)
    single_indices = []
    # get the all indices in a list to make it easier to iterate over them
    # important: these gives not the pandas index but the row numbers
    for train_index, test_index in kf.split(rnn_features):
        single_indices.append(test_index)
    single_indices = np.array(single_indices)
    idxs = np.arange(ncv)
    return idxs, single_indices, None
