"""Module with higher level processing functions.

Combines several sequence processing steps for convenience and sequence-
unrelated processing functions.
"""

import numpy as np
import pandas as pd
from pyteomics import parser
from tensorflow.keras import utils

from xirt import sequences as xs


def prepare_seqs(psms_df, seq_cols=["Peptide1", "Peptide2"]):
    """Convert Peptide sequences to unified format.

    This conversion simplifies the alphabet of the amino acids, removes special characters,
    replaces digits to written numbers, removes n-terminal modifications, rewrites Xmod
    modifications to modX format and splits the amino acid string into a list.

    Args:
        psms_df:  df, psm dataframe
        seq_cols: list, list of columns names that contain the peptide sequences with
        modifications

    Returns:
        df, sequence arrays

    Information:
        Performs the following steps:
        - simplifie amino acid alphabet by replacing non-sandard aa
        - remove special characters (brackets, _)
        - replace numbers by written numbers (3 -> three)
        - remove n-terminal mods since they are currently not supported
        - rewrite peptide sequence to modX format (e.g. Mox -> oxM)
        - parse a string sequence into a list of amino acids, adding termini
    """
    # for all sequence columns in the dataframe perform the processing
    # if linear only a single column, if cross-linked two columns are processed
    for seq_col in seq_cols:
        # code if sequences are represented with N.SEQUENCE.C
        if "." in psms_df.iloc[0][seq_col]:
            psms_df["Seq_" + seq_col] = psms_df[seq_col].str.split(".").str[1]
        else:
            psms_df["Seq_" + seq_col] = psms_df[seq_col]

        sequences = psms_df["Seq_" + seq_col]
        sequences = sequences.apply(xs.simplify_alphabet)
        sequences = sequences.apply(xs.remove_brackets_underscores)
        sequences = sequences.apply(xs.replace_numbers)
        sequences = sequences.apply(xs.remove_nterm_mod)
        sequences = sequences.apply(xs.rewrite_modsequences)
        sequences = sequences.apply(parser.parse, show_unmodified_termini=True)
        psms_df["Seqar_" + seq_col] = sequences
    return(psms_df)


def featurize_sequences(psms_df, seq_cols=["Seqar_Peptide1", "Seqar_Peptide2"], max_length=-1):
    """Generate a featureized version of sequences from a data frame.

    The featureization is done via by retrieving all modifications, the relevant alphabet of amino
    acids and then applying label encoding do the amino acid sequences.

    :param psms_df: df, dataframe with identifications
    :param seq_cols: list, list with column names
    :param max_legnth: int, maximal length for peptides to be included as feature
    :return:
    """
    # transform a list of amino acids to a feature dataframe
    # if two sequence columns (crosslinks)

    # define lambda function to get the alphabet in both cases
    f = lambda x: xs.get_alphabet(x)

    # get padding length
    if max_length == -1:
        max_length = psms_df[seq_cols].applymap(len).max().max()

    # get amino acid alphabet
    if len(seq_cols) > 1:
        alphabet = np.union1d(f(psms_df[seq_cols[0]].str.join(sep="").drop_duplicates()),
                              f(psms_df[seq_cols[1]].str.join(sep="").drop_duplicates()))
    else:
        alphabet = f(psms_df[seq_cols[0]].str.join(sep="").drop_duplicates())

    # perform the label encoding + padding
    encoded_s1, le = xs.label_encoding(psms_df[seq_cols[0]], max_length, alphabet=alphabet)
    seq1_padded = generate_padded_df(encoded_s1, psms_df.index)

    if len(seq_cols) > 1:
        encoded_s2, _ = xs.label_encoding(psms_df[seq_cols[1]], max_length, alphabet=alphabet,
                                          le=le)
        seq2_padded = generate_padded_df(encoded_s2, psms_df.index)
    else:
        seq2_padded = pd.DataFrame()

    return(seq1_padded, seq2_padded, le)


def generate_padded_df(encoded_ar, index):
    """Generate indexed dataframe from already label-encoded sequences.

    Args:
        encoded_ar:
        index:

    Returns:
        dataframe, label encoded peptides as rows
    """
    seqs_padded = pd.DataFrame(encoded_ar, index=index)
    seqs_padded.columns = ["rnn_{}".format(str(i).zfill(2)) for i in np.arange(len(encoded_ar[0]))]
    return seqs_padded


def fraction_encoding(psms_df, rt_methods):
    """
    Add the necessary fraction columns.

    Following columns are added: prefix_0based, Actual_prefix and prefix_1hot, where
    prefix is taken from the rt_methods argument.

    Parameters:
        psms_df: df, dataframe with peptide identifications and fraction columns
        rt_methods: ar-like, list of columns that have the fraction encoding.

    Return:
        None
    """
    for col in rt_methods:
        # encode the classes for the fractions 0-based
        frac_unique = psms_df[col].drop_duplicates().sort_values()
        # get 0-based numbers
        frac_unique = pd.Series(np.arange(frac_unique.size), index=frac_unique)
        # add new column to dataframe
        psms_df[col + "_0based"] = frac_unique.loc[psms_df[col]].values

        # special encoding
        nclasses = len(frac_unique)
        classes = {}
        for class_i in np.arange(0, nclasses):
            init_class_encoding = np.zeros(nclasses)
            init_class_encoding[:class_i+1] = 1
            classes[class_i] = init_class_encoding

        # make categorical
        frac_categorical = utils.to_categorical(psms_df[col + "_0based"])

        psms_df[col + "_1hot"] = list(frac_categorical)
        psms_df[col + "_ordinal"] = psms_df[col + "_0based"].map(classes)


def transform_RT(retention_times):
    """
    Return RT in minutes.

    Args:
        retention_times: ar-like, observed RT data

    Returns:
        ar-like, transformed retention times in minutes
    """
    if (retention_times > 720).sum() > 0:
        retention_times = retention_times / 60.
    return retention_times
