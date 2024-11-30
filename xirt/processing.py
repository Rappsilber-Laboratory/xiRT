"""Module with higher level processing functions.

Combines several sequence processing steps for convenience and sequence-
unrelated processing functions.
"""

import numpy as np
import pandas as pd
from pyteomics import parser
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils
import logging
from xirt import sequences as xs
import multiprocessing as mp
from functools import partial
from math import ceil


logger = logging.getLogger('xirt').getChild(__name__)


def prepare_seqs_mp(psms_df, seq_cols):
    n_worker = min(
        [
            ceil(len(psms_df)/10_000),
            mp.cpu_count()
        ]
    )
    slice_size = ceil(len(psms_df)/n_worker)
    slices = [
        psms_df[seq_cols].iloc[i*slice_size:(i+1)*slice_size]
        for i in range(n_worker)
    ]

    prepare_job = partial(prepare_seqs, seq_cols=seq_cols)

    with mp.Pool(n_worker) as pool:
        results = pool.map(
            prepare_job,
            slices
        )

    result = pd.concat(results)
    for c in result.columns:
        psms_df[c] = result[c]

    return psms_df


def prepare_seqs(psms_df, seq_cols):
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
    logger.info("Preparing peptide sequences for columns: {}".format(",".join(seq_cols)))

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
    return psms_df


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
    logger.info("Setting max_length to: {}".format(max_length))

    # get amino acid alphabet
    if len(seq_cols) > 1:
        alphabet = np.union1d(f(psms_df[seq_cols[0]].str.join(sep="").drop_duplicates()),
                              f(psms_df[seq_cols[1]].str.join(sep="").drop_duplicates()))
    else:
        alphabet = f(psms_df[seq_cols[0]].str.join(sep="").drop_duplicates())

    logger.info("alphabet: {}".format(alphabet))

    # perform the label encoding + padding
    logger.debug("labeling encoding")
    min_length = psms_df[seq_cols[0]].str.len().max()
    if len(seq_cols) > 1:
        min_length = max([
            min_length,
            psms_df[seq_cols[1]].str.len().max()
        ])
    le = LabelEncoder()
    le.fit(alphabet)
    encoded_s1, _ = xs.label_encoding(
        psms_df[seq_cols[0]],
        min_length,
        max_length,
        alphabet=alphabet
    )
    logger.debug("generating padded DF")
    seq1_padded = generate_padded_df(encoded_s1, psms_df.index)

    if len(seq_cols) > 1:
        logger.debug("repeat for peptide 2")
        encoded_s2, _ = xs.label_encoding(
            psms_df[seq_cols[1]],
            min_length,
            max_length,
            alphabet=alphabet,
        )
        seq2_padded = generate_padded_df(encoded_s2, psms_df.index)
    else:
        seq2_padded = pd.DataFrame()

    return seq1_padded, seq2_padded, le


def generate_padded_df(encoded_ar, index):
    """Generate indexed dataframe from already label-encoded sequences.

    Args:
        encoded_ar:
        index:

    Returns:
        dataframe, label encoded peptides as rows
    """
    seqs_padded = pd.DataFrame(
        np.array(encoded_ar.to_list()),
        index=index
    )
    seqs_padded.columns = [f"rnn_{str(i).zfill(2)}" for i in np.arange(len(encoded_ar[0]))]
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
            # version 1
            # 1, 2, 3 -> [1, 0, 0], [1, 1, 0], [1, 1, 1]
            # init_class_encoding = np.zeros(nclasses)
            # init_class_encoding[:class_i+1] = 1
            # classes[class_i] = init_class_encoding
            # version 2
            init_class_encoding = np.ones(nclasses)
            init_class_encoding[class_i:] = 0
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
