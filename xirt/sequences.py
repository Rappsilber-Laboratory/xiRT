"""Module to process peptide sequences."""
import re

import numpy as np
from pyteomics import parser
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import sequence as ts


def simplify_alphabet(sequence):
    """Replace amibguous amino acids.

    Some sequences are encoded with 'U', arbitrarily choose C as residue to
    replace any U (Selenocystein).

    Parameters:
    ----------
    sequence: string,
              peptide sequences
    """
    return (sequence.replace("U", "C"))


def remove_brackets_underscores(sequence):
    """Remove all brackets (and underscores...) from protein sequences.

    Needed for MaxQuant processing.

    Parameters:
    ----------
    sequence: string,
              peptide sequences
    """
    return (re.sub("[\(\)\[\]_\-]", "", sequence))


def replace_numbers(sequence):
    """Replace digits to words (necessary for modX format to work.

    :param sequence:
    :return:
    """
    rep = {"1": "one",
           "2": "two",
           "3": "three",
           "4": "four",
           "5": "five",
           "6": "six",
           "7": "seven",
           "8": "eight",
           "9": "nine",
           "0": "zero"}
    pattern = re.compile("|".join(rep.keys()))
    return (pattern.sub(lambda m: rep[re.escape(m.group(0))], sequence))


def remove_nterm_mod(sequence):
    """Remove the nterminal modification.

    Meant to be used for "ac" modifications in front of the sequence.
    They are not currently supported and need to be removed.

    :param sequence: str, peptide sequence
    :return:
    """
    return (re.sub(r'^([a-z]+)([A-Z])', r'\2', sequence, flags=re.MULTILINE))


def rewrite_modsequences(sequence):
    """Rewrite modified sequences to modX format.

    Requires the input to be preprocessed such that no brackets are in the sequences.

    Meant to be used via apply.

    Example:
    -------
    sequence = "ELVIS"
    sequence = "ELVISCcmASD"

    :param sequence: str, peptide sequence
    :return:
    """
    return (re.sub("([A-Z])([^A-Z]+)", r'\2\1', sequence))


def remove_lower_letters(sequence):
    """Remove lower capital letters from the sequence.

    :param sequence: str, peptide sequence
    :return:
    """
    return (re.sub("[a-z]", "", sequence))


def to_unmodified_sequence(sequence):
    """Remove lower capital letters, brackets from the sequence.

    :param sequence: str, peptide sequence
    :return:
    """
    return (re.sub("[^[A-Z]", "", sequence))


def reorder_sequences(psms_df):
    """Reorder peptide sequences by length.

    Defining the longer peptide as alpha peptide and the shorter petpide as
    beta peptide. Ties are resolved lexicographically.

    Args:
        psms_df: df, dataframe with peptide identifications.

    Returns:
        ar, ar, ar: peptides1, peptides2, swapped
    """
    order_lex = (psms_df["Peptide1"] > psms_df["Peptide2"]).values
    order_lon = (psms_df["Peptide1"].apply(len) > psms_df["Peptide2"].apply(len)).values
    order_sma = (psms_df["Peptide1"].apply(len) < psms_df["Peptide2"].apply(len)).values
    seq1_ar = [""] * len(psms_df)
    seq2_ar = [""] * len(psms_df)
    swapped = [False] * len(psms_df)
    for idx, (seq1, seq2) in enumerate(zip(psms_df.Peptide1, psms_df.Peptide2)):
        if order_lon[idx]:
            # easy, longer peptide goes first
            seq1_ar[idx] = seq1
            seq2_ar[idx] = seq2
            swapped[idx] = False

        elif order_sma[idx]:
            # easy, shorter peptide goes last
            seq1_ar[idx] = seq2
            seq2_ar[idx] = seq1
            swapped[idx] = True
        else:
            # equal length
            if order_lex[idx]:
                # higher first
                seq1_ar[idx] = seq1
                seq2_ar[idx] = seq2
                swapped[idx] = False
            else:
                seq1_ar[idx] = seq2
                seq2_ar[idx] = seq1
                swapped[idx] = True

    return seq1_ar, seq2_ar, swapped


def modify_cl_residues(psms_df, seq_in=["Peptide1", "Peptide2"]):
    """Change the cross-linked residues to modified residues.

    Args:
        psms_df: df, dataframe with peptide identifications. Required columns
        seq_in:

    Returns:
        psms_df: df, dataframe with adapted sequences in-place
    """
    # increase the alphabet by distinguishing between crosslinked K and non-crosslinked K
    # introduce a new prefix cl for each crosslinked residue
    for seq_id, seq_i in enumerate(seq_in):
        for idx, row in psms_df.iterrows():
            residue = row["Seqar_" + seq_i][row["LinkPos" + str(seq_id + 1)]]
            psms_df.at[idx, "Seqar_" + seq_i][row["LinkPos" + str(seq_id + 1)]] = "cl" + residue


def get_mods(sequences):
    """Retrieve modifciations from dataframe in the alphabet.

    Parameters
    ----------
    sequences : ar-like
        peptide sequences.

    Returns
    -------
    List with modification strings.
    """
    # get a list of all modifications (cm, ox, bs3ohX, etc) in the data
    mods = np.unique(re.findall("-OH|H-|[a-z0-9]+[A-Z]", " ".join(sequences)))
    return (mods)


def get_alphabet(sequences):
    """Retrieve alphabet of amino acids with modifications.

    Parameters
    ----------
    sequences : ar-like
        peptide sequences.

    Returns
    -------
    List with modification strings.
    """
    alphabet = np.unique(re.findall("-OH|H-|[a-z0-9]+[A-Z]|[A-Z]", " ".join(sequences)))
    return alphabet


def label_encoding(sequences, max_sequence_length, alphabet=[], le=None):
    """Label encode a list of peptide sequences.

    Parameters
    ----------
    sequences : ar-like
        list of amino acid characters (n/c-term/modifications.
    max_sequence_length : int
        maximal sequence length (for padding).
    alphabet : list, optional
        list of the unique characters given in the sequences
    le : TYPE, optional
        label encoder instance, can be a prefitted model.

    Returns
    -------
    X_encoded : TYPE
        DESCRIPTION.
    """
    # if no alternative alphabet is given, use the defaults
    if len(alphabet) == 0:
        alphabet = parser.std_amino_acids

    if not le:
        # init encoder for the AA alphabet
        le = LabelEncoder()
        le.fit(alphabet)

    # use an offset of +1 since shorter sequences will be padded with zeros
    # to achieve equal sequence lengths
    X_encoded = sequences.apply(le.transform) + 1
    X_encoded = ts.pad_sequences(X_encoded, maxlen=max_sequence_length)
    return X_encoded, le
