"""Module to process peptide sequences."""
import re
from collections import Counter
import numpy as np
from pyteomics import parser
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import sequence as ts
import cython
import logging
from xiutilities import pandas_utils

from xirt import const

logger = logging.getLogger('xirt').getChild(__name__)

def simplify_alphabet(sequence):
    """Replace ambiguous amino acids.

    Some sequences are encoded with 'U', arbitrarily choose C as residue to
    replace any U (Selenocystein).

    Parameters:
    ----------
    sequence: string,
              peptide sequences
    """
    return sequence.replace("U", "C")


def remove_brackets_underscores(sequence):
    """Remove all brackets (and underscores...) from protein sequences.

    Needed for MaxQuant processing.

    Parameters:
    ----------
    sequence: string,
              peptide sequences
    """
    return re.sub(r"[\(\)\[\]_\-]", "", sequence)


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
    pattern = re.compile(r"|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], sequence)


def remove_nterm_mod(sequence):
    """Remove the nterminal modification.

    Meant to be used for "ac" modifications in front of the sequence.
    They are not currently supported and need to be removed.

    :param sequence: str, peptide sequence
    :return:
    """
    return re.sub(r'^([a-z]+)([A-Z])', r'\2', sequence, flags=re.MULTILINE)


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
    return re.sub(r"([A-Z])([^A-Z]+)", r'\2\1', sequence)


def remove_lower_letters(sequence):
    """Remove lower capital letters from the sequence.

    :param sequence: str, peptide sequence
    :return:
    """
    return re.sub(r"[a-z]", "", sequence)


def to_unmodified_sequence(sequence):
    """Remove lower capital letters, brackets from the sequence.

    :param sequence: str, peptide sequence
    :return:
    """
    return (re.sub(r"[^[A-Z]", "", sequence))


def reorder_sequences(matches_df, column_names=const.default_column_names):
    """Reorder peptide sequences by length.

    Defining the longer peptide as alpha peptide and the shorter petpide as
    beta peptide. Ties are resolved lexicographically.

    Args:
        matches_df: df, dataframe with peptide identifications.
        column_names: Column names of input file.

    Returns:
        df, dataframe with the swapped cells and additional indicator column ('swapped')
    """
    # compile regex to match columns with 1/2 in the end of the string
    # check if all pairwise columns are there
    r = re.compile(r"\w+(?:1$|2$)")
    # get matching columns
    match_columns = list(filter(r.match, matches_df.columns))
    # remove number
    pairs_noidx = [i[:-1] for i in match_columns]
    # count occurrences and check if pairs are there
    counts = [(i, j) for i, j in Counter(pairs_noidx).items() if j == 2]
    if len(counts) * 2 != len(pairs_noidx):
        print(match_columns)
        raise ValueError("Error! Automatic column matching could not find pairwise crosslink "
                         "columns. Please make sure that you have a peptide1, peptide2 "
                         "column name pattern for crosslinks. Columns must appear in pairs if there"
                         " is a number in the end of the name.")

    # order logic, last comparison checks lexigographically
    is_longer = (matches_df[column_names['peptide1_sequence']].apply(len) > matches_df[column_names['peptide2_sequence']].apply(len)).values
    is_shorter = (matches_df[column_names['peptide1_sequence']].apply(len) < matches_df[column_names['peptide2_sequence']].apply(len)).values
    is_greater = (matches_df[column_names['peptide1_sequence']] > matches_df[column_names['peptide2_sequence']]).values

    # create a copy of the dataframe
    swapping_df = matches_df.copy()
    swapped = np.ones(len(matches_df), dtype=bool)

    # z_idx for 0-based index
    # df_idx for pandas
    for z_idx, df_idx in enumerate(matches_df.index):
        if not is_shorter[z_idx] and is_greater[z_idx] or is_longer[z_idx]:
            # for example: AC - AA, higher first, no swapping required
            swapped[z_idx] = False
        else:
            # for example: AA > AC, other case, swap
            swapped[z_idx] = True

        if swapped[z_idx]:
            for col in pairs_noidx:
                swapping_df.at[df_idx, col + str(2)] = matches_df.iloc[z_idx][col + str(1)]
                swapping_df.at[df_idx, col + str(1)] = matches_df.iloc[z_idx][col + str(2)]
    swapping_df["swapped"] = swapped
    return swapping_df


def modify_cl_residues(matches_df, seq_in, column_names=const.default_column_names, reduce_cl=False):
    """
    Change the cross-linked residues to modified residues.

    Function uses the Seqar_*suf columns to compute the new peptides.

    Args:
        matches_df: df, dataframe with peptide identifications. Required columns
        seq_in: ar-like, list of columns with peptide entries
        reduce_cl: bool, if true crosslinked residues are reduced to X, else the linked residue
        is kept (default: False). This is useful for transfer learning or promiscuous crosslinker
        such as SDA.
    Returns:
        psms_df: df, dataframe with adapted sequences in-place
    """
    # increase the alphabet by distinguishing between crosslinked K and non-crosslinked K
    # introduce a new prefix cl for each crosslinked residue
    for seq_id, seq_i in enumerate(seq_in):
        #error_df = matches_df[
        #    matches_df["Seqar_" + seq_i].str.len() <= matches_df[column_names['link_pos_basename'] + str(seq_id + 1)]
        #][["Seqar_" + seq_i, column_names['link_pos_basename'] + str(seq_id + 1)]]

        #if len(error_df) > 0:
        #    print(f"List index out of range for {seq_id}. Check peptide sequence for unwanted characters")
        #    print(error_df)

        matches_df["Seqar_" + seq_i] = matches_df.apply(
            lambda r: convert_seqar(
                r["Seqar_" + seq_i],
                r[column_names['link_pos_basename'] + str(seq_id + 1)],
                reduce_cl
            ),
            axis=1
        )
        #matches_df["Seqar_" + seq_i] = pandas_utils.async_apply(
        #    matches_df,
        #    lambda r: convert_seqar(
        #        r["Seqar_" + seq_i],
        #        r["link_pos_p" + str(seq_id + 1)],
        #        reduce_cl
        #    ),
        #    axis=1
        #)


@cython.ccall
def convert_seqar(seqar: list, linkpos: int, reduce_cl=False):
    if not isinstance(linkpos, int):
        return seqar
    if linkpos < 0 or linkpos >= len(seqar):
        return seqar
    residue = seqar[linkpos]
    if reduce_cl:
        seqar[linkpos] = "X"
    else:
        seqar[linkpos] = "cl" + residue
    return seqar


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
    return np.unique(re.findall("-OH|H-|[a-z0-9]+[A-Z]", " ".join(sequences)))


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
    return np.unique(
        re.findall("-OH|H-|[a-z0-9]+[A-Z]|[A-Z]", " ".join(sequences))
    )


def label_encoding(sequences, min_sequence_length, max_sequence_length, alphabet=[], le=None):
    """Label encode a list of peptide sequences.

    Parameters
    ----------
    sequences : ar-like
        list of amino acid characters (n/c-term/modifications.
    max_sequence_length : int
        maximal sequence length (for padding).
    min_sequence_lenth: int
        minimal sequence length (for padding).
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

    logger.debug("Padding sequences")

    max_arr_len = min([
        sequences.str.len().max(),
        max_sequence_length
    ])

    max_arr_len = max([
        max_arr_len,
        min_sequence_length
    ])

    # use an offset of +1 since shorter sequences will be padded with zeros
    # to achieve equal sequence lengths
    X_encoded = pandas_utils.async_apply(
            sequences,
            lambda x: np.concatenate([
                np.zeros(max_arr_len),  # Pad zeros
                le.transform(x) + 1  # Add one to avoid zero padding conflict
            ])[-max_arr_len:]  # Cut to length
    )
    return X_encoded, le
