import numpy as np
import pandas as pd
import pytest
from pyteomics.parser import std_amino_acids

from xirt import sequences as xs


def test_simplify_alphabet():
    peptide = "ELVISUK"
    expected = "ELVISCK"
    result = xs.simplify_alphabet(peptide)
    assert result == expected


def test_remove_brackets_underscores():
    peptide = "ELVISM_ox_MM(ox)"
    expected = "ELVISMoxMMox"
    result = xs.remove_brackets_underscores(peptide)
    assert result == expected


def test_replace_numbers():
    peptide = "ELVISKbs3KING"
    expected = "ELVISKbsthreeKING"
    result = xs.replace_numbers(peptide)
    assert result == expected


def test_remove_nterm_mod():
    peptide = "acALVISMoxLIVES"
    expected = "ALVISMoxLIVES"
    result = xs.remove_nterm_mod(peptide)
    assert result == expected


def test_rewrite_modsequences():
    peptide = "ELVISMoxLIVES"
    expected = "ELVISoxMLIVES"
    result = xs.rewrite_modsequences(peptide)
    assert expected == result


def test_remove_lower_letters():
    peptide = "ELVISKbsthreeKING"
    expected = "ELVISKKING"
    result = xs.remove_lower_letters(peptide)
    assert result == expected


def test_to_unmodified_sequence():
    peptide = "n.acELVISKbsthreeKIN(ox)G"
    expected = "ELVISKKING"
    result = xs.to_unmodified_sequence(peptide)
    assert result == expected


def test_reorder_Sequences():
    # test reorder sequence based on length
    # A- B -> B-A
    # A - BA -> BA - A
    # AA  AC -> AC - AA
    matches_df = pd.DataFrame()
    matches_df["Peptide1"] = ["A", "B", "AB", "A", "AB", "AC", "AA"]
    matches_df["Peptide2"] = ["B", "A", "A", "AB", "AB", "AA", "AC"]
    matches_df["PeptidePos1"] = [1, 2, 3, 4, 5, 6, 7]
    matches_df["PeptidePos2"] = [11, 12, 13, 14, 15, 16, 17]
    matches_df.index = np.arange(100, 107, 1)

    seq1_exp = np.array(["B", "B", "AB", "AB", "AB", "AC", "AC"])
    seq2_exp = np.array(["A", "A", "A", "A", "AB", "AA", "AA"])

    pos1_exp = np.array([11, 2, 3, 14, 15, 6, 17])
    pos2_exp = np.array([1, 12, 13, 4, 5, 16, 7])

    swapped_exp = np.array([True, False, False, True, True, False, True])

    swapped_df = xs.reorder_sequences(matches_df)
    assert np.all(seq1_exp == swapped_df.Peptide1)
    assert np.all(seq2_exp == swapped_df.Peptide2)
    assert np.all(pos1_exp == swapped_df.PeptidePos1)
    assert np.all(pos2_exp == swapped_df.PeptidePos2)
    assert np.all(swapped_exp == swapped_df.swapped)


def test_reorder_Sequences_test_raise():
    # test reorder sequence based on length
    # A- B -> B-A
    # A - BA -> BA - A
    # AA  AC -> AC - AA
    matches_df = pd.DataFrame()
    matches_df["Peptide1"] = ["A", "B", "AB", "A", "AB", "AC", "AA"]
    matches_df["Peptide2"] = ["B", "A", "A", "AB", "AB", "AA", "AC"]
    matches_df["PeptidePos1"] = [1, 2, 3, 4, 5, 6, 7]
    matches_df.index = np.arange(100, 107, 1)
    # this should raise an error since PeptidePos2 is missing
    with pytest.raises(ValueError):
        xs.reorder_sequences(matches_df)


def test_modify_cl_residues():
    # modify cl residues by cl prefix, test by sequence and cross-link site information
    psms_df = pd.DataFrame()
    psms_df["Seqar_Peptide1"] = [list(i) for i in ["ELVISKLIVES", "KINGR", "PEPKR"]]
    psms_df["Seqar_Peptide2"] = [list(i) for i in ["AAKAAA", "KCCCC", "DDRK"]]
    psms_df["LinkPos1"] = [5, 0, 3]
    psms_df["LinkPos2"] = [2, 0, 2]

    seq1_exp = [['E', 'L', 'V', 'I', 'S', 'clK', 'L', 'I', 'V', 'E', 'S'],
                ['clK', 'I', 'N', 'G', 'R'],
                ['P', 'E', 'P', 'clK', 'R']]
    seq2_exp = [['A', 'A', 'clK', 'A', 'A', 'A'],
                ['clK', 'C', 'C', 'C', 'C'],
                ['D', 'D', 'clR', 'K']]

    xs.modify_cl_residues(psms_df, seq_in=["Peptide1", "Peptide2"])

    assert np.all(psms_df["Seqar_Peptide1"] == pd.Series(seq1_exp))
    assert np.all(psms_df["Seqar_Peptide2"] == pd.Series(seq2_exp))


def test_modify_cl_residues_reduce():
    # modify cl residues by cl prefix, test by sequence and cross-link site information
    psms_df = pd.DataFrame()
    psms_df["Seqar_Peptide1"] = [list(i) for i in ["ELVISKLIVES", "KINGR", "PEPKR"]]
    psms_df["Seqar_Peptide2"] = [list(i) for i in ["AAKAAA", "KCCCC", "DDRK"]]
    psms_df["LinkPos1"] = [5, 0, 3]
    psms_df["LinkPos2"] = [2, 0, 2]

    seq1_exp = [['E', 'L', 'V', 'I', 'S', 'clX', 'L', 'I', 'V', 'E', 'S'],
                ['clX', 'I', 'N', 'G', 'R'],
                ['P', 'E', 'P', 'clX', 'R']]
    seq2_exp = [['A', 'A', 'clX', 'A', 'A', 'A'],
                ['clX', 'C', 'C', 'C', 'C'],
                ['D', 'D', 'clX', 'K']]

    xs.modify_cl_residues(psms_df, seq_in=["Peptide1", "Peptide2"], reduce_cl=True)

    assert np.all(psms_df["Seqar_Peptide1"] == pd.Series(seq1_exp))
    assert np.all(psms_df["Seqar_Peptide2"] == pd.Series(seq2_exp))


def test_get_mods():
    # test modx extraction function
    sequence_ar = ["ELVISoxM", "KINGclRLINKED", "PEPTIDEcmCLIVEK"]
    f = lambda x: xs.get_mods(x)
    mods = sorted(f(sequence_ar))
    exp_mods = sorted(["oxM", "clR", "cmC"])

    assert np.all(mods == exp_mods)


def test_get_alphabet():
    sequence_ar = ["KKKoxMCCC", "KRKRcmC", "PEPTIDE"]
    f = lambda x: xs.get_alphabet(x)
    mods = sorted(f(sequence_ar))
    exp_mods = sorted(["K", "oxM", "C", "R", "cmC", "P", "E", "T", "D", "I"])
    assert np.all(mods == exp_mods)


def test_label_encoding():
    psms_df = pd.DataFrame()
    psms_df["Seqar_Peptide1"] = [["H-", "P", "E", "-OH"],
                                 ["H-", "oxM", "P", "E", "-OH"]]
    psms_df["Seqar_Peptide2"] = [["H-", "P", "E", "P", "-OH"],
                                 ["H-", "oxM", "E", "E", "E", "-OH"]]

    all_aminoacids = np.array(['-OH', 'E', 'H-', 'P', 'oxM'])
    exp_rnn_seq1 = [[0, 0, 3, 4, 2, 1], [0, 3, 5, 4, 2, 1]]
    exp_rnn_seq2 = [[0, 3, 4, 2, 4, 1], [3, 5, 2, 2, 2, 1]]

    encoded_s1, le1 = xs.label_encoding(psms_df["Seqar_Peptide1"], 6, alphabet=all_aminoacids)
    encoded_s2, le2 = xs.label_encoding(psms_df["Seqar_Peptide2"], 6, alphabet=all_aminoacids,
                                        le=le1)

    assert np.all(encoded_s1 == exp_rnn_seq1)
    assert np.all(encoded_s2 == exp_rnn_seq2)
    assert le1 == le2


def test_label_encoding_alphabet():
    psms_df = pd.DataFrame()
    psms_df["Seqar_Peptide1"] = [sorted(std_amino_acids)]
    psms_df["Seqar_Peptide2"] = [sorted(std_amino_acids)]

    encoded_s1, le1 = xs.label_encoding(psms_df["Seqar_Peptide1"], 20)
    encoded_s2, le2 = xs.label_encoding(psms_df["Seqar_Peptide2"], 20, le=le1)

    assert np.all(encoded_s1 == np.arange(1, 21))
    assert np.all(encoded_s1 == np.arange(1, 21))
    assert le1 == le2
