import os

import numpy as np
import pandas as pd
import pytest
import yaml
from pyteomics.parser import std_amino_acids

from xirt import sequences as xs


fixtures_loc = os.path.join(os.path.dirname(__file__), 'fixtures')

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
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params_3RT.yaml")),
                           Loader=yaml.FullLoader)
    # modify cl residues by cl prefix, test by sequence and cross-link site information
    psms_df = pd.DataFrame()
    psms_df["Seqar_Peptide1"] = [list(i) for i in ["ELVISKLIVES", "KINGR", "PEPKR"]]
    psms_df["Seqar_Peptide2"] = [list(i) for i in ["AAKAAA", "KCCCC", "DDRK"]]
    psms_df["LinkPos1"] = [5, 0, 3]
    psms_df["LinkPos2"] = [2, 0, 2]

    seq1_exp = np.array(
        [
            ['E', 'L', 'V', 'I', 'S', 'clK', 'L', 'I', 'V', 'E', 'S'],
            ['clK', 'I', 'N', 'G', 'R'],
            ['P', 'E', 'P', 'clK', 'R']
        ], dtype=object
    )
    seq2_exp = np.array(
        [
            ['A', 'A', 'clK', 'A', 'A', 'A'],
            ['clK', 'C', 'C', 'C', 'C'],
            ['D', 'D', 'clR', 'K']
        ],
        dtype=object
    )

    xs.modify_cl_residues(
        psms_df,
        column_names=xiRTconfig['column_names'],
        seq_in=["Peptide1", "Peptide2"]
    )

    assert np.all(psms_df["Seqar_Peptide1"].values == seq1_exp)
    assert np.all(psms_df["Seqar_Peptide2"].values == seq2_exp)


def test_modify_cl_residues_reduce():
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params_3RT.yaml")),
                           Loader=yaml.FullLoader)
    # modify cl residues by cl prefix, test by sequence and cross-link site information
    psms_df = pd.DataFrame()
    psms_df["Seqar_Peptide1"] = [list(i) for i in ["ELVISKLIVES", "KINGR", "PEPKR"]]
    psms_df["Seqar_Peptide2"] = [list(i) for i in ["AAKAAA", "KCCCC", "DDRK"]]
    psms_df["LinkPos1"] = [5, 0, 3]
    psms_df["LinkPos2"] = [2, 0, 2]

    seq1_exp = pd.DataFrame(columns=['Seqar_Peptide1'])
    seq1_exp['Seqar_Peptide1'] = [
        ['E', 'L', 'V', 'I', 'S', 'clX', 'L', 'I', 'V', 'E', 'S'],
        ['clX', 'I', 'N', 'G', 'R'],
        ['P', 'E', 'P', 'clX', 'R']
    ]

    seq2_exp = pd.DataFrame(columns=['Seqar_Peptide2'])
    seq2_exp['Seqar_Peptide2'] = [
        ['A', 'A', 'clX', 'A', 'A', 'A'],
        ['clX', 'C', 'C', 'C', 'C'],
        ['D', 'D', 'clX', 'K']
    ]

    xs.modify_cl_residues(
        psms_df,
        column_names=xiRTconfig['column_names'],
        seq_in=["Peptide1", "Peptide2"],
        reduce_cl=True
    )

    assert np.all(psms_df["Seqar_Peptide1"] == seq1_exp['Seqar_Peptide1'].values)
    assert np.all(psms_df["Seqar_Peptide2"] == seq2_exp['Seqar_Peptide2'].values)


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
    exp_rnn_seq1 = pd.DataFrame([[0, 0, 3, 4, 2, 1], [0, 3, 5, 4, 2, 1]])
    exp_rnn_seq2 = pd.DataFrame([[0, 3, 4, 2, 4, 1], [3, 5, 2, 2, 2, 1]])

    encoded_s1, le1 = xs.label_encoding(
        psms_df["Seqar_Peptide1"],
        min_sequence_length=6,
        alphabet=all_aminoacids,
        max_sequence_length=50,
    )
    encoded_s2, le2 = xs.label_encoding(
        psms_df["Seqar_Peptide2"],
        min_sequence_length=6,
        alphabet=all_aminoacids,
        le=le1,
        max_sequence_length=50,
    )

    assert np.all(np.vstack(encoded_s1) == exp_rnn_seq1.values)
    assert np.all(np.vstack(encoded_s2) == exp_rnn_seq2.values)
    assert le1 == le2


def test_label_encoding_alphabet():
    psms_df = pd.DataFrame()
    psms_df["Seqar_Peptide1"] = [sorted(std_amino_acids)]
    psms_df["Seqar_Peptide2"] = [sorted(std_amino_acids)]

    encoded_s1, le1 = xs.label_encoding(
        psms_df["Seqar_Peptide1"],
        min_sequence_length=1,
        max_sequence_length=30,
    )
    encoded_s2, le2 = xs.label_encoding(
        psms_df["Seqar_Peptide2"],
        min_sequence_length=1,
        max_sequence_length=50,
        le=le1
    )

    assert np.all(encoded_s1.values[0] == np.arange(1, 21))
    assert np.all(encoded_s2.values[0] == np.arange(1, 21))
    assert le1 == le2
