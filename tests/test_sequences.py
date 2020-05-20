import numpy as np
import pandas as pd

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
    # test redorder sequence based on length
    xidf = pd.DataFrame()
    xidf["Peptide1"] = ["A", "B", "AB", "A", "AB", "AC", "AA"]
    xidf["Peptide2"] = ["B", "A", "A", "AB", "AB", "AA", "AC"]

    seq1_exp = np.array(["B", "B", "AB", "AB", "AB", "AC", "AC"])
    seq2_exp = np.array(["A", "A", "A", "A", "AB", "AA", "AA"])
    swapped_exp = np.array([True, False, False, True, True, False, True])

    seq1, seq2, swapped = xs.reorder_sequences(xidf)
    assert np.all(seq1_exp == seq1)
    assert np.all(seq2_exp == seq2)
    assert np.all(swapped == swapped_exp)


def test_modify_cl_residues():
    # modify cl residues by cl prefix, test by sequence and cross-link site information
    psms_df = pd.DataFrame()
    psms_df["Seqar_Peptide1"] = [list(i) for i in ["ELVISKLIVES", "KINGR", "PEPKR"]]
    psms_df["Seqar_Peptide2"] = [list(i) for i in ["AAKAAA", "KCCCC", "DDRK"]]
    psms_df["LinkPos1"] = [5, 0, 3]
    psms_df["LinkPos2"] = [2, 0, 2]

    seq1_exp = np.array([
        ['E', 'L', 'V', 'I', 'S', 'clK', 'L', 'I', 'V', 'E', 'S'],
        ['clK', 'I', 'N', 'G', 'R'],
        ['P', 'E', 'P', 'clK', 'R']])
    seq2_exp = np.array([
        ['A', 'A', 'clK', 'A', 'A', 'A'],
        ['clK', 'C', 'C', 'C', 'C'],
        ['D', 'D', 'clR', 'K']])

    xs.modify_cl_residues(psms_df, seq_in=["Peptide1", "Peptide2"])

    assert np.all(psms_df["Seqar_Peptide1"].values == seq1_exp)
    assert np.all(psms_df["Seqar_Peptide2"].values == seq2_exp)


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
