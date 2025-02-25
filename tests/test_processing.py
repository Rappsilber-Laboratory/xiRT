import numpy as np
import pandas as pd

from xirt import processing as xp


def test_prepare_seqs_standard():
    # test high level processing function for peptide sequences, from input to features
    psms_df = pd.DataFrame()
    psms_df["Peptide1"] = ["PEPTIDE", "EMoxLUR"]
    psms_df["Peptide2"] = ["ELP(ph)R", "IKbs3_hydING"]

    proc_df = xp.prepare_seqs(psms_df, seq_cols=["Peptide1", "Peptide2"])

    exp_seqar1 = pd.Series([
        ["H-", "P", "E", "P", "T", "I", "D", "E", "-OH"],
        ["H-", "E", "oxM", "L", "C", "R", "-OH"]
    ])
    exp_seqar2 = pd.Series([
        ["H-", "E", "L", "phP", "R", "-OH"],
        ["H-", "I", "bsthreehydK", "I", "N", "G", "-OH"]
    ])
    assert np.all(proc_df["Seqar_Peptide1"] == exp_seqar1)
    assert np.all(proc_df["Seqar_Peptide2"] == exp_seqar2)


def test_prepare_seqs_specialseq():
    # test high level processing function for peptide sequences, from input to features
    psms_df = pd.DataFrame()
    psms_df["Peptide1"] = ["M.ELVR.R"]
    psms_df["Peptide2"] = ["M.ELVR.R"]
    proc_df = xp.prepare_seqs(psms_df, seq_cols=["Peptide1", "Peptide2"])
    exp_seqar1 = pd.Series([["H-", "E", "L", "V", "R", "-OH"]])
    exp_seqar2 = pd.Series([["H-", "E", "L", "V", "R", "-OH"]])
    assert np.all(proc_df["Seqar_Peptide1"].values[0] == exp_seqar1[0])
    assert np.all(proc_df["Seqar_Peptide2"].values[0] == exp_seqar2[0])


def test_generate_padded_df():
    # encoded_ar is a label encoded amino acid representation array that needs to be transformed
    # into an indexed dataframe
    encoded_ar = np.array([[0, 0, 1, 1, 5, 16],
                           [0, 0, 1, 5, 5, 16]])
    index = pd.Series(range(len(encoded_ar)))

    seq_padded_df = xp.generate_padded_df(encoded_ar, index)

    exp_index = [0, 1]
    exp_shape = (2, 6)
    exp_columns = pd.Series(["rnn_00", "rnn_01", "rnn_02", "rnn_03", "rnn_04", "rnn_05"])

    assert seq_padded_df.shape == exp_shape
    assert seq_padded_df.index.tolist() == exp_index
    assert np.all(seq_padded_df.columns == exp_columns)


def test_featurize_sequences_crosslinks():
    # test label encoding with padding and +1 for padding
    # label encoded uses sorted alphabets
    psms_df = pd.DataFrame()
    psms_df["Seqar_Peptide1"] = [["H-", "P", "E", "-OH"],
                                 ["H-", "oxM", "P", "E", "-OH"]]
    psms_df["Seqar_Peptide2"] = [["H-", "P", "E", "P", "-OH"],
                                 ["H-", "oxM", "E", "E", "E", "-OH"]]

    all_aminoacids = np.array(['-OH', 'E', 'H-', 'P', 'oxM'])
    exp_rnn_seq1 = xp.generate_padded_df(
        [[0, 0, 3, 4, 2, 1], [0, 3, 5, 4, 2, 1]],
        index=[0, 1]
    )
    exp_rnn_seq2 = xp.generate_padded_df(
        [[0, 3, 4, 2, 4, 1], [3, 5, 2, 2, 2, 1]],
        index=[0, 1]
    )

    features_rnn_seq1, features_rnn_seq2, le =\
        xp.featurize_sequences(psms_df, seq_cols=["Seqar_Peptide1", "Seqar_Peptide2"],
                               max_length=-1)

    assert np.all(features_rnn_seq1 == exp_rnn_seq1)
    assert np.all(features_rnn_seq2 == exp_rnn_seq2)
    assert np.all(le.classes_ == all_aminoacids)


def test_featurize_sequences_linear():
    # test label encoding with padding and +1 for padding
    # label encoded uses sorted alphabets
    psms_df = pd.DataFrame()
    psms_df["Seqar_Peptide1"] = [["H-", "P", "E", "-OH"],
                                 ["H-", "oxM", "P", "E", "-OH"]]

    all_aminoacids = np.array(['-OH', 'E', 'H-', 'P', 'oxM'])
    exp_rnn_seq1 = xp.generate_padded_df([[0, 3, 4, 2, 1], [3, 5, 4, 2, 1]], index=[0, 1])

    features_rnn_seq1, features_rnn_seq2, le =\
        xp.featurize_sequences(psms_df, seq_cols=["Seqar_Peptide1"], max_length=-1)

    assert np.all(features_rnn_seq1 == exp_rnn_seq1)
    assert np.all(features_rnn_seq2 == pd.DataFrame())
    assert np.all(le.classes_ == all_aminoacids)


def test_fraction_encoding():
    psms_df = pd.DataFrame()
    psms_df["xirt_hsax"] = np.arange(5, 10)
    psms_df["xirt_scx"] = np.array([1, 2, 3, 1, 2])

    exp_hsax_1hot = np.array([np.array([1, 0, 0, 0, 0]),
                              np.array([0, 1, 0, 0, 0]),
                              np.array([0, 0, 1, 0, 0]),
                              np.array([0, 0, 0, 1, 0]),
                              np.array([0, 0, 0, 0, 1])])

    # v1
    # exp_hsax_ordinal = np.array([np.array([1, 0, 0, 0, 0]),
    #                              np.array([1, 1, 0, 0, 0]),
    #                              np.array([1, 1, 1, 0, 0]),
    #                              np.array([1, 1, 1, 1, 0]),
    #                              np.array([1, 1, 1, 1, 1])])
    # v2
    exp_hsax_ordinal = np.array([np.array([0, 0, 0, 0, 0]),
                                 np.array([1, 0, 0, 0, 0]),
                                 np.array([1, 1, 0, 0, 0]),
                                 np.array([1, 1, 1, 0, 0]),
                                 np.array([1, 1, 1, 1, 0])])
    exp_hsax_0based = np.arange(0, 5)

    exp_scx_1hot = np.array([np.array([1, 0, 0]),
                             np.array([0, 1, 0]),
                             np.array([0, 0, 1]),
                             np.array([1, 0, 0]),
                             np.array([0, 1, 0])])

    # v1
    # exp_scx_ordinal = np.array([np.array([1, 0, 0]),
    #                             np.array([1, 1, 0]),
    #                             np.array([1, 1, 1]),
    #                             np.array([1, 0, 0]),
    #                             np.array([1, 1, 0])])
    # v2
    exp_scx_ordinal = np.array([np.array([0, 0, 0]),
                                np.array([1, 0, 0]),
                                np.array([1, 1, 0]),
                                np.array([0, 0, 0]),
                                np.array([1, 0, 0])])
    exp_scx_0based = [0, 1, 2, 0, 1]

    xp.fraction_encoding(psms_df, ["xirt_hsax", "xirt_scx"])

    assert np.all([a == b for a, b in zip(exp_scx_1hot, psms_df["xirt_scx_1hot"].values)])
    assert np.all([a == b for a, b in zip(exp_hsax_1hot, psms_df["xirt_hsax_1hot"].values)])

    assert np.all([a == b for a, b in zip(exp_hsax_ordinal, psms_df["xirt_hsax_ordinal"].values)])
    assert np.all([a == b for a, b in zip(exp_scx_ordinal, psms_df["xirt_scx_ordinal"].values)])

    assert np.all([a == b for a, b in zip(exp_hsax_0based, psms_df["xirt_hsax_0based"].values)])
    assert np.all([a == b for a, b in zip(exp_scx_0based, psms_df["xirt_scx_0based"].values)])


def test_transform_RT():
    expected_RTs = np.array([10000, 2000])
    assert np.all(expected_RTs / 60 == xp.transform_RT((expected_RTs)))


def test_transform_RT_no():
    expected_RTs = np.array([65, 90])
    assert np.all(expected_RTs == xp.transform_RT(expected_RTs))
