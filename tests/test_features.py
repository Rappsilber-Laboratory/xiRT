import numpy as np
import pandas as pd

from xirt import features


def test_create_simple_features():
    df = pd.DataFrame()
    df["Sequence"] = ["ELVIS", "LIVES"]

    ff_df = features.create_simple_features(df, seq_column="Sequence")
    exp_columns = sorted(
        ["turnP", "turn", "helix", "sheet", "pi", "aromaticity", "estimated_charge",
         "loglength"])

    exp_shape = (2, len(exp_columns))

    assert sorted(ff_df.columns.tolist()) == exp_columns
    assert ff_df.shape == exp_shape


def test_create_all_features():
    df = pd.DataFrame()
    df["Sequence"] = ["ELVIS", "LIVES"]

    ff_df = features.create_all_features(df)
    exp_columns = sorted(
        ["estimated_charge", "loglength", "cterm", "nterm", "netcharge", "nterm_res",
         "turnP", "turn", "helix", "sheet", "pi", "aromaticity", "hydrophobicity"])
    exp_columns.extend(["E", "L", "V", "I", "S"])
    exp_columns.sort()
    exp_shape = (2, len(exp_columns))

    assert sorted(ff_df.columns.tolist()) == exp_columns
    assert ff_df.shape == exp_shape


def test_create_all_features_mods():
    # format for mods is here Xmod
    df = pd.DataFrame()
    df["Sequence"] = ["ELVIS", "LIVES"]
    df["Modified sequence"] = ["axELVoxIS", "LIVES"]

    ff_df = features.create_all_features(df)
    exp_columns = sorted(
        ["estimated_charge", "loglength", "cterm", "nterm", "netcharge", "nterm_res",
         "turnP", "turn", "helix", "sheet", "pi", "aromaticity", "hydrophobicity"])
    exp_columns.extend(["E", "L", "V", "I", "S", "axE", "oxV", "ax"])
    exp_columns.sort()
    exp_shape = (2, len(exp_columns))

    assert sorted(ff_df.columns.tolist()) == sorted(exp_columns)
    assert ff_df.shape == exp_shape


def test_get_hydrophobicity():
    from Bio.SeqUtils.ProtParamData import kd

    peptide = "ELVIS"
    exp_hydro = np.sum([kd["E"] + kd["L"] + kd["V"] + kd["I"] + kd["S"]])
    assert exp_hydro == features.get_hydrophobicity(peptide)


def test_nterm_mod():
    peptide = "acHELVIS"
    mod = "ac"
    result = features.get_nterm_mod(peptide, mod)
    expected = 1
    assert result == expected


def test_nterm_mod_nmod():
    peptide = "HELVIS"
    assert 0 == features.get_nterm_mod(peptide, "ac")


def test_get_loglength():
    peptide = "AABBCCDDEE"
    expected = np.log(10)
    result = features.get_loglength(peptide)
    assert expected == result


def test_shortest_distance_nterm():
    peptide = "KELVIDS"
    expected = 1
    result = features.get_shortest_distance(peptide, opt="nterm")
    assert expected == result


def test_get_cterm_residue_indicator():
    peptide = "ELVISK"
    result = features.get_cterm_residue_indicator(peptide)
    assert result == 1


def test_getnmods():
    peptide = "ELVISoxMLIVEoxMoxS"
    expected = 3
    result = features.get_nmods(peptide, mod_str="ox")
    assert result == expected


def test_estimated_charge():
    peptide = "DEFWYKR"
    expected = 1. + 1. + 0.3 + 0.8 + 0.6 - 1 - 1
    result = features.get_estimated_charge(peptide)
    assert np.round(expected, 4) == np.round(result, 4)


def test_get_residue_charge():
    peptide = "DEFWYKR"
    expected = 1 + 1 - 1 - 1
    result = features.get_residue_charge(peptide)
    assert np.round(expected, 4) == np.round(result, 4)


def test_get_aa_count():
    peptide = "DDEEFFK"
    expected = 2
    result = features.get_aa_count(peptide, "D", pos=-1, direction="N")
    assert expected == result


def test_get_aa_count_positions():
    peptide = "DDEEFFK"
    expected = 0
    result = features.get_aa_count(peptide, "D", pos=5, direction="N")
    assert expected == result


def test_add_shortest_distance_cterm():
    peptide = "EPER"
    exp_distance = 2
    assert exp_distance == features.add_shortest_distance(peptide, opt="cterm")


def test_add_shortest_distance_nterm():
    peptide = "EPER"
    exp_distance = 4
    assert exp_distance == features.add_shortest_distance(peptide, opt="nterm")


def test_extract_nterm_mods():
    peptide = ["acAMELVISLIVEphSK", "ELVIS"]
    expected = ["ac"]
    result = list(np.hstack(features.extract_nterm_mods(peptide)))
    assert result == expected


def test_patches_counts():
    peptide = "PPPPPDEPPPPDEE"
    counts = 2
    assert counts == features.get_patches(peptide, aa_set1=["D", "E"], counts_only=True)


def test_patches_patches():
    peptide = "PPPPPDEPPPPDEE"
    patches = {"DD": 0, "DE": 2, "EE": 1}
    assert patches == features.get_patches(peptide, aa_set1=["D", "E"], counts_only=False)


def test_patches_patches_2():
    peptide = "PPPPPDEPPPPDEEKK"
    patches = {"EK": 1, "KE": 0}
    assert patches == features.get_patches(peptide, aa_set1=["E"], aa_set2=["K"],
                                           counts_only=False)


def test_get_sandwich():
    peptide = "AAAFXYAAA"
    expected = 1
    result = features.get_sandwich(peptide, aa="FY")
    assert expected == result


def test_get_structure_perc():
    peptide = "VIYFWL" + "NPGS" + "EMAL"
    helix = 7 / 14.
    turn = 4 / 14.
    sheet = 4 / 14.
    assert np.isclose(helix, features.get_structure_perc(peptide, "helix"), 2)
    assert np.isclose(turn, features.get_structure_perc(peptide, "turn"), 2)
    assert np.isclose(sheet, features.get_structure_perc(peptide, "sheet"), 2)


def test_simply_alphabet():
    peptide = "PPPPUPPP"
    exp_peptide = "PPPPCPPP"
    assert exp_peptide == features.simply_alphabet(peptide)


def test_turn_indicator_middle():
    sequence = "ASDPASDL"
    length = 8.
    assert 3 / length == features.get_turn_indicator(sequence)


def test_turn_indicator_start():
    sequence = "PASDASDL"
    assert 0 == features.get_turn_indicator(sequence)


def test_turn_indicator_multiple():
    # every three amino acids
    sequence = "AAPAAPAA"
    assert 3 / len(sequence) == features.get_turn_indicator(sequence)


def test_get_AA_matrix():
    sequences = ["QWERTYIPAS", 'DFGHKLCVNM']
    AA_counts = features.get_AA_matrix(sequences)

    exp_columns = sorted(['D', 'F', 'G', 'H', 'K', 'L', 'C', 'V', 'N', 'M',
                          'Q', 'W', 'E', 'R', 'T', 'Y', 'I', 'P', 'A', 'S'])

    assert sorted(AA_counts.columns) == exp_columns
    assert AA_counts.shape == (2, len(exp_columns))


def test_get_AA_matrix_custom_alphabet():
    sequences = ["DE", 'FE']
    AA_counts = features.get_AA_matrix(sequences, residues=["D", "E", "F"])
    exp_columns = sorted(['D', 'E', 'F'])
    assert sorted(AA_counts.columns) == exp_columns
    assert AA_counts.shape == (2, len(exp_columns))


def test_get_AA_matrix_custom_alphabet_lcp():
    sequences = ["DEE", 'FEE']
    AA_counts = features.get_AA_matrix(sequences, residues=["D", "E", "F"], lcp=1, correct=True)

    exp_row1 = np.array([1. + 1 * np.log(3)]) * np.array([1, 2, 0])
    exp_row2 = np.array([1. + 1 * np.log(3)]) * np.array([0, 2, 1])

    assert np.all(np.isclose(AA_counts.iloc[0].values, exp_row1, 2))
    assert np.all(np.isclose(AA_counts.iloc[1].values, exp_row2, 2))


def test_get_gravy():
    # http://www.gravy-calculator.de/index.php?page=result&by=text&from=direct
    peptide = "ELVIS"
    exp_gravy = 1.64
    return exp_gravy == features.get_gravy(peptide)


def test_get_pi():
    # http://www.gravy-calculator.de/index.php?page=result&by=text&from=direct
    peptide = "ELVIS"
    exp_pi = 4.00
    return exp_pi == np.round(features.get_pi(peptide))


def test_get_mw():
    # http://www.gravy-calculator.de/index.php?page=result&by=text&from=direct
    peptide = "ELVIS"
    exp_mw = 559.66
    return np.round(exp_mw) == np.round(features.get_weight(peptide))


def test_get_AA_matrix_custom_position():
    sequences = ["PPEPP"]
    AA_counts = features.get_AA_matrix(sequences, residues=["P", "E"], lcp=1, correct=False,
                                       pos_specific=True, ntermini=2)

    columns = ["P", "E", "NE0", "NE1", "NP0", "NP1", "CE1", "CP1"]
    # C-terminal first residue is "ignored" since usually trypsin
    exp_results = pd.DataFrame([1, 1, 0, 0, 1, 1, 0, 1]).transpose()
    exp_results.columns = columns
    assert np.all(AA_counts[columns].values == exp_results.values)


def test_get_AA_matrix_custom_position_short_pep():
    sequences = ["EE"]
    AA_counts = features.get_AA_matrix(sequences, residues=["E"], lcp=1, correct=False,
                                       pos_specific=True, ntermini=5)

    columns = ['E', 'CE1', 'CE2', 'CE3', 'CE4', 'NE0', 'NE1', 'NE2', 'NE3', 'NE4']
    # C-terminal first residue is "ignored" since usually trypsin
    exp_results = pd.DataFrame([1, 0, 0, 0, 0, 1, 0, 0, 0, 0]).transpose()
    exp_results.columns = columns
    assert np.all(AA_counts[columns].values == exp_results.values)


def test_compute_prediction_errors_single():
    tasks = ["rp"]
    obs_df = pd.DataFrame([1, 2, 3, 4, 5])
    obs_df.columns = tasks

    preds_df = pd.DataFrame(index=obs_df.index)
    preds_df["rp-prediction"] = 3

    features.compute_prediction_errors(obs_df, preds_df, tasks, frac_cols=[],
                                       single_predictions=False)

    exp_errors = obs_df["rp"] - 3
    assert np.all(preds_df["rp-error"] == exp_errors)


def test_compute_prediction_errors_multi():
    tasks = ["rp"]
    obs_df = pd.DataFrame([1, 2, 3, 4, 5])
    obs_df.columns = tasks

    preds_df = pd.DataFrame(index=obs_df.index)
    preds_df["rp-prediction"] = 3
    preds_df["rp-prediction-peptide1"] = 1
    preds_df["rp-prediction-peptide2"] = 2

    features.compute_prediction_errors(obs_df, preds_df, tasks, single_predictions=True)

    exp_errors = obs_df["rp"] - 3
    exp_errors_p1 = obs_df["rp"] - 1
    exp_errors_p2 = obs_df["rp"] - 2
    assert np.all(preds_df["rp-error"] == exp_errors)
    assert np.all(preds_df["rp-error-peptide1"] == exp_errors_p1)
    assert np.all(preds_df["rp-error-peptide2"] == exp_errors_p2)


def test_add_interactions():
    df1 = pd.DataFrame()
    df1["feature1"] = [2, 2, 2]
    df1["feature2"] = [1, 2, 3]
    df1_feat = features.add_interactions(df1, degree=2, interactions_only=True)

    exp_feats = [2.0, 4, 6.0]
    assert np.all(df1_feat["feature1 feature2"] == exp_feats)


def test_add_rt_features_rponly():
    df = pd.DataFrame()
    df["rp-error"] = [2]
    df["rp-error-peptide1"] = [-3]
    df["rp-error-peptide2"] = [3]
    df_features_rp = features.add_rt_features(df)

    assert df_features_rp.shape == (1, 13)
    assert df_features_rp["feature_rp-error"][0] == [2]
    assert df_features_rp["feature_rp-error-peptide1"][0] == [-3]
    assert df_features_rp["feature_rp-error-peptide2"][0] == [3]

    assert df_features_rp["feature_initial_prod"][0] == [np.log2(2.1*3.1*3.1)]
    assert df_features_rp["feature_initial_min"][0] == [2]
    assert df_features_rp["feature_initial_max"][0] == [3]

    assert df_features_rp["feature_rp-error_square"][0] == [4]
    assert df_features_rp["feature_rp-error_abs"][0] == [2]
    assert df_features_rp["feature_rp-error-peptide1_square"][0] == [9]
    assert df_features_rp["feature_rp-error-peptide1_abs"][0] == [3]
    assert df_features_rp["feature_rp-error-peptide2_square"][0] == [9]
    assert df_features_rp["feature_rp-error-peptide2_abs"][0] == [3]


def test_add_rt_features_multi():
    df = pd.DataFrame()
    df["rp-error"] = [1]
    df["rp-error-peptide1"] = [-1]
    df["rp-error-peptide2"] = [1]

    df["scx-error"] = [2]
    df["scx-error-peptide1"] = [-2]
    df["scx-error-peptide2"] = [2]

    df["sec-error"] = [3]
    df["sec-error-peptide1"] = [-3]
    df["sec-error-peptide2"] = [3]

    df_features_rp = features.add_rt_features(df)

    assert df_features_rp.shape == (1, 43)
    assert df_features_rp["feature_rp-error"][0] == [1]
    assert df_features_rp["feature_rp-error-peptide1"][0] == [-1]
    assert df_features_rp["feature_rp-error-peptide2"][0] == [1]

    assert df_features_rp["feature_scx-error"][0] == [2]
    assert df_features_rp["feature_scx-error-peptide1"][0] == [-2]
    assert df_features_rp["feature_scx-error-peptide2"][0] == [2]

    assert df_features_rp["feature_sec-error"][0] == [3]
    assert df_features_rp["feature_sec-error-peptide1"][0] == [-3]
    assert df_features_rp["feature_sec-error-peptide2"][0] == [3]

    assert df_features_rp["feature_initial_prod"][0] == [np.log2(1.1 * 1.1 * 1.1 * 2.1 * 2.1 * 2.1
                                                                 * 3.1 * 3.1 *3.1)]
    assert df_features_rp["feature_initial_min"][0] == [1]
    assert df_features_rp["feature_initial_max"][0] == [3]
