import os

import numpy as np
import pandas as pd
import pytest
import yaml

from xirt import predictor as xr
from xirt import xirtnet as xnet

fixtures_loc = os.path.join(os.path.dirname(__file__), 'fixtures')


def test_preprocessing_crosslinks():
    psms_df = pd.read_csv(os.path.join(fixtures_loc, "50pCSMFDR_universal_final.csv")).head(100)
    training_data = xr.preprocess(psms_df, "crosslink", max_length=-1, cl_residue=False)

    assert len(training_data.features1) == 100
    assert len(training_data.features2) == 100
    assert np.any(training_data.features1 != training_data.features2)


def test_preprocessing_linear():
    psms_df = pd.read_csv(os.path.join(fixtures_loc, "50pCSMFDR_universal_final.csv")).head(100)
    training_data = xr.preprocess(psms_df, "linear", max_length=-1, cl_residue=False)
    assert len(training_data.features1) == 100
    assert len(training_data.features2) == 0


def test_ModelData():
    data = xr.ModelData(pd.DataFrame(), [1], [2], None)
    assert data.psms.empty
    assert data.features1 == [1]
    assert data.features2 == [2]
    assert data.le is None


def test_model_data_trainable():
    # simple test to check fdr and TT filter
    # one pair above FDR, one not TT
    peptide1 = ["Peptide" + str(i) for i in np.arange(1, 6)]
    peptide2 = ["Peptide" + str(i) for i in np.arange(11, 16)]
    fdr = np.array([0, 0.01, 0.02, 0.03, 0.056])
    fasta1 = ["HUMAN"] * 5
    fasta2 = ["HUMAN"] * 5
    isTT = [True, True, True, False, True]
    data = xr.ModelData(
        pd.DataFrame(np.column_stack([peptide1, peptide2, fasta1, fasta2]),
                     columns=["Peptide1", "Peptide2", "Fasta1", "Fasta2"]),
        [], [], None)
    data.psms["FDR"] = fdr
    data.psms["isTT"] = isTT
    data.set_fdr_mask(0.05)

    exp_trainable = np.array([True, True, True, False, False])
    assert np.all(data.psms["fdr_mask"].values == exp_trainable)


def test_model_data_shuffle():
    peptide1 = ["Peptide" + str(i) for i in np.arange(1, 6)]
    peptide2 = ["Peptide" + str(i) for i in np.arange(11, 16)]
    fdr_mask = np.array([True, True, True, False, False])
    is_duplicate = [False, False, True, True, True]
    data = xr.ModelData(
        pd.DataFrame(np.column_stack([peptide1, peptide2]), columns=["Peptide1", "Peptide2"]),
        [], [], None)
    data.psms["fdr_mask"] = fdr_mask
    data.psms["Duplicate"] = is_duplicate
    data.set_unique_shuffled_sampled_training_idx(sample_frac=1, random_state=42)

    # viable training ids, order should be shuffled -> != comparison
    exp_train_idx = [0, 1]
    # viable predicton rows, must not be shuffled
    exp_predict_idx = [2, 3, 4]

    assert data.train_idx.tolist() != exp_train_idx
    assert data.predict_idx.tolist() == exp_predict_idx
    assert sorted(data.train_idx.tolist()) == exp_train_idx
    assert sorted(data.predict_idx.tolist()) == exp_predict_idx


def test_iter_splits():
    psms_df = pd.read_csv(os.path.join(fixtures_loc, "50pCSMFDR_universal_final.csv"),
                          nrows=10)
    training_data = xr.preprocess(psms_df, "crosslink", max_length=-1, cl_residue=False)
    training_data.set_fdr_mask(0.4)

    # test error with small validation set
    with pytest.raises(ValueError):
        for train_idx, val_idx, pred_idx in training_data.iter_splits(3, 0.1):
            pass

    training_data.set_unique_shuffled_sampled_training_idx(sample_frac=1, random_state=42)
    with pytest.raises(ValueError):
        for train_idx, val_idx, pred_idx in training_data.iter_splits(3, 0):
            pass

    folds = []
    for train_idx, val_idx, pred_idx in training_data.iter_splits(3, 0.1):
        folds.append([train_idx, val_idx, pred_idx])

    # check if folds are returned correctly
    assert len(folds) == 3


def test_get_train_psms():
    peptide1 = ["Peptide" + str(i) for i in np.arange(1, 6)]
    peptide2 = ["Peptide" + str(i) for i in np.arange(11, 16)]
    data = xr.ModelData(
        pd.DataFrame(np.column_stack([peptide1, peptide2]), columns=["Peptide1", "Peptide2"]),
        [], [], None)
    data.psms["FDR"] = [0, 0.01, 0.05, 0.1, 0.2]
    data.psms["isTT"] = True
    data.psms["Duplicate"] = [False, False, False, False, False]
    data.set_fdr_mask(0.05)
    data.set_unique_shuffled_sampled_training_idx()
    assert np.all(data.get_train_psms().index == np.array([0, 1, 2]))
    assert np.all(data.get_predict_psms().index == np.array([3, 4]))


def test_get_classes_cont():
    psms_df = pd.read_csv(os.path.join(fixtures_loc, "50pCSMFDR_universal_final.csv")).head(100)
    training_data = xr.preprocess(psms_df, "crosslink", max_length=-1, cl_residue=False)
    idx = training_data.psms.index[0:10]
    rp = training_data.get_classes(idx, frac_cols=[], cont_cols=["RP"])
    assert len(rp) == 1
    assert rp[0].shape == (10,)


def test_get_classes_frac():
    psms_df = pd.read_csv(os.path.join(fixtures_loc, "50pCSMFDR_universal_final.csv")).head(100)
    psms_df["SCX"] = np.repeat([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], 10)
    training_data = xr.preprocess(psms_df, "crosslink", max_length=-1, cl_residue=False,
                                  fraction_cols=["SCX"])
    idx = training_data.psms.index
    SCX = training_data.get_classes(idx, frac_cols=["SCX_ordinal"], cont_cols=[])
    assert len(SCX) == 1
    assert SCX[0].shape == (100, 5)


def test_get_classes_cont_frac():
    psms_df = pd.read_csv(os.path.join(fixtures_loc, "50pCSMFDR_universal_final.csv")).head(100)
    psms_df["SCX"] = np.repeat([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], 10)

    training_data = xr.preprocess(psms_df, "crosslink", max_length=-1, cl_residue=False,
                                  fraction_cols=["SCX"])
    idx = training_data.psms.index
    rts = training_data.get_classes(idx, frac_cols=["SCX_ordinal"], cont_cols=["RP"])
    assert len(rts) == 2
    assert rts[0][0].shape == (100, 5)
    assert rts[1][0].shape == (100, )


def test_store_predictions():
    psms_df = pd.read_csv(os.path.join(fixtures_loc, "50pCSMFDR_universal_final.csv")).head(3)
    training_data = xr.preprocess(psms_df, "crosslink", max_length=-1, cl_residue=False,
                                  fraction_cols=["SCX"])

    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params_3RT.yaml")),
                           Loader=yaml.FullLoader)
    xirtnetwork = xnet.xiRTNET(xiRTconfig, input_dim=100)

    store_idx = psms_df.iloc[0:3].index
    xirtnetwork.tasks = ["scx", "rp", "hsax"]
    xirtnetwork.output_p["scx-activation"] = "softmax"
    xirtnetwork.output_p["hsax-activation"] = "sigmoid"
    # 3 classes fake
    predictions = [np.array([[0, 0.1, 0.9], [0.5, 0.4, 0.3], [0.2, 0.8, 0.2]]),
                   [1, 2, 3],
                   np.array([[0.4, 0.24, 0.1], [0.6, 0.24, 0.1], [0.9, 0.6, 0.5]])]

    training_data.store_predictions(xirtnetwork, predictions, store_idx)

    exp_idx = psms_df.index
    exp_scx = [3, 1, 2]
    exp_rp = [1, 2, 3]
    exp_hsax = [1, 2, 3]
    assert np.all(training_data.prediction_df.loc[exp_idx]["scx-prediction"] == exp_scx)
    assert np.all(training_data.prediction_df.loc[exp_idx]["rp-prediction"] == exp_rp)
    assert np.all(training_data.prediction_df.loc[exp_idx]["hsax-prediction"] == exp_hsax)


def test_sigmoid_to_class():
    predictions = np.array([[0.4, 0.24, 0.1],  # 1class
                            [0.6, 0.49, 0.3],  # 2nd class
                            [0.8, 0.7, 0.6],  # 3rd class
                            [0.8, 0.7, 0.4]])  # 3rd clas
    exp_preds = [0, 1, 2, 2]
    preds = xr.sigmoid_to_class(predictions, 0.5)
    assert np.all(preds == exp_preds)
