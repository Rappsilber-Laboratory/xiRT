import os

import pandas as pd
import numpy as np

from xirt import predictor as xr

fixtures_loc = os.path.join(os.path.dirname(__file__), 'fixtures')


def test_preprocessing_crosslinks():
    psms_df = pd.read_csv(fixtures_loc + r"\50pCSMFDR_universal_final.csv")
    training_data = xr.preprocess(psms_df, "crosslink", max_length=-1, cl_residue=False)
    # TODO test something
    assert True


def test_preprocessing_linear():
    psms_df = pd.read_csv(
        r"fixtures\\50pCSMFDR_universal_final.csv")
#        r"C:\\Users\\Hanjo\\Documents\\xiRT\\tests\\fixtures\\50pCSMFDR_universal_final.csv")
    training_data = xr.preprocess(psms_df, "linear", max_length=-1, cl_residue=False)
    # TODO test something
    assert True


def test_data():
    data = xr.model_data(pd.DataFrame(), [1], [2], None)
    assert data.psms == pd.DataFrame()
    assert data.features1 == [1]
    assert data.features2 == [2]
    assert data.le == pd.DataFrame()


def test_model_data_trainable():
    # simple test to check fdr and TT filter
    # one pair above FDR, one not TT
    peptide1 = ["Peptide" + str(i) for i in np.arange(1, 6)]
    peptide2 = ["Peptide" + str(i) for i in np.arange(11, 16)]
    fdr = np.array([0, 0.01, 0.02, 0.03, 0.056])
    fasta1 = ["HUMAN"] * 5
    fasta2 = ["HUMAN"] * 5
    isTT = [True, True, True, False, True]
    data = xr.model_data(
        pd.DataFrame(np.column_stack([peptide1, peptide2, fasta1, fasta2]),
                     columns=["Peptide1", "Peptide2", "Fasta1", "Fasta2"]),
        [], [], None)
    data.psms["FDR"] = fdr
    data.psms["isTT"] = isTT
    data.set_trainable_psms(0.05)

    exp_trainable = np.array([True, True, True, False, False])
    assert np.all(data.psms["fdr_mask"].values == exp_trainable)


def test_model_data_shuffle():
    peptide1 = ["Peptide" + str(i) for i in np.arange(1, 6)]
    peptide2 = ["Peptide" + str(i) for i in np.arange(11, 16)]
    fdr_mask = np.array([True, True, True, False, False])
    is_duplicate = [False, False, True, True, True]
    data = xr.model_data(
        pd.DataFrame(np.column_stack([peptide1, peptide2]), columns=["Peptide1", "Peptide2"]),
        [], [], None)
    data.psms["fdr_mask"] = fdr_mask
    data.psms["Duplicate"] = is_duplicate
    data.set_unique_shuffled_sampled_training_idx(sample_frac=1, random_state=42)

    exp_train_idx = [0, 1]
    exp_predict_idx = [2, 3, 4]

    assert data.train_idx.tolist() != exp_train_idx
    assert data.predict_idx.tolist() != exp_predict_idx
    assert sorted(data.train_idx.tolist()) == exp_train_idx
    assert sorted(data.predict_idx.tolist()) == exp_predict_idx


def test_model_data_set_cv3():
    ncv = 3
    peptide1 = ["Peptide" + str(i) for i in np.arange(1, 101)]
    peptide2 = ["Peptide" + str(i) for i in np.arange(101, 201)]
    features1 = pd.DataFrame(np.random.rand(len(peptide1), 50))
    features1.index = np.arange(100, 200)
    data = xr.model_data(
        pd.DataFrame(np.column_stack([peptide1, peptide2]), columns=["Peptide1", "Peptide2"]),
        features1, [], None)
    data.train_idx = np.arange(100, 180)
    data.predict_idx = np.arange(180, 200)
    data.set_cv_indices(ncv=ncv)

    assert data.cv_folds_ar == np.arange(1, ncv+1)
    assert len(data.train_idx) != len(np.unique(np.concatenate(data.cv_indices_ar)))


def test_model_data_set_cv1():
    ncv = 1
    peptide1 = ["Peptide" + str(i) for i in np.arange(1, 101)]
    peptide2 = ["Peptide" + str(i) for i in np.arange(101, 201)]
    features1 = pd.DataFrame(np.random.rand(len(peptide1), 50))
    features1.index = np.arange(100, 200)
    data = xr.model_data(
        pd.DataFrame(np.column_stack([peptide1, peptide2]), columns=["Peptide1", "Peptide2"]),
        features1, [], None)
    data.train_idx = np.arange(100, 180)
    data.predict_idx = np.arange(180, 200)
    data.set_cv_indices(ncv=ncv)

    assert data.cv_folds_ar == np.arange(1, ncv+1)
    assert len(data.train_idx) != len(np.unique(data.cv_indices_ar))
