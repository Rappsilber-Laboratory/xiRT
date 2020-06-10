import os

import pandas as pd

from xirt import predictor as xr


def test_preprocessing_crosslinks():
    current_dir = os.path.dirname(__file__)
    psms_df = pd.read_csv(r"\tests\fixtures\50pCSMFDR_universal_final.csv")
#        r"C:\\Users\\Hanjo\\Documents\\xiRT\\tests\\fixtures\\50pCSMFDR_universal_final.csv")
    training_data = xr.preprocess(psms_df, "crosslink", max_length=-1, cl_residue=False)
    assert True


def test_preprocessing_linear():
    psms_df = pd.read_csv(
        r"fixtures\\50pCSMFDR_universal_final.csv")
#        r"C:\\Users\\Hanjo\\Documents\\xiRT\\tests\\fixtures\\50pCSMFDR_universal_final.csv")
    training_data = xr.preprocess(psms_df, "linear", max_length=-1, cl_residue=False)
    assert True


def test_data():
    data = xr.data(pd.DataFrame(), [], [], None)
    assert data.psms == pd.DataFrame()
    assert data.features1 == []
    assert data.features2 == []
    assert data.le == pd.DataFrame()