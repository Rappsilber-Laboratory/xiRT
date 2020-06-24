import os

import numpy as np
import pandas as pd
import yaml

from xirt import predictor as xr
from xirt import xirtnet


def test_xirt_class():
    current_dir = os.path.dirname(__file__)
    xiRTconfig = yaml.load(open(current_dir + r"\fixtures\xirt_params.yaml"),
                           Loader=yaml.FullLoader)
    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    assert xirtnetwork.model is None
    assert xirtnetwork.input_dim == 100
    assert xirtnetwork.LSTM_p == xiRTconfig["LSTM"]
    assert xirtnetwork.conv_p == xiRTconfig["conv"]
    assert xirtnetwork.dense_p == xiRTconfig["dense"]
    assert xirtnetwork.embedding_p == xiRTconfig["embedding"]
    assert xirtnetwork.learning_p == xiRTconfig["learning"]
    assert xirtnetwork.output_p == xiRTconfig["output"]
    assert xirtnetwork.siamese_p == xiRTconfig["siamese"]


def test_xirt_normal_model():
    current_dir = os.path.dirname(__file__)
    xiRTconfig = yaml.load(open(current_dir + r"\fixtures\xirt_params.yaml"),
                           Loader=yaml.FullLoader)

    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    xirtnetwork.build_model(siamese=False)
    xirtnetwork.export_model_visualization("model_figure_normal")
    assert True


def test_xirt_siamese_model():
    current_dir = os.path.dirname(__file__)
    xiRTconfig = yaml.load(open(current_dir + r"\fixtures\xirt_params.yaml"),
                           Loader=yaml.FullLoader)

    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    xirtnetwork.build_model(siamese=True)
    xirtnetwork.export_model_visualization("model_figure_siamese")
    assert True


def test_xirt_visualization():
    current_dir = os.path.dirname(__file__)
    xiRTconfig = yaml.load(open(current_dir + r"\fixtures\xirt_params.yaml"),
                           Loader=yaml.FullLoader)

    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    xirtnetwork.build_model(siamese=True)
    # xirtnetwork.export_model_visualization("model_figure_siamese")
    assert True


def test_xirt_compilation():
    current_dir = os.path.dirname(__file__)
    xiRTconfig = yaml.load(open(current_dir + r"\fixtures\xirt_params.yaml"),
                           Loader=yaml.FullLoader)
    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    xirtnetwork.build_model(siamese=True)
    xirtnetwork.compile()
    assert xirtnetwork.model._is_compiled is True


def test_xirt_train():
    test_size = 0.1
    n_splits = 3
    # current_dir = os.path.dirname(__file__)
    # matches_df = pd.read_csv(current_dir + r"\fixtures\50pCSMFDR_universal_final.csv")
    matches_df = pd.read_csv(
        r"C:\\Users\\Hanjo\\Documents\\xiRT\\tests\\fixtures\\50pCSMFDR_universal_final.csv")
    matches_df = matches_df.sample(frac=0.5)

    # xiRTconfig = yaml.load(open(current_dir + r"\fixtures\xirt_params.yaml"),
    # Loader=yaml.FullLoader)
    xiRTconfig = yaml.load(
        open(r"C:\\Users\\Hanjo\\Documents\\xiRT\\tests\\fixtures\xirt_params.yaml"),
        Loader=yaml.FullLoader)

    # preprocess training data
    training_data = xr.preprocess(matches_df, "crosslink", max_length=-1, cl_residue=False,
                                  fraction_cols=["xirt_SCX", "xirt_hSAX"])
    training_data.set_fdr_mask(fdr_cutoff=0.05)
    training_data.set_unique_shuffled_sampled_training_idx()
    training_data.psms["xirt_RP"] = training_data.psms["xirt_RP"] / 60.0

    cv_counter = 1
    for train_idx, val_idx, pred_idx in training_data.iter_splits(n_splits=n_splits,
                                                                  test_size=test_size):
        break
        xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=training_data.features2.shape[1])
        xirtnetwork.build_model(siamese=True)
        xirtnetwork.compile()

        # assemble training data
        X = (training_data.features1.filter(regex="rnn").loc[train_idx],
             training_data.features2.filter(regex="rnn").loc[train_idx])
        y = [reshapey(training_data.psms["xirt_hSAX_ordinal"].loc[train_idx].values),
             reshapey(training_data.psms["xirt_SCX_ordinal"].loc[train_idx].values),
             training_data.psms["xirt_RP"].loc[train_idx].values]

        history = xirtnetwork.model.fit(X, y, epochs=50, batch_size=256, verbose=2)
        df_history = pd.DataFrame(history.history)
        df_history["CV"] = cv_counter
        cv_counter += 1


def reshapey(values):
    """Flattens the arrays that were stored in a single data frame cell, such
    that the shape is again usable for the neural network input.

    :param values:
    :return:
    """
    nrows = len(values)
    ncols = values[0].size
    return np.array([y for x in values for y in x]).reshape(nrows, ncols)
