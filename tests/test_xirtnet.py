import os

import numpy as np
import pandas as pd
import yaml

from xirt import predictor as xr
from xirt import xirtnet

fixtures_loc = os.path.join(os.path.dirname(__file__), 'fixtures')


def test_xirt_class():
    # simple test to check if the parameter files were parsed
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params.yaml")),
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
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params.yaml")),
                           Loader=yaml.FullLoader)

    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    xirtnetwork.build_model(siamese=False)
    xirtnetwork.export_model_visualization("model_figure_normal")
    assert True


def test_xirt_siamese_model():
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params.yaml")),
                           Loader=yaml.FullLoader)

    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    xirtnetwork.build_model(siamese=True)
    xirtnetwork.export_model_visualization("model_figure_siamese")
    assert True


def test_xirt_visualization():
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params.yaml")),
                           Loader=yaml.FullLoader)

    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    xirtnetwork.build_model(siamese=True)
    # xirtnetwork.export_model_visualization("model_figure_siamese")
    # todo remove pdf
    assert True


def test_xirt_compilation():
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params.yaml")),
                           Loader=yaml.FullLoader)
    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    xirtnetwork.build_model(siamese=True)
    xirtnetwork.compile()
    assert xirtnetwork.model._is_compiled is True


def test_xirt_compilation_siameseoptions():
    # test the usage of different combination layers
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params.yaml")),
                           Loader=yaml.FullLoader)

    merge_options = ["add", "multiply", "average", "concatenate", "maximum"]
    for merge_opt in merge_options:
        xiRTconfig["siamese"]["merge_type"] = merge_opt
        xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
        xirtnetwork.build_model(siamese=True)
        xirtnetwork.compile()
        assert xirtnetwork.model._is_compiled


def test_xirt_compilation_lstm_options():
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params.yaml")),
                           Loader=yaml.FullLoader)
    rnn_options = ["LSTM", "GRU"]  # "CuDNNGRU", "CuDNNLSTM"
    for rnn_type in rnn_options:
        xiRTconfig["LSTM"]["type"] = rnn_type
        xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
        xirtnetwork.build_model(siamese=True)
        xirtnetwork.compile()
        assert xirtnetwork.model._is_compiled


def test_xirt_train():
    # read data
    matches_df = pd.read_csv(os.path.join(fixtures_loc, "50pCSMFDR_universal_final.csv"), nrows=100)
    matches_df = matches_df.sample(frac=0.5)

    # standard processing before training
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params.yaml")),
                           Loader=yaml.FullLoader)
    training_data = xr.preprocess(matches_df, "crosslink", max_length=-1, cl_residue=False,
                                  fraction_cols=["SCX", "hSAX"])
    training_data.set_fdr_mask(fdr_cutoff=0.05)
    training_data.set_unique_shuffled_sampled_training_idx()
    training_data.psms["RP"] = training_data.psms["RP"] / 60.0
    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=training_data.features2.shape[1])
    # TODO train network
    assert True


def reshapey(values):
    """Flattens the arrays that were stored in a single data frame cell, such
    that the shape is again usable for the neural network input.

    :param values:
    :return:
    """
    nrows = len(values)
    ncols = values[0].size
    return np.array([y for x in values for y in x]).reshape(nrows, ncols)
