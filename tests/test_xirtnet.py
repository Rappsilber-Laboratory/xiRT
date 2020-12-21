import os

import numpy as np
import pytest
import yaml

from xirt import xirtnet

fixtures_loc = os.path.join(os.path.dirname(__file__), 'fixtures')


def test_xirt_class():
    # simple test to check if the parameter files were parsed
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params_3RT.yaml")),
                           Loader=yaml.FullLoader)
    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    assert xirtnetwork.model is None
    assert xirtnetwork.input_dim == 100
    assert xirtnetwork.LSTM_p == xiRTconfig["LSTM"]
    assert xirtnetwork.dense_p == xiRTconfig["dense"]
    assert xirtnetwork.embedding_p == xiRTconfig["embedding"]
    assert xirtnetwork.learning_p == xiRTconfig["learning"]
    assert xirtnetwork.output_p == xiRTconfig["output"]
    assert xirtnetwork.siamese_p == xiRTconfig["siamese"]


def test_xirt_compilation():
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params_3RT.yaml")),
                           Loader=yaml.FullLoader)
    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    xirtnetwork.build_model(siamese=True)
    xirtnetwork.compile()
    assert xirtnetwork.model._is_compiled is True


def test_xirt_compilation_sgd():
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params_3RT.yaml")),
                           Loader=yaml.FullLoader)
    xiRTconfig["learning"]["optimizer"] = "sgd"
    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    xirtnetwork.build_model(siamese=True)
    xirtnetwork.compile()
    assert xirtnetwork.model.optimizer._name.lower() == "sgd"


def test_xirt_compilation_rmsprop():
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params_3RT.yaml")),
                           Loader=yaml.FullLoader)
    xiRTconfig["learning"]["optimizer"] = "rmsprob"
    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    xirtnetwork.build_model(siamese=True)
    xirtnetwork.compile()
    assert xirtnetwork.model.optimizer._name.lower() == "rmsprop"


def test_xirt_compilation_nadam():
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params_3RT.yaml")),
                           Loader=yaml.FullLoader)
    xiRTconfig["learning"]["optimizer"] = "nadam"
    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    xirtnetwork.build_model(siamese=True)
    xirtnetwork.compile()
    assert xirtnetwork.model.optimizer._name.lower() == "nadam"


def test_xirt_compilation_siameseoptions():
    # test the usage of different combination layers
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params_3RT.yaml")),
                           Loader=yaml.FullLoader)

    merge_options = ["add", "multiply", "average", "concatenate", "maximum"]
    for merge_opt in merge_options:
        xiRTconfig["siamese"]["merge_type"] = merge_opt
        xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
        xirtnetwork.build_model(siamese=True)
        xirtnetwork.compile()
        assert xirtnetwork.model._is_compiled


def test_xirt_compilation_lstm_options():
    # test if compilaton works for all options (gpu not added ...)
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params_3RT.yaml")),
                           Loader=yaml.FullLoader)
    rnn_options = ["LSTM", "GRU"]  # "CuDNNGRU", "CuDNNLSTM"
    for rnn_type in rnn_options:
        xiRTconfig["LSTM"]["type"] = rnn_type
        xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
        xirtnetwork.build_model(siamese=True)
        xirtnetwork.compile()
        assert xirtnetwork.model._is_compiled


# def test_xirt_train():
#     # read data
#     matches_df = pd.read_csv(os.path.join(fixtures_loc, "DSS_xisearch_fdr_CSM50percent.csv"),
#     nrows=100)
#     matches_df = matches_df.sample(frac=0.5)
#
#     # standard processing before training
#     xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params_3RT.yaml")),
#                            Loader=yaml.FullLoader)
#     training_data = xr.preprocess(matches_df, "crosslink", max_length=-1, cl_residue=False,
#                                   fraction_cols=["SCX", "hSAX"])
#     training_data.set_fdr_mask(fdr_cutoff=0.05)
#     training_data.set_unique_shuffled_sampled_training_idx()
#     training_data.psms["RP"] = training_data.psms["RP"] / 60.0
#     xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=training_data.features2.shape[1])
#     # TODO train network
#     assert True


def test_get_callbacks(tmpdir):
    # test currently excludes tensorboard, all other callbacks sum up to 6 (2x ModelCheckpoint)
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params_3RT.yaml")),
                           Loader=yaml.FullLoader)
    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    xirtnetwork.callback_p["callback_path"] = os.path.abspath(tmpdir.mkdir("tmp"))
    callbacks = xirtnetwork.get_callbacks("test")
    assert len(callbacks) == 6


def test_init_regularizer_l1():
    reg_tmp = xirtnet.xiRTNET._init_regularizer("l1", 0.1)
    assert np.allclose([reg_tmp.get_config()["l1"]], [0.1], 0.001)


def test_init_regularizer_l2():
    reg_tmp = xirtnet.xiRTNET._init_regularizer("l2", 0.1)
    assert np.allclose([reg_tmp.get_config()["l2"]], [0.1], 0.001)


def test_init_regularizer_l1l2():
    reg_tmp = xirtnet.xiRTNET._init_regularizer("l1l2", 0.1)
    assert np.allclose([reg_tmp.get_config()["l1"]], [0.1], 0.001)
    assert np.allclose([reg_tmp.get_config()["l2"]], [0.1], 0.001)


def test_init_regularizer_l3():
    # test raising error for l3 (doesnt exist)
    with pytest.raises(KeyError):
        xirtnet.xiRTNET._init_regularizer("l3", 0.1)


def test_print_layers(capsys):
    # simple test that checks if key info is contained in the printed output
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params_3RT.yaml")),
                           Loader=yaml.FullLoader)
    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=10)
    xirtnetwork.build_model(siamese=True)
    xirtnetwork.print_layers()
    captured = capsys.readouterr()
    assert "rp" in captured.out
    assert "siamese" in captured.out


def test_print_parameters(capsys):
    # simple test that checks if key info is contained in the printed output
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params_3RT.yaml")),
                           Loader=yaml.FullLoader)
    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=10)
    xirtnetwork.build_model(siamese=True)
    xirtnetwork.get_param_overview()
    captured = capsys.readouterr()
    assert "Total params:" in captured.out
    assert "Trainable params:" in captured.out
    assert "Non-trainable params:" in captured.out


def test_params_to_df(tmpdir):
    # take yaml file and test to df function
    p = tmpdir.mkdir("tmp").join("params.csv")
    params_df = xirtnet.params_to_df(os.path.join(fixtures_loc, "xirt_params_3RT.yaml"), p)
    assert os.path.isfile(p)
    assert not params_df.empty
    os.remove(p)


def test_adjust():
    xiRTconfig = yaml.load(open(os.path.join(fixtures_loc, "xirt_params_3RT.yaml")),
                           Loader=yaml.FullLoader)
    xirtnetwork = xirtnet.xiRTNET(xiRTconfig, input_dim=100)
    xirtnetwork.build_model(siamese=True)
    # was 9 in config
    xirtnetwork.output_p["scx-dimension"] = 5
    # was 10
    xirtnetwork.output_p["hsax-dimension"] = 5

    # difference should be:
    # hsax: 10 (neurons) * 10 = 100 + 10 = 110
    # hsax: 10 (neurons) * 5 = 50 + 5 = 55
    # scx: 10 (neurons) * 5 = 50 + 5 = 55
    # scx: 10 (neurons) * 9 = 90 + 9 = 99
    exp_diff = 110 + 99 - 55 - 55
    params_old = xirtnetwork.model.count_params()

    # adjust model
    xirtnetwork.adjust_model()
    assert (params_old - xirtnetwork.model.count_params()) == exp_diff
