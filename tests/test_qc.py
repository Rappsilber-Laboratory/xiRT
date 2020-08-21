from xirt import qc
import numpy as np


def test_custom_r2():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    r2 = qc.custom_r2(x, y)
    exp_r2 = 1.0
    assert r2 == exp_r2


def test_relaxed_accuracy():
    x = np.array([1, 2, 3, 4, 5])
    # 5 = 1 off, 7 = 2 off
    # --> 4/5 = 80% = 0.8
    y = np.array([1, 2, 3, 5, 7])
    racc = qc.relaxed_accuracy(x, y)
    racc_exp = 0.8
    assert racc == racc_exp


def test_plot_cv_eval():
    pass


def test_plot_split_eval():
    pass


def test_epoch_qc_cv():
    pass


def test_save_fig():
    pass


def test_add_scatter():
    pass


def test_add_heatmpa():
    pass
