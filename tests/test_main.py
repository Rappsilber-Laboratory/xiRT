import os
from xirt import __main__

fixtures_loc = os.path.join(os.path.dirname(__file__), 'fixtures')


def test_xirt_runner_rp_crosslinks_cv(tmpdir):
    xirt_loc = os.path.join(fixtures_loc, "xirt_params_rp.yaml")
    setup_loc = os.path.join(fixtures_loc, "learning_params_training_cv.yaml")
    peptides_in = os.path.join(fixtures_loc, "DSS_xisearch_fdr_CSM50percent.csv")

    __main__.xirt_runner(peptides_file=peptides_in, out_dir=tmpdir.mkdir("xiRT_results"),
                         xirt_loc=xirt_loc, setup_loc=setup_loc, nrows=1000)
    assert True


def test_xirt_runner_rp_crosslinks_train(tmpdir):
    # test xurt with RP, crosslinks in the training mode
    pass


def test_xirt_runner_rp_crosslinks_predict(tmpdir):
    # test xurt with RP, crosslinks in the predict mode, (requires trained weights)
    pass


def test_xirt_runner_rp_crosslinks_cv_refit():
    # test xirt with rp, crosslinks, cv mode and refit the classifier
    pass


def test_xirt_runner_3d_crosslinks_cv():
    pass


def test_xirt_linear_rp():
    pass


def test_xirt_pseudolinear():
    pass


def test_arg_parser():
    print("test")
    parser = __main__.arg_parser()
    assert len(parser.description) > 1
