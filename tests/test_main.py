import os

import yaml

from xirt import __main__

fixtures_loc = os.path.join(os.path.dirname(__file__), 'fixtures')


def test_xirt_runner_rp_crosslinks_cv(tmpdir):
    xirt_loc = os.path.join(fixtures_loc, "xirt_params_rp.yaml")
    setup_loc = os.path.join(fixtures_loc, "learning_params_training_cv.yaml")
    peptides_in = os.path.join(fixtures_loc, "DSS_xisearch_fdr_CSM50percent.csv")

    with open(xirt_loc, "r") as file:
        xirt_params = yaml.safe_load(file)

    with open(setup_loc, "r") as file:
        learning_params = yaml.safe_load(file)

    __main__.xirt_runner(
        peptides_file=peptides_in,
        out_dir=tmpdir.mkdir("xiRT_results"),
        xirt_params=xirt_params,
        learning_params=learning_params,
        nrows=500,
        perform_qc=False,
        write_dummy=False
    )
    assert True


def test_xirt_runner_rp_crosslinks_train(tmpdir):
    # test xirt with RP, crosslinks in the training mode
    xirt_loc = os.path.join(fixtures_loc, "xirt_params_rp.yaml")
    setup_loc = os.path.join(fixtures_loc, "learning_params_training_nocv.yaml")
    peptides_in = os.path.join(fixtures_loc, "DSS_xisearch_fdr_CSM50percent.csv")

    with open(xirt_loc, "r") as file:
        xirt_params = yaml.safe_load(file)

    with open(setup_loc, "r") as file:
        learning_params = yaml.safe_load(file)

    __main__.xirt_runner(
        peptides_file=peptides_in,
        out_dir=tmpdir.mkdir("xiRT_results"),
        xirt_params=xirt_params,
        learning_params=learning_params,
        nrows=500,
        perform_qc=False,
        write_dummy=False
    )
    assert True


def test_xirt_runner_rp_crosslinks_predict(tmpdir):
    # test xirt with RP, crosslinks in the predict mode, (requires trained weights)
    pass


def test_xirt_runner_rp_crosslinks_cv_refit(tmpdir):
    # test xirt with rp, crosslinks, cv mode and refit the classifier
    # test xirt with RP, crosslinks in the training mode
    xirt_loc = os.path.join(fixtures_loc, "xirt_params_rp.yaml")
    setup_loc = os.path.join(fixtures_loc, "learning_params_training_nocv.yaml")
    peptides_in = os.path.join(fixtures_loc, "DSS_xisearch_fdr_CSM50percent.csv")

    with open(xirt_loc, "r") as file:
        xirt_params = yaml.safe_load(file)

    with open(setup_loc, "r") as file:
        learning_params = yaml.safe_load(file)

    __main__.xirt_runner(
        peptides_file=peptides_in,
        out_dir=tmpdir.mkdir("xiRT_results"),
        xirt_params=xirt_params,
        learning_params=learning_params,
        nrows=500,
        perform_qc=False,
        write_dummy=False
    )
    assert True


def test_xirt_runner_scx_crosslinks_cv_refit(tmpdir):
    # test xirt with rp, crosslinks, cv mode and refit the classifier
    # test xirt with RP, crosslinks in the training mode
    xirt_loc = os.path.join(fixtures_loc, "xirt_params_scx.yaml")
    setup_loc = os.path.join(fixtures_loc, "learning_params_training_cv.yaml")
    peptides_in = os.path.join(fixtures_loc, "DSS_xisearch_fdr_CSM50percent.csv")

    with open(xirt_loc, "r") as file:
        xirt_params = yaml.safe_load(file)

    with open(setup_loc, "r") as file:
        learning_params = yaml.safe_load(file)

    __main__.xirt_runner(
        peptides_file=peptides_in,
        out_dir=tmpdir.mkdir("xiRT_results"),
        xirt_params=xirt_params,
        learning_params=learning_params,
        nrows=1000,
        perform_qc=False,
        write_dummy=False
    )
    assert True


def test_xirt_runner_3d_crosslinks_cv(tmpdir):
    xirt_loc = os.path.join(fixtures_loc, "xirt_params_3RT.yaml")
    setup_loc = os.path.join(fixtures_loc, "learning_params_training_nocv.yaml")
    peptides_in = os.path.join(fixtures_loc, "DSS_xisearch_fdr_CSM50percent.csv")

    with open(xirt_loc, "r") as file:
        xirt_params = yaml.safe_load(file)

    with open(setup_loc, "r") as file:
        learning_params = yaml.safe_load(file)

    __main__.xirt_runner(
        peptides_file=peptides_in,
        out_dir=tmpdir.mkdir("xiRT_results"),
        xirt_params=xirt_params,
        learning_params=learning_params,
        nrows=500,
        perform_qc=False,
        write_dummy=False
    )
    assert True


def test_xirt_linear_rp(tmpdir):
    xirt_loc = os.path.join(fixtures_loc, "xirt_params_rp_linear.yaml")
    setup_loc = os.path.join(fixtures_loc, "learning_params_training_nocv_linear.yaml")
    peptides_in = os.path.join(fixtures_loc, "DSS_xisearch_fdr_CSM50percent.csv")

    with open(xirt_loc, "r") as file:
        xirt_params = yaml.safe_load(file)

    with open(setup_loc, "r") as file:
        learning_params = yaml.safe_load(file)

    __main__.xirt_runner(
        peptides_file=peptides_in,
        out_dir=tmpdir.mkdir("xiRT_results"),
        xirt_params=xirt_params,
        learning_params=learning_params,
        nrows=500,
        perform_qc=False,
        write_dummy=False
    )
    assert True


def test_xirt_pseudolinear(tmpdir):
    xirt_loc = os.path.join(fixtures_loc, "xirt_params_3RT.yaml")
    setup_loc = os.path.join(fixtures_loc, "learning_params_training_nocv_pseudolinear.yaml")
    peptides_in = os.path.join(fixtures_loc, "DSS_xisearch_fdr_CSM50percent.csv")

    with open(xirt_loc, "r") as file:
        xirt_params = yaml.safe_load(file)

    with open(setup_loc, "r") as file:
        learning_params = yaml.safe_load(file)

    __main__.xirt_runner(
        peptides_file=peptides_in,
        out_dir=tmpdir.mkdir("xiRT_results"),
        xirt_params=xirt_params,
        learning_params=learning_params,
        nrows=500,
        perform_qc=False,
        write_dummy=False
    )
    assert True


def test_arg_parser():
    print("test")
    parser = __main__.arg_parser()
    assert len(parser.description) > 1
