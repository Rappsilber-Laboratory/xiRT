import os
from xirt import __main__

fixtures_loc = os.path.join(os.path.dirname(__file__), 'fixtures')


def test__main__():
    xirt_loc = os.path.join(fixtures_loc, "xirt_params_rp.yaml")
    setup_loc = os.path.join(fixtures_loc, "learning_params_training_cv.yaml")
    peptides_in = os.path.join(fixtures_loc, "50pCSMFDR_universal_final.csv")

    __main__.xirt_runner(peptides_file=peptides_in, xirt_loc=xirt_loc,
                         setup_loc=setup_loc)
    assert True
