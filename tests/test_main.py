import os
from xirt import __main__

fixtures_loc = os.path.join(os.path.dirname(__file__), 'fixtures')

def test_main():
    xirt_loc = os.path.join(fixtures_loc, "xirt_params.yaml")
    learning_loc = os.path.join(fixtures_loc, "learning_params.yaml")
    peptides_in = os.path.join(fixtures_loc, "50pCSMFDR_universal_final.csv")

    __main__.xirt_runner(peptides_file=peptides_in, xirt_loc=xirt_loc,
                         learning_loc=learning_loc)
