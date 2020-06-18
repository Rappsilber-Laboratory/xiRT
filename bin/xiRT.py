"""This module contains the script to run xiML."""
import argparse
import os
import sys
import os

import pandas as pd
import yaml

from xirt import predictor as xr
from xirt import xirtnet
import pandas as pd

from xirt import basic


if __name__ == "__main__":
    print("Hello there")
    print(basic.add_numbers(10, 20))

    print("Running {}".format(sys.argv[0]))
    print("xiML config {}".format(sys.argv[1]))
    print("xiRT config {}".format(sys.argv[2]))
    print("Working Directory: {}".format(os.getcwd()))

    # parser = argparse.ArgumentParser(
    #     description='xiRT - Retention Time Prediction for Linear and Cross-Linked Peptides '
    #                 'in Mulitiple Dimensions.')
    # parser.add_argument('-c', action='store', dest='config', help='YAML configuration file.')

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