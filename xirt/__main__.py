"""xiRT main module to run the training and prediction."""

import argparse
import os
import sys

import pandas as pd
import numpy as np
import yaml

from xirt import predictor as xr
from xirt import xirtnet


def arg_parser():
    """
    Parse the arguments from the CLI.

    Returns:
        arguments, from parse_args
    """
    description = """
    xiRT is a machine learning tool for the (multidimensional) RT prediction of linear and
    crosslinked peptides. Use --help to see the command line arguments or visit the github page:
    #TODO 
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-p", "--xirt_params", "xirt_params", meta_var="xirt_params",
                        help="YAML parameter file to control xiRT's deep learning architecture.",
                        required=True, action="store", dest="xirt_params")

    parser.add_argument("-c", "--learning_params", "learning_params", meta_var="learning_params",
                        help="YAML parameter file to control training and testing splits and data.",
                        required=True, action="store", dest="learning_params")

    args = parser.parse_args()
    return args


def xirt_runner(peptides_file, xirt_loc, learning_loc):
    xirt_params = yaml.load(open(xirt_loc), Loader=yaml.FullLoader)
    learning_params = yaml.load(open(learning_loc), Loader=yaml.FullLoader)
    matches_df = pd.read_csv(peptides_file)

    # convenience short cuts
    n_splits = learning_params["train"]["ncv"]
    test_size = learning_params["train"]["test_frac"]

    # preprocess training data
    training_data = xr.preprocess(matches_df,
                                  sequence_type=learning_params["train"]["sequence_type"],
                                  max_length=learning_params["preprocessing"]["max_length"],
                                  cl_residue=False, # learning_params["preprocessing"]["cl_residue"]
                                  fraction_cols=xirt_params["predictions"]["fractions"])

    # set training index by FDR and duplicates
    training_data.set_fdr_mask(fdr_cutoff=learning_params["train"]["fdr"])
    training_data.set_unique_shuffled_sampled_training_idx(
        sample_frac=learning_params["train"]["sample_frac"],
        random_state=learning_params["train"]["sample_state"])

    # adjust RT if necessary to guarantee smooth learning TODO remove!
    training_data.psms["rp"] = training_data.psms["RP"] / 60.0

    # init neural network structure
    xirtnetwork = xirtnet.xiRTNET(xirt_params, input_dim=training_data.features2.shape[1])

    # get the columns where the RT information is stored
    frac_cols = [xirtnetwork.output_p[tt.lower() + "-column"] for tt in
                 xirt_params["predictions"]["fractions"]]

    cont_cols = [xirtnetwork.output_p[tt.lower() + "-column"] for tt in
                 xirt_params["predictions"]["continues"]]
    metric_columns = xirtnetwork.model.metrics_names

    cv_counter = 1
    # heart of the function
    # perform crossvalidation
    # train on n-1 fold, use test_size from n-1 folds for validation and test/predict RT
    # on the remaining fold
    for train_idx, val_idx, pred_idx in training_data.iter_splits(n_splits=n_splits,
                                                                  test_size=test_size):
        # assemble the layers
        xirtnetwork.build_model(siamese=xirt_params["siamese"]["use"])

        # compile for training
        xirtnetwork.compile()

        # assemble training data
        xt_cv = training_data.get_features(train_idx)
        yt_cv = training_data.get_classes(train_idx, frac_cols=frac_cols, cont_cols=cont_cols)

        xv_cv = training_data.get_features(val_idx)
        yv_cv = training_data.get_classes(val_idx, frac_cols=frac_cols, cont_cols=cont_cols)

        # fit the mode, use the validation split to determine the best
        # epoch for selecting the best weights
        history = xirtnetwork.model.fit(xt_cv, yt_cv, validation_data=(xv_cv, yv_cv),
                                        epochs=xirt_params["learning"]["epochs"],
                                        batch_size=xirt_params["learning"]["batch_size"],
                                        verbose=xirt_params["learning"]["verbose"])

        predictions = xirtnetwork.model.predict(training_data.get_features(pred_idx))

        # store predictions
        # store metrics
        # store model? / callback
        df_history = pd.DataFrame(history.history)
        df_history["CV"] = cv_counter
        cv_counter += 1


if __name__ == "__main__":

    # parse arguments
    args = arg_parser()

    # call function
    xirt_runner(args.peptides_file, args.xirt_params, args.learning_params,
                n_splits=3, test_size=0.1)
