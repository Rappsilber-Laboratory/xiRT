"""xiRT main module to run the training and prediction."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

from xirt import features as xf
from xirt import predictor as xr
from xirt import xirtnet


def arg_parser(arg_list):  # pragma: not covered
    """
    Parse the arguments from the CLI.

    Args:
        arg_list: ar-like, parameters as list from CLI input

    Returns:
        arguments, from parse_args
    """
    description = """
    xiRT is a machine learning tool for the (multidimensional) RT prediction of linear and
    crosslinked peptides. Use --help to see the command line arguments or visit the github page:
    #TODO
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--in_peptides",
                        help="Input peptide table to learn (and predict) the retention times.",
                        required=True, action="store", dest="in_peptides")

    parser.add_argument("-o", "--out_dir",
                        help="Directory to store the results",
                        required=True, action="store", dest="out_dir")

    parser.add_argument("-p", "--xirt_params",
                        help="YAML parameter file to control xiRT's deep learning architecture.",
                        required=True, action="store", dest="xirt_params")

    parser.add_argument("-c", "--learning_params",
                        help="YAML parameter file to control training and testing splits and data.",
                        required=True, action="store", dest="learning_params")

    args = parser.parse_args(arg_list)
    return args


def xirt_runner(peptides_file, out_dir, xirt_loc, setup_loc, nrows=None):
    """
    Execute xiRT, train a model or generate predictions for RT across multiple RT domains.

    Args:
        peptides_file: str, location of the input psm/csm file
        out_dir: str, folder to store the results to
        xirt_loc: str, location of the yaml file for xirt
        setup_loc: str, location of the setup yaml
        single_pep_predictions:
        nrows: int, number of rows to sample (for quicker testing purposes only)

    Returns:
        None
    """
    xirt_params = yaml.load(open(xirt_loc), Loader=yaml.FullLoader)
    learning_params = yaml.load(open(setup_loc), Loader=yaml.FullLoader)
    matches_df = pd.read_csv(peptides_file, nrows=nrows)

    # convenience short cuts
    n_splits = learning_params["train"]["ncv"]
    test_size = learning_params["train"]["test_frac"]
    outpath = os.path.abspath(out_dir)
    xirt_params["callbacks"]["callback_path"] = os.path.join(outpath, "callbacks")

    # preprocess training data
    training_data = xr.preprocess(matches_df,
                                  sequence_type=learning_params["train"]["sequence_type"],
                                  max_length=learning_params["preprocessing"]["max_length"],
                                  cl_residue=False,
                                  # learning_params["preprocessing"]["cl_residue"]
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

    # init data structures for results
    histories = []
    model_summary = []

    cv_counter = 1
    # perform crossvalidation
    # train on n-1 fold, use test_size from n-1 folds for validation and test/predict RT
    # on the remaining fold
    for train_idx, val_idx, pred_idx in training_data.iter_splits(n_splits=n_splits,
                                                                  test_size=test_size):
        # init the network model
        xirtnetwork.build_model(siamese=xirt_params["siamese"]["use"])
        xirtnetwork.compile()
        callbacks = xirtnetwork.get_callbacks(suffix=str(cv_counter).zfill(2))

        # assemble training data
        xt_cv = training_data.get_features(train_idx)
        yt_cv = training_data.get_classes(train_idx, frac_cols=frac_cols, cont_cols=cont_cols)
        # validation data
        xv_cv = training_data.get_features(val_idx)
        yv_cv = training_data.get_classes(val_idx, frac_cols=frac_cols, cont_cols=cont_cols)
        # prediction data
        xp_cv = training_data.get_features(pred_idx)
        yp_cv = training_data.get_classes(pred_idx, frac_cols=frac_cols, cont_cols=cont_cols)

        # fit the mode, use the validation split to determine the best
        # epoch for selecting the best weights
        history = xirtnetwork.model.fit(xt_cv, yt_cv, validation_data=(xv_cv, yv_cv),
                                        epochs=xirt_params["learning"]["epochs"],
                                        batch_size=xirt_params["learning"]["batch_size"],
                                        verbose=xirt_params["learning"]["verbose"],
                                        callbacks=callbacks)
        metric_columns = xirtnetwork.model.metrics_names

        # model evaluation training (t), validation (v), prediction (p)
        model_summary.append(xirtnetwork.model.evaluate(xt_cv, yt_cv, batch_size=512))
        model_summary.append(xirtnetwork.model.evaluate(xv_cv, yv_cv, batch_size=512))
        model_summary.append(xirtnetwork.model.evaluate(xp_cv, yp_cv, batch_size=512))

        # use the training
        training_data.predict_and_store(xirtnetwork, xp_cv, pred_idx)

        # store metrics
        # store model? / callback
        df_history = pd.DataFrame(history.history)
        df_history["CV"] = cv_counter
        cv_counter += 1
        histories.append(df_history)

    print("Done Training")
    # store model summary data
    model_summary_df = pd.DataFrame(model_summary, columns=metric_columns)
    model_summary_df["CV"] = np.repeat(np.arange(1, n_splits + 1), 3)
    model_summary_df["Split"] = np.tile(["Train", "Validation", "Prediction"], 3)

    # CV training done, now deal with the data not used for training
    refit = False
    if refit:
        callbacks = xirtnetwork.get_callbacks(suffix="full")
        xrefit = training_data.get_features(training_data.train_idx)
        yrefit = training_data.get_classes(training_data.train_idx, frac_cols=frac_cols,
                                           cont_cols=cont_cols)
        xirtnetwork.build_model(siamese=xirt_params["siamese"]["use"])
        xirtnetwork.compile()
        _ = xirtnetwork.model.fit(xrefit, yrefit, validation_split=test_size,
                                  epochs=xirt_params["learning"]["epochs"],
                                  batch_size=xirt_params["learning"]["batch_size"],
                                  verbose=xirt_params["learning"]["verbose"],
                                  callbacks=callbacks)
    else:
        # load the best performing model across cv from the validation split
        best_model_idx = np.argmin(
            model_summary_df[model_summary_df["Split"] == "Validation"]["loss"])
        xirtnetwork.model.load_weights(os.path.join(xirtnetwork.callback_p["callback_path"],
                                                    "xirt_weights_{}.h5".format(
                                                        str(best_model_idx + 1).zfill(2))))

    # get the 'unvalidation data', e.g. data that was not used during training becasue
    # here the CSMs are that we want to save / resore later!
    xu = training_data.get_features(training_data.predict_idx)
    yu = training_data.get_classes(training_data.predict_idx, frac_cols=frac_cols,
                                   cont_cols=cont_cols)
    training_data.predict_and_store(xirtnetwork, xu, training_data.predict_idx)
    eval_unvalidation = xirtnetwork.model.evaluate(xu, yu, batch_size=512)
    eval_unvalidation.extend([-1, "Unvalidation"])
    model_summary_df.loc[len(model_summary_df)] = eval_unvalidation

    # collect epoch training data
    df_history_all = pd.concat(histories)
    df_history_all = df_history_all.reset_index(drop=False).rename(columns={"index": "epoch"})
    df_history_all["epoch"] += 1

    # compute features
    xf.compute_prediction_errors(training_data.psms, training_data.prediction_df,
                                 xirtnetwork.tasks, xirtnetwork.siamese_p["single_predictions"])

    # create more features?

    # store results
    df_history_all.to_excel(os.path.join(outpath, "temporary_test.xls"))
    training_data.prediction_df.to_excel(os.path.join(outpath, "temporary_predictions.xls"))

    print(df_history_all)
    print("Done.")


if __name__ == "__main__":   # pragma: no cover
    # parse arguments
    args = arg_parser(sys.argv[1:])

    # call function
    xirt_runner(args.in_peptides, args.out_dir,
                args.xirt_params, args.learning_params,
                n_splits=3, test_size=0.1)
