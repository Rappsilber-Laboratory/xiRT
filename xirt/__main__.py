"""xiRT main module to run the training and prediction."""

import argparse
import logging
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from xirt import __version__ as xv
from xirt import features as xf
from xirt import predictor as xr
from xirt import xirtnet, qc
import matplotlib

matplotlib.use('Agg')
logger = logging.getLogger(__name__)


def arg_parser():  # pragma: not covered
    """
    Parse the arguments from the CLI.

    Returns:
        arguments, from parse_args
    """
    description = """
    xiRT is a machine learning tool for the (multidimensional) RT prediction of linear and
    crosslinked peptides. Use --help to see the command line arguments.

    Visit the documentation to get more information:
    https://xirt.readthedocs.io/en/latest/

    Current Version: {}
    """.format(xv)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--in_peptides",
                        help="Input peptide table to learn (and predict) the retention times.",
                        required=True, action="store", dest="in_peptides")

    parser.add_argument("-o", "--out_dir",
                        help="Directory to store the results",
                        required=True, action="store", dest="out_dir")

    parser.add_argument("-x", "--xirt_params",
                        help="YAML parameter file to control xiRT's deep learning architecture.",
                        required=True, action="store", dest="xirt_params")

    parser.add_argument("-l", "--learning_params",
                        help="YAML parameter file to control training and testing splits and data.",
                        required=True, action="store", dest="learning_params")

    parser.add_argument('--write', dest='write', action='store_true',
                        help="Flag for writing result prediction files. If false only summaries"
                             "are written (default: --write).")
    parser.add_argument('--no-write', dest='write', action='store_false',
                        help="Flag for writing result prediction files. If false only summaries"
                             "are written (default: --write).")
    parser.set_defaults(write=True)
    return parser


def xirt_runner(peptides_file, out_dir, xirt_loc, setup_loc, nrows=None, perform_qc=True,
                write=True, write_dummy=True):
    """
    Execute xiRT, train a model or generate predictions for RT across multiple RT domains.

    Args:
        peptides_file: str, location of the input psm/csm file
        out_dir: str, folder to store the results to
        xirt_loc: str, location of the yaml file for xirt
        setup_loc: str, location of the setup yaml
        single_pep_predictions:
        nrows: int, number of rows to sample (for quicker testing purposes only)
        perform_qc: bool, indicates if qc plots should be done.
        write: bool,  indicates result predictions should be stored
        write_dummy: bool if true dummy txt file is written after execution (for snakemake usag)

    Returns:
        None
    """
    start_time = time.time()
    xirt_params = yaml.load(open(xirt_loc), Loader=yaml.FullLoader)
    learning_params = yaml.load(open(setup_loc), Loader=yaml.FullLoader)
    matches_df = pd.read_csv(peptides_file, nrows=nrows)

    logger.info("xi params: {}".format(xirt_loc))
    logger.info("learning_params: {}".format(setup_loc))
    logger.info("peptides: {}".format(peptides_file))

    # convenience short cuts
    if learning_params["train"]["mode"] == "train":
        n_splits = 1
    elif learning_params["train"]["mode"] == "predict":
        n_splits = 0
    else:
        n_splits = learning_params["train"]["ncv"]

    test_size = learning_params["train"]["test_frac"]
    outpath = os.path.abspath(out_dir)
    xirt_params["callbacks"]["callback_path"] = os.path.join(outpath, "callbacks")

    # preprocess training data
    training_data = xr.preprocess(matches_df,
                                  sequence_type=learning_params["train"]["sequence_type"],
                                  max_length=learning_params["preprocessing"]["max_length"],
                                  cl_residue=learning_params["preprocessing"]["cl_residue"],
                                  fraction_cols=xirt_params["predictions"]["fractions"])

    # set training index by FDR and duplicates
    training_data.set_fdr_mask(fdr_cutoff=learning_params["train"]["fdr"],
                               str_filter=learning_params["preprocessing"]["filter"])
    training_data.set_unique_shuffled_sampled_training_idx(
        sample_frac=learning_params["train"]["sample_frac"],
        random_state=learning_params["train"]["sample_state"])

    # adjust RT if necessary to guarantee smooth learning
    # gradient length > 30 minutes (1500 seconds)
    for cont_col in xirt_params["predictions"]["continues"]:
        if training_data.psms[cont_col].max() > 1500:
            training_data.psms[cont_col] = training_data.psms[cont_col] / 60.0

    # init neural network structure
    xirtnetwork = xirtnet.xiRTNET(xirt_params, input_dim=training_data.features1.shape[1])

    # get the columns where the RT information is stored
    frac_cols = sorted([xirtnetwork.output_p[tt.lower() + "-column"] for tt in
                        xirt_params["predictions"]["fractions"]])

    cont_cols = sorted([xirtnetwork.output_p[tt.lower() + "-column"] for tt in
                        xirt_params["predictions"]["continues"]])

    # init data structures for results
    histories = []
    model_summary = []
    # manual accuracy for ordinal data
    accuracies_all = []
    if "ordinal" in ";".join([xirtnetwork.output_p[i + "-column"] for i in xirtnetwork.tasks]):
        has_ordinal = True
    else:
        has_ordinal = False

    cv_counter = 1
    # perform crossvalidation
    # train on n-1 fold, use test_size from n-1 folds for validation and test/predict RT
    # on the remaining fold
    logger.info("Starting crossvalidation (nfolds={})".format(n_splits))
    start_timecv = time.time()
    for train_idx, val_idx, pred_idx in training_data.iter_splits(n_splits=n_splits,
                                                                  test_size=test_size):
        logger.info("---------------------------------------------------------")
        logger.info("Starting crossvalidation iteration: {}".format(cv_counter))
        logger.info("# Train peptides: {}".format(len(train_idx)))
        logger.info("# Validation peptides: {}".format(len(val_idx)))
        logger.info("# Prediction peptides: {}".format(len(pred_idx)))
        logger.info("# Rescoring Candidates peptides: {}".format(len(training_data.predict_idx)))

        logger.info("Train indices: {}".format(train_idx[0:10]))
        logger.info("Validation indices: {}".format(val_idx[0:10]))
        logger.info("Prediction indices: {}".format(pred_idx[0:10]))
        logger.info("Rescoring Candidates indices: {}".format(training_data.predict_idx[0:10]))

        # init the network model
        # either load from existing or create from config
        if learning_params["train"]["pretrained_model"].lower() != "none":
            logger.info("Loading existing model from reference.")
            xirtnetwork.load_model(learning_params["train"]["pretrained_model"])
        else:
            logger.info("Building new model from config.")
            xirtnetwork.build_model(siamese=xirt_params["siamese"]["use"])

        # once model architecture is there, check if weights should be loaded
        # loading predefined weights
        if learning_params["train"]["pretrained_weights"].lower() != "none":
            logger.info("Loading pre-trained weights into model.")
            xirtnetwork.model.load_weights(learning_params["train"]["pretrained_weights"])

        # after loading the weights, we need to pop / re-design the layers for transfer learning
        if learning_params["train"]["pretrained_model"].lower() != "none":
            logger.info("Adjusting model architecture.")
            xirtnetwork.adjust_model()

        # finally compile
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
        logger.info("Fitting model.")
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

        # use the training for predicting unseen RTs
        training_data.predict_and_store(xirtnetwork, xp_cv, pred_idx, cv=cv_counter)

        if has_ordinal:
            train_preds = xirtnetwork.model.predict(xt_cv)
            val_preds = xirtnetwork.model.predict(xv_cv)
            pred_preds = xirtnetwork.model.predict(xp_cv)

            accuracies_all.extend(xr.compute_accuracy(train_preds,
                                                      training_data.psms.loc[train_idx],
                                                      xirtnetwork.tasks, xirtnetwork.output_p))
            accuracies_all.extend(xr.compute_accuracy(val_preds,
                                                      training_data.psms.loc[val_idx],
                                                      xirtnetwork.tasks, xirtnetwork.output_p))
            accuracies_all.extend(xr.compute_accuracy(pred_preds,
                                                      training_data.psms.loc[pred_idx],
                                                      xirtnetwork.tasks, xirtnetwork.output_p))

        # store metrics
        # store model? / callback
        df_history = pd.DataFrame(history.history)
        df_history["CV"] = cv_counter
        df_history["epoch"] = np.arange(1, len(df_history) + 1)
        cv_counter += 1
        histories.append(df_history)
    logger.info("Finished CV training model.")

    # CV training done, now deal with the data not used for training
    if learning_params["train"]["mode"] != "predict":
        # store model summary data
        model_summary_df = pd.DataFrame(model_summary, columns=metric_columns)
        model_summary_df["CV"] = np.repeat(np.arange(1, n_splits + 1), 3)
        model_summary_df["Split"] = np.tile(["Train", "Validation", "Prediction"], n_splits)

        # store manual accuray for ordinal data
        if has_ordinal:
            for count, task_i in enumerate(frac_cols):
                taski_short = task_i.split("_")[0]
                if "ordinal" in xirtnetwork.output_p[taski_short + "-column"]:
                    # accuracies_all contains accuracies for train, validation, pred
                    # problem is that accuracies are computed after the loop and stored in a 1d-ar
                    # retrieve information again
                    if len(frac_cols) == 1:
                        model_summary_df[taski_short + "_ordinal-accuracy"] = accuracies_all
                    else:
                        model_summary_df[taski_short + "_ordinal-accuracy"] = \
                            accuracies_all[count::len(frac_cols)]

        if learning_params["train"]["refit"]:
            logger.info("Refitting model on entire data to predict unseen data.")
            callbacks = xirtnetwork.get_callbacks(suffix="full")
            xrefit = training_data.get_features(training_data.train_idx)
            yrefit = training_data.get_classes(training_data.train_idx, frac_cols=frac_cols,
                                               cont_cols=cont_cols)
            xirtnetwork.build_model(siamese=xirt_params["siamese"]["use"])
            xirtnetwork.compile()

            if learning_params["train"]["pretrained_weights"].lower() != "none":
                logger.info("Loading pretrained weights into model for refit option.")
                xirtnetwork.model.load_weights(learning_params["train"]["pretrained_weights"])

            _ = xirtnetwork.model.fit(xrefit, yrefit, validation_split=test_size,
                                      epochs=xirt_params["learning"]["epochs"],
                                      batch_size=xirt_params["learning"]["batch_size"],
                                      verbose=xirt_params["learning"]["verbose"],
                                      callbacks=callbacks)
        else:
            logger.info("Selecting best performing CV model to predict unseen data.")

            # load the best performing model across cv from the validation split
            best_model_idx = np.argmin(
                model_summary_df[model_summary_df["Split"] == "Validation"]["loss"].values)
            logger.info("Best Model: {}".format(best_model_idx))
            logger.info("Loading weights.")
            xirtnetwork.model.load_weights(os.path.join(xirtnetwork.callback_p["callback_path"],
                                                        "xirt_weights_{}.h5".format(
                                                            str(best_model_idx + 1).zfill(2))))
            logger.info("Model Summary:")
            logger.info(model_summary_df.groupby("Split").agg([np.mean, np.std]).to_string())
    else:
        logger.info("Loading model weights.")
        xirtnetwork.build_model(siamese=xirt_params["siamese"]["use"])
        xirtnetwork.compile()
        xirtnetwork.model.load_weights(learning_params["train"]["pretrained_weights"])
        logger.info("Reassigning prediction index for prediction mode.")
        training_data.predict_idx = training_data.psms.index
        model_summary_df = pd.DataFrame()
        df_history_all = pd.DataFrame()

    # get the 'unvalidation data', e.g. data that was not used during training because
    # here the CSMs are that we want to save / rescore later!
    # assign prediction fold to entire data
    xu = training_data.get_features(training_data.predict_idx)
    yu = training_data.get_classes(training_data.predict_idx, frac_cols=frac_cols,
                                   cont_cols=cont_cols)
    training_data.predict_and_store(xirtnetwork, xu, training_data.predict_idx, cv=-1)
    eval_unvalidation = xirtnetwork.model.evaluate(xu, yu, batch_size=512)

    if has_ordinal:
        accs_tmp = xr.compute_accuracy(xirtnetwork.model.predict(xu),
                                       training_data.psms.loc[training_data.predict_idx],
                                       xirtnetwork.tasks, xirtnetwork.output_p)
        eval_unvalidation.extend(np.hstack([-1, "Unvalidation", accs_tmp]))
    else:
        eval_unvalidation.extend([-1, "Unvalidation"])

    # prediction modes dont have any training information
    if learning_params["train"]["mode"] != "predict":
        # collect epoch training data
        model_summary_df.loc[len(model_summary_df)] = eval_unvalidation
        df_history_all = pd.concat(histories)
        df_history_all = df_history_all.reset_index(drop=False).rename(columns={"index": "epoch"})
        df_history_all["epoch"] += 1

    # compute features for rescoring
    xf.compute_prediction_errors(training_data.psms, training_data.prediction_df,
                                 xirtnetwork.tasks, frac_cols,
                                 (xirtnetwork.siamese_p["single_predictions"]
                                  & xirtnetwork.siamese_p["use"]))

    # store results
    features_exhaustive = xf.add_interactions(training_data.prediction_df.filter(regex="error"),
                                              degree=len(xirtnetwork.tasks))

    # qc
    # only training procedure includes qc
    if perform_qc and (learning_params["train"]["mode"] != "predict"):
        logger.info("Generating qc plots.")
        qc.plot_epoch_cv(callback_path=xirtnetwork.callback_p["callback_path"],
                         tasks=xirtnetwork.tasks, xirt_params=xirt_params, outpath=outpath)

        qc.plot_summary_strip(model_summary_df, tasks=xirtnetwork.tasks, xirt_params=xirt_params,
                              outpath=outpath)

        qc.plot_cv_predictions(training_data.prediction_df, training_data.psms,
                               xirt_params=xirt_params, outpath=outpath)

        qc.plot_error_characteristics(training_data.prediction_df, training_data.psms,
                                      xirtnetwork.tasks, xirt_params, outpath, max_fdr=0.01)

    logger.info("Writing output tables.")
    # store setup in summary
    model_summary_df["xirt_params_loc"] = xirt_loc
    model_summary_df["xirt_params_base"] = os.path.basename(xirt_loc).split(".")[0]
    model_summary_df["learning_params"] = setup_loc
    model_summary_df["peptides"] = peptides_file

    df_history_all.to_csv(os.path.join(outpath, "epoch_history.csv"))
    model_summary_df.to_csv(os.path.join(outpath, "model_summary.csv"))

    if write:
        # store data
        training_data.psms.to_csv(os.path.join(outpath, "processed_psms.csv"))
        training_data_Xy = ((training_data.features1, training_data.features2),
                            training_data.get_classes(training_data.psms.index,
                                                      frac_cols=frac_cols, cont_cols=cont_cols))

        with open(os.path.join(xirt_params["callbacks"]["callback_path"], "Xy_data.p"), 'wb') as po:
            pickle.dump(training_data_Xy, po, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(xirt_params["callbacks"]["callback_path"], "encoder.p"), 'wb') as po:
            pickle.dump(training_data.le, po, protocol=pickle.HIGHEST_PROTOCOL)

        training_data.psms.to_csv(os.path.join(outpath, "processed_psms.csv"))
        training_data.prediction_df.to_csv(os.path.join(outpath, "error_features.csv"))
        features_exhaustive.to_csv(os.path.join(outpath, "error_features_interactions.csv"))

    # write a text file to indicate xirt is done.
    if write_dummy:
        with open(xirt_loc.replace(".yaml", ".txt"), "w") as of:
            of.write("done.")
    logger.info("Completed xiRT run.")
    logger.info("End Time: {}".format(datetime.now().strftime("%H:%M:%S")))
    end_time = time.time()
    logger.info("xiRT CV-training took: {:.2f} minutes".format((end_time - start_timecv) / 60.))
    logger.info("xiRT took: {:.2f} minutes".format((end_time - start_time) / 60.))


def main():  # pragma: no cover
    """Run xiRT main function."""
    parser = arg_parser()
    try:
        args = parser.parse_args(sys.argv[1:])
    except TypeError:
        parser.print_usage()

    # create dir if not there
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # create logger
    logger = logging.getLogger('xirt')
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.FileHandler(os.path.join(args.out_dir, "xirt_logger.log"), "w")
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(sh)

    logger.info("command line call:")
    logger.info(
        "xirt -i {} -o {} -x {} -l {}".format(args.in_peptides, args.out_dir, args.xirt_params,
                                              args.learning_params))
    logger.info("Init logging file.")
    logger.info("Starting Time: {}".format(datetime.now().strftime("%H:%M:%S")))
    logger.info("Starting xiRT.")
    logger.info("Using xiRT version: {}".format(xv))

    # call function
    xirt_runner(args.in_peptides, args.out_dir, args.xirt_params, args.learning_params,
                write=args.write)


if __name__ == "__main__":  # pragma: no cover
    main()
