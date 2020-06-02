import pickle
import sys
import time

import numpy as np
import pandas as pd
from tqdm.keras import TqdmCallback
import tensorflow as tf
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import Embedding, GRU, BatchNormalization, \
    Input, concatenate, Dropout, Dense, LSTM, Bidirectional, Flatten, Lambda, Add, Maximum, Multiply, Average, Concatenate

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.layers import CuDNNGRU, CuDNNLSTM
from sklearn.model_selection import ShuffleSplit
import yaml
from tensorflow.keras import backend as K
from tensorflow.keras import losses


class xiRTNET:
    def __init__(self):
        self.model = None

    def build_model(self):
        pass

    def compile(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def store(self):
        pass

    def print_layers(self):
        print([i.name for i in self.model.layers])

    def get_param_overview(self):
        trainable_count = np.sum([K.count_params(w) for w in self.model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in self.model.non_trainable_weights])

        print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))


def loss_ordered(y_true, y_pred):
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))
                     / (K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)


gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    # Currently, memory growth needs to be the same across GPUs
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def params_to_df(params, outpath):
    """Give a list of parameters as dictionary entries and transform it into a
    dataframe.

    :return:
    """
    params_df = [pd.DataFrame(list(params[i].items())).transpose() for i in np.arange(len(params))]
    params_df = pd.concat(params_df).reset_index()
    cols = params_df.iloc[0]
    params_df.columns = cols
    params_df = params_df[params_df["output_scx-weight"] != "output_scx-weight"]
    params_df = params_df.drop(0, axis=1)
    params_df = params_df.reset_index(drop=True)
    params_df["paramid"] = np.arange(len(params))
    params_df.to_csv(outpath)
    return (params_df)


def export_model_vis(fig_path, model):
    """Saves a given a model as png and pdf plots.

    :param fig_path: str, path where to store the model
    :param model: keras.model, network model to plot
    :return:
    """
    # plot pdf
    if fig_path != "":
        # plot png
        plot_model(model, to_file=fig_path + "Network_Model.png", show_shapes=True,
                   show_layer_names=True, dpi=300, expand_nested=True)
        try:
            plot_model(model, to_file=fig_path + "Network_Model.pdf", show_shapes=True,
                       show_layer_names=True, dpi=300, expand_nested=True)
        except ValueError as err:
            print("Encountered an VaulueError, PDF is still written. ({})".format(err))


def get_callbacks(outname, reduce_lr=True, early_stopping=True, check_point=True, log_csv=True,
                  tensor_board=False, progressbar=True, prefix_path="", patience=10):
    """Returns a list of callbacks.

    :param outname: str, path for the output
    :param early_stopping: bool, if True adds this callback to the return arguments
    :param check_point: bool, if True adds this callback to the return arguments
    :param log_csv: bool, if True adds this callback to the return arguments
    :param tensor_board: bool, if True adds this callback to the return arguments
    :param prefix_path: str, path to store the callback results
    :param patience: int, number of epochs without improvement to tolerate before stopping
    :return:
    """
    callbacks = []

    if progressbar:
        callbacks.append(TqdmCallback(verbose=0))

    if reduce_lr:
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1,
                                           min_delta=1e-4, mode='min')
        # factor=0.1
        callbacks.append(reduce_lr_loss)

    if early_stopping:
        # if the val_loss does not improve, stop
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience,
                           restore_best_weights=True)
        callbacks.append(es)

    if check_point:
        # save the best performing model
        mc = ModelCheckpoint(prefix_path + '/{}_model.h5'.format(outname),
                             monitor='val_loss', mode='min', verbose=0,
                             save_best_only=True)
        # "/model_{}{}_{}.h5".format(date_prefix, id_str_cv, name)
        mc2 = ModelCheckpoint(prefix_path + '/{}_weights.h5'.format(outname),
                              monitor='val_loss', mode='min', verbose=0,
                              save_best_only=True, save_weights_only=True)
        callbacks.append(mc)
        callbacks.append(mc2)

    if log_csv:
        # log some stuff (what?)
        csv_logger = CSVLogger(prefix_path + '/{}_epochlog.log'.format(outname))
        callbacks.append(csv_logger)

    if tensor_board:
        # use tensorboard logger
        tb = TensorBoard(log_dir='tensorboard/', embeddings_freq=0)
        callbacks.append(tb)

    return callbacks


def add_recursive_layer(config, prev_layer, i=0, name=""):
    """Adds a GRU / LSTM layer.

    :param config: dict, settings for the sequence layers
    :param prev_layer: keras.layer, the previous layer where this layer is added to
    :param i: int, current number of layer to add
    :param name: str, name of the layer
    :return:
    """

    if i == 0:
        return_seqs = True
    else:
        return_seqs = False

    # take config for LSTM / GRU
    tmp_conf = config["LSTM"]
    units = tmp_conf["units"]
    act = tmp_conf["activation"]

    # add regularizer
    if tmp_conf["kernel_regularization"] == "l1":
        reg_kernel = regularizers.l1(tmp_conf["kernelregularizer_value"])

    elif tmp_conf["kernel_regularization"] == "l2":
        reg_kernel = regularizers.l2(tmp_conf["kernelregularizer_value"])

    elif tmp_conf["kernel_regularization"] == "l1l2":
        reg_kernel = regularizers.l1_l2(tmp_conf["kernelregularizer_value"],
                                        tmp_conf["kernelregularizer_value"])

    else:
        sys.exit("Regularizer not defined ({})".format(tmp_conf["kernel_regularizer"]))

    # add regularizer
    if tmp_conf["activity_regularization"] == "l1":
        reg_act = regularizers.l1(tmp_conf["activityregularizer_value"])

    elif tmp_conf["activity_regularization"] == "l2":
        reg_act = regularizers.l2(tmp_conf["activityregularizer_value"])

    elif tmp_conf["kernel_regularization"] == "l1l2":
        reg_act = regularizers.l1_l2(tmp_conf["activityregularizer_value"],
                                     tmp_conf["activityregularizer_value"])
    else:
        sys.exit("Regularizer not defined ({})".format(tmp_conf["kernel_regularizer"]))

    # add a bidirectional layer?
    if tmp_conf["bidirectional"] is True:
        if tmp_conf["type"] == "GRU":
            lstm = Bidirectional(GRU(units, activation=act, activity_regularizer=reg_act,
                                     kernel_regularizer=reg_kernel,
                                     return_sequences=return_seqs),
                                 name=name + "BiGRU")(prev_layer)

        elif tmp_conf["type"] == "LSTM":
            lstm = Bidirectional(LSTM(units, activation=act, activity_regularizer=reg_act,
                                      kernel_regularizer=reg_kernel,
                                      return_sequences=return_seqs,
                                      name=name + "BiLSTM"))(prev_layer)

        elif tmp_conf["type"] == "CuDNNGRU":
            # only tanh activiation cudnngru
            # TODO! GRU
            # CuDNNGRU, CuDNNLSTM
            lstm = Bidirectional(CuDNNGRU(units, activity_regularizer=reg_act,
                                          kernel_regularizer=reg_kernel,
                                          return_sequences=return_seqs,
                                          name=name + "BiCuGRU"))(prev_layer)

        elif tmp_conf["type"] == "CuDNNLSTM":
            # only tanh activiation cudnngru
            # TODO! CuDNNLSTM
            lstm = Bidirectional(CuDNNLSTM(units, activity_regularizer=reg_act,
                                           kernel_regularizer=reg_kernel,
                                           return_sequences=return_seqs, ),
                                 name=name + "BiCuLSTM")(prev_layer)

        else:
            sys.exit("Option for LSTM not defined. ({})".format(
                tmp_conf["type"]))
    # one-way
    else:
        if tmp_conf["type"] == "GRU":
            lstm = GRU(units, activation=act, activity_regularizer=reg_act,
                       kernel_regularizer=reg_kernel, return_sequences=return_seqs,
                       name=name + "GRU")(prev_layer)

        elif tmp_conf["type"] == "LSTM":
            lstm = LSTM(units, activation=act, activity_regularizer=reg_act,
                        kernel_regularizer=reg_kernel, return_sequences=return_seqs,
                        name=name + "LSTM")(prev_layer)

        elif tmp_conf["type"] == "CuDNNGRU":
            # only tanh activiation cudnngru
            lstm = CuDNNGRU(units, activity_regularizer=reg_act, kernel_regularizer=reg_kernel,
                            return_sequences=return_seqs,
                            name=name + "CuGRU")(prev_layer)

        elif tmp_conf["type"] == "CuDNNLSTM":
            # only tanh activiation cudnngru
            lstm = CuDNNLSTM(units, activity_regularizer=reg_act, kernel_regularizer=reg_kernel,
                             return_sequences=return_seqs,
                             name=name + "CuLSTM")(prev_layer)

        else:
            sys.exit("Option for LSTM not defined. ({})".format(
                tmp_conf["type"]))

    # add batch normalization ?
    if tmp_conf["lstm_bn"] is True:
        lstm = BatchNormalization(name=name + "lstm_bn_" + str(i))(lstm)
    return lstm


def add_dense_layer(config, idx, prev_layer):
    """Adds a dense layer.

    Parameters:
    ----------
    idx: int,
            integer indicating the idx'th layer that was added
    prev_layer: keras layer,
                Functional API object from the definition of the network.
    """
    tmp_conf = config["dense"]
    if tmp_conf["kernel_regularizer"][idx] == "l1":
        reg_ = regularizers.l1(tmp_conf["regularizer_value"][idx])

    elif tmp_conf["kernel_regularizer"][idx] == "l2":
        reg_ = regularizers.l2(tmp_conf["regularizer_value"][idx])

    elif tmp_conf["kernel_regularizer"][idx] == "l1l2":
        reg_ = regularizers.l1_l2(tmp_conf["regularizer_value"][idx],
                                  tmp_conf["regularizer_value"][idx])

    else:
        sys.exit("Regularizer not defined ({})".format(tmp_conf["kernel_regularizer"]))

    # dense layer
    dense = Dense(tmp_conf["neurons"][idx], kernel_regularizer=reg_)(prev_layer)

    if tmp_conf["dense_bn"][idx] is True:
        dense = BatchNormalization()(dense)

    # this can be used for uncertainty estimation
    # dense = Dropout(tmp_conf["dropout"][idx])(dense, training=True)
    dense = Dropout(tmp_conf["dropout"][idx])(dense)
    return dense


def add_task_dense_layers(config, net, net_meta=None):
    """Adds task specific dense layers. If net_meta is set also adds the meta
    information as input for each individual layer.

    :param config: dict,
    :param net:
    :param net_meta:
    :return:
    """
    for i in np.arange(config["dense"]["nlayers"]):
        # the first layer requires special handling, it takes the input from the shared
        # sequence layers
        if i == 0:
            if net_meta is not None:
                task = concatenate([net.output, net_meta.output])
                task = add_dense_layer(config, i, task)
            else:
                task = add_dense_layer(config, i, net)
        else:
            task = add_dense_layer(config, i, task)
    return task


def build_xirtnet(config, input_dim, input_meta=None, single_task="None"):
    """Builds the network model without compiling or fitting it.

    Generates the xiRTNET
    :return:
    config = param
    input_dim = 50
    fig_path = ""
    single_task = "None"
    input_meta=None
    """
    # input for the network
    inlayer, net = build_base_network(input_dim, config)
    model = build_task_network(config, inlayer, input_meta, net, single_task)
    return(model)


def build_task_network(config, inlayer, input_meta, net, single_task):
    """

    Parameters
    ----------
    config
    inlayer
    input_meta
    net
    single_task

    Returns
    -------

    """
    # if desired the sequence input can be supplemented by precomputed features
    if input_meta is not None:
        in_meta = Input(shape=(input_meta,))
        net_meta = Model(inputs=in_meta, outputs=in_meta)
        net = Model(inputs=inlayer, outputs=net)
    else:
        net_meta = None

    # create the individual prediction net works
    act_conf = config["output"]
    hsax_task = add_task_dense_layers(config, net, net_meta)
    hsax_task = Dense(act_conf["hsax-dimension"], activation=act_conf["hsax-activation"],
                      name="hSAX")(hsax_task)

    if single_task.lower() == "hsax":
        model = Model(inputs=inlayer, outputs=[hsax_task])
        return model

    scx_task = add_task_dense_layers(config, net, net_meta)
    scx_task = Dense(act_conf["scx-dimension"], activation=act_conf["scx-activation"],
                     name="SCX")(scx_task)
    if single_task.lower() == "scx":
        model = Model(inputs=inlayer, outputs=[scx_task])
        return model

    rp_task = add_task_dense_layers(config, net, net_meta)
    rp_task = Dense(act_conf["rp-dimension"], activation=act_conf["rp-activation"],
                    name="RP")(rp_task)

    if single_task.lower() == "rp":
        model = Model(inputs=inlayer, outputs=[rp_task])
        return model

    if input_meta is None:
        model = Model(inputs=inlayer, outputs=[hsax_task, scx_task, rp_task])
    else:
        model = Model(inputs=[net.input, net_meta.input], outputs=[hsax_task, scx_task, rp_task])

    return model


def build_base_network(input_dim, config):
    """Construct a simple network that consists of an input layer, an embedding
    layer, and 1 or more recurrent-type layers (LSTM/GRU)

    :param input_dim:
    :param config:
    :return:
    """
    inlayer = Input(shape=input_dim, name="main_input")

    # translate labels into continuous space
    net = Embedding(input_dim=input_dim, output_dim=config["embedding"]["length"],
                    embeddings_initializer="he_normal", name="main_embedding")(inlayer)

    # sequence layers (LSTM-type) + batch normalization if in config
    for i in np.arange(config["LSTM"]["nlayers"]):
        if config["LSTM"]["nlayers"] > 1:
            net = add_recursive_layer(config, net, i, name="shared{}_".format(i))
        else:
            # return only sequence when there are more than 1 recurrent layers
            net = add_recursive_layer(config, net, 1, name="shared{}_".format(i))
    # return Model(inlayer, net)
    return inlayer, net


def build_siamese(config, input_dim, input_meta=None, single_task="None"):
    """

    :return:
    config = param
    input_dim = 50
    """
    # input for the network
    # %%
    inlayer, net = build_base_network(input_dim, config)
    base_network = Model(inlayer, net, name="siamese")

    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # merge the lower and upper part of the network
    if config["siamese"]["merge_type"].lower() == "add":
        merger_layer = Add()([processed_a, processed_b])

    elif config["siamese"]["merge_type"].lower() == "multiply":
        merger_layer = Multiply()([processed_a, processed_b])

    elif config["siamese"]["merge_type"].lower() == "average":
        merger_layer = Average()([processed_a, processed_b])

    elif config["siamese"]["merge_type"].lower() == "concatenate":
        merger_layer = Concatenate()([processed_a, processed_b])

    elif config["siamese"]["merge_type"].lower() == "maximum":
        merger_layer = Maximum()([processed_a, processed_b])

    else:
        sys.exit("Merge Type not implemented. Must be one of: "
                 + "add, multiply, average, concatenate, maximum")

    net = Model([input_a, input_b], merger_layer)

    # create the individual prediction net works
    act_conf = config["output"]
    hsax_task = add_task_dense_layers(config, merger_layer, None)
    hsax_task = Dense(act_conf["hsax-dimension"], activation=act_conf["hsax-activation"],
                      name="hSAX")(hsax_task)

    scx_task = add_task_dense_layers(config, merger_layer, None)
    scx_task = Dense(act_conf["scx-dimension"], activation=act_conf["scx-activation"],
                     name="SCX")(scx_task)

    rp_task = add_task_dense_layers(config, merger_layer, None)
    rp_task = Dense(act_conf["rp-dimension"], activation=act_conf["rp-activation"],
                    name="RP")(rp_task)

    model_full = Model(inputs=net.input, outputs=[hsax_task, scx_task, rp_task])
    return(model_full)


def dev_loading_models():
    # following code is dev code and shows how to overwrite weights!
    # simple case
    # merger_layer = Dense(1, activation="relu")(merger_layer)
    model = Model([input_a, input_b], merger_layer)
    siamese_model = model.get_layer("siamese")
    export_model_vis("siamese_part", siamese_model)
    export_model_vis("siamese_full", model_full)
    export_model_vis("reference", reference)

    get_param_overview(reference)
    get_param_overview(model_full)

    print_layers(reference)
    print_layers(model_full)
    print_layers(model_full.get_layer("siamese"))

    # model_full.get_layer("siamese").get_layer("main_embedding").get_weights()
    # reference.get_layer("main_embedding").get_weights()
    # %%
    # model_full.get_layer("siamese").get_layer("main_embedding").set_weights(reference.get_layer("main_embedding").get_weights())
    x1 = model_full.get_layer("siamese").get_layer("main_embedding").get_weights()
    # weights = reference.get_weights()
    # model_full.set_weights(weights)
    model_full.get_layer("siamese").load_weights(
        'paper/data/results_2020_new/20200226_5pPSMFDR_linear_crossvalidation/callbacks/xiRTNET_CV1_weights.h5', by_name=True)
    old_model = load_model(
        'paper/data/results_2020_new/20200226_5pPSMFDR_linear_crossvalidation/callbacks/xiRTNET_CV1_model.h5')
    old_embed = old_model.get_layer("main_embedding").get_weights()
    x2 = model_full.get_layer("siamese").get_layer("main_embedding").get_weights()
    model_full.get_layer("siamese").load_weights(
        'paper/data/results_2020_new/20200226_5pPSMFDR_linear_crossvalidation/callbacks/xiRTNET_CV1_weights.h5',
        by_name=True)
    x1[0][0] == x2[0][0]
    # %%


def compile_model(model, config, loss=None, metric=None, loss_weights=None):
    comp_params = config["learning"]
    opt_params = config["output"]

    # compile optimizer
    adam = optimizers.Adam(lr=comp_params["learningrate"])

    if not loss:
        loss = opt_params["loss"]

    if not metric:
        metric = [opt_params["metric"]]

    model.compile(loss=loss, optimizer=adam, metrics=metric, loss_weights=loss_weights)


def fit_model(model, config, X_train, y_train, X_val, y_val, loss=None, metric=None,
              loss_weights=None, v=0, outname="", X_train_meta=None, X_val_meta=None):
    """Compile and / or fit the model.

    :param model:
    :param config:
    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :param loss:
    :param metric:
    :param loss_weights:
    :param v:
    :param outname:
    :param X_train_meta:
    :param X_val_meta:
    :return:

    model = xiRTNET
    config = param
    X_train = np.asarray(Xcv_train)
    y_train = ycv_train
    X_val = np.asarray(Xcv_val)
    y_val = ycv_val
    loss = multi_task_loss
    metric = metrics
    loss_weights = loss_weights
    """
    comp_params = config["learning"]
    opt_params = config["output"]

    # compile optimizer
    adam = optimizers.Adam(lr=comp_params["learningrate"])

    if not loss:
        loss = opt_params["loss"]

    if not metric:
        metric = [opt_params["metric"]]

    model.compile(loss=loss, optimizer=adam, metrics=metric, loss_weights=loss_weights)

    # configure validation data format, depending on supplied validation split
    callbacks = get_callbacks(outname, check_point=True, log_csv=True, early_stopping=True,
                              patience=comp_params["patience"],
                              prefix_path=opt_params["callback-path"])
    if len(X_val) == 0:
        # if no validation data is given, make sure to use at least 10% for early stopping
        validation_data = None
        validation_split = 0.1

    else:
        validation_split = 0.0
        if X_train_meta is None:
            validation_data = (X_val, y_val)
        else:
            validation_data = ([X_val, X_val_meta], y_val)

    # configure train data format, depending on supplied meta data
    if X_train_meta is None:
        train_data = X_train
    else:
        train_data = [X_train, X_train_meta]

    # fit model
    print("Fitting Model: ")
    history = model.fit(train_data, y_train, batch_size=comp_params["batch_size"],
                        epochs=comp_params["epochs"], verbose=v, callbacks=callbacks,
                        validation_data=validation_data, validation_split=validation_split)

    print("Callbacks are stored here: {}".format(opt_params["callback-path"]))
    return history, model


def init_prediction_df(xifdr_df, paramid=-1, linear_cols=True):
    """Creates a dataframe that is used for storing the predictions from the
    network.

    :param xifdr_df:
    :param paramid:
    :param linear_cols:
    :return:
    """
    # init output dataframe
    nrows = xifdr_df.shape[0]
    prediction_df = pd.DataFrame()
    prediction_df["hSAX_prediction"] = np.zeros(nrows)
    prediction_df["SCX_prediction"] = np.zeros(nrows)
    prediction_df["RP_prediction"] = np.zeros(nrows)

    prediction_df["hSAX_probability"] = np.zeros(nrows)
    prediction_df["SCX_probability"] = np.zeros(nrows)
    prediction_df["hSAX_prob_ar"] = np.zeros(nrows)
    prediction_df["SCX_prob_ar"] = np.zeros(nrows)
    prediction_df.index = xifdr_df.index

    prediction_df["Actual_RP"] = -1
    prediction_df["Actual_SCX0based"] = -1
    prediction_df["Actual_hSAX0based"] = -1
    prediction_df["CV_FOLD"] = -1
    prediction_df["paramid"] = -1

    if linear_cols:
        prediction_df["hSAX_prediction_pepseq1"] = np.zeros(nrows)
        prediction_df["SCX_prediction_pepseq1"] = np.zeros(nrows)
        prediction_df["RP_prediction_pepseq1"] = np.zeros(nrows)
        prediction_df["hSAX_probability_pepseq1"] = np.zeros(nrows)
        prediction_df["SCX_probability_pepseq1"] = np.zeros(nrows)

        prediction_df["hSAX_prediction_pepseq2"] = np.zeros(nrows)
        prediction_df["SCX_prediction_pepseq2"] = np.zeros(nrows)
        prediction_df["RP_prediction_pepseq2"] = np.zeros(nrows)
        prediction_df["hSAX_probability_pepseq2"] = np.zeros(nrows)
        prediction_df["SCX_probability_pepseq2"] = np.zeros(nrows)
    return (prediction_df)


def store_predictions(prediction_df, X_pred_cv, length, hsax, scx, rp, le, model, i,
                      single_seq_pred=True, xcv_pred_meta=pd.DataFrame()):
    """Writes the predictions to the result dataframe and optionally computes
    the predictions for the individual linear peptide sequences.

    Parameters:
    ----------
    prediction_df: df, dataframe where the predictions should be stored
    X_pred_cv: df, dataframe with an iteration of CV folds that is used for prediction.
    length
    :return:
    """
    prediction_df["hSAX_prediction"].loc[X_pred_cv.index] = np.argmax(hsax, axis=1)
    prediction_df["hSAX_probability"].loc[X_pred_cv.index] = np.max(hsax, axis=1)

    prediction_df["SCX_prediction"].loc[X_pred_cv.index] = np.argmax(scx, axis=1)
    prediction_df["SCX_probability"].loc[X_pred_cv.index] = np.max(scx, axis=1)

    prediction_df["RP_prediction"].loc[X_pred_cv.index] = np.ravel(rp)
    prediction_df["CV_FOLD"].loc[X_pred_cv.index] = i

    if single_seq_pred:
        pepseq1 = sequence.pad_sequences(df.retrieve_pepseq(X_pred_cv, le, peptide_id=0), length)
        pepseq2 = sequence.pad_sequences(df.retrieve_pepseq(X_pred_cv, le, peptide_id=1), length)
        names = ["pepseq1", "pepseq2"]
        for seq, name in zip([pepseq1, pepseq2], names):
            if xcv_pred_meta.empty:
                hsax_indv, scx_indv, rp_indv = model.predict(seq)
            else:
                hsax_indv, scx_indv, rp_indv = model.predict((seq, xcv_pred_meta))

            prediction_df["hSAX_prediction_{}".format(name)].loc[X_pred_cv.index] = \
                np.argmax(hsax_indv, axis=1)
            prediction_df["hSAX_probability_{}".format(name)].loc[X_pred_cv.index] = \
                np.max(hsax_indv, axis=1)
            prediction_df["SCX_prediction_{}".format(name)].loc[X_pred_cv.index] = \
                np.argmax(scx_indv, axis=1)
            prediction_df["SCX_probability_{}".format(name)].loc[X_pred_cv.index] = \
                np.max(scx_indv, axis=1)
            prediction_df["RP_prediction_{}".format(name)].loc[X_pred_cv.index] = \
                np.ravel(rp_indv)


def format_metrics(metrics, train_metrics):
    train_str = "; ".join(["{}: {:.2f}".format(i, j) for i, j in zip(metrics[:-1],
                                                                     train_metrics)])
    return train_str


def pseudo_csv_logger(cv_fold, date_prefix, df_metrics, outpath, paramid):
    """Writes the results of the CV iteration to a file.

    :param cv_fold: int, fold iteration
    :param date_prefix: str, prefix to use for storing
    :param df_metrics: df, dataframe with metric results
    :param outpath: str, where to write the results to
    :param paramid: int, integer indicator parameter combination
    :return:
    """
    csv_pseudo_logger = open(outpath + "{}_depart-simple_results.log".format(date_prefix),
                             "a")
    if (paramid == 0) and (cv_fold == 0):
        df_metrics.to_csv(csv_pseudo_logger, header=True)
    else:
        df_metrics.to_csv(csv_pseudo_logger, header=False)
    csv_pseudo_logger.close()


def load_model_time(logging, model_path):
    """Loads the best performing neural network model over time. Model was
    saved over callbacks.

    :param logging: logging obj, used for system logging
    :param model_path: str, the path to the file that should be loaded
    :return:
    """
    acv = time.time()
    model = load_model(model_path)
    bcv = time.time()
    logging.info("Loaded model and weights together. (took {:.2f} minutes.)".format(
        (bcv - acv) / 60.))
    return model


def create_params(param):
    """Creates dictionaries with parameters based on a param dictionary. The
    result can be passed to a neural network model for compilation.

    :param param: dict, parameters for metrics, losses and task weights
    :return:
    """
    metrics = {"RP": param["output"]["rp-metrics"],
               "SCX": param["output"]["scx-metrics"],
               "hSAX": param["output"]["hsax-metrics"]}

    loss_weights = {"RP": param["output"]["rp-weight"],
                    "hSAX": param["output"]["class-weight"],
                    "SCX": param["output"]["class-weight"]}

    multi_task_loss = {"RP": param["output"]["rp-loss"],
                       "SCX": param["output"]["scx-loss"],
                       "hSAX": param["output"]["scx-loss"]}
    return loss_weights, metrics, multi_task_loss


def prepare_multitask_y(y_train, params):
    """Reshapes the target variables into a format useable for the multi-task
    model.

    Parameter:
    ---------
    y_train: ar-like, must contain defined columns
    """
    if params["hsax-activation"] == "linear":
        return ([(y_train[params["hsax-column"]].values),
                 (y_train[params["scx-column"]].values),
                 (y_train[params["rp-column"]].values)])
    else:
        return ([reshapey(y_train[params["hsax-column"]].values),
                 reshapey(y_train[params["scx-column"]].values),
                 y_train[params["rp-column"]].values])


def reshapey(values):
    """Flattens the arrays that were stored in a single data frame cell, such
    that the shape is again usable for the neural network input.

    :param values:
    :return:
    """
    nrows = len(values)
    ncols = values[0].size
    return np.array([y for x in values for y in x]).reshape(nrows, ncols)
