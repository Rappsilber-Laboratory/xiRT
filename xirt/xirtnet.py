"""Module to build the xiRT-network"""
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard, \
    ReduceLROnPlateau
from tensorflow.keras.layers import Embedding, GRU, BatchNormalization, \
    Input, concatenate, Dropout, Dense, LSTM, Bidirectional, Add, Maximum, Multiply, Average, \
    Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.layers import CuDNNGRU, CuDNNLSTM
from tqdm.keras import TqdmCallback


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


class xiRTNET:
    """
    A class used to build, train and modify xiRT for RT prediction.

    This class is can be used to build customized networks based on the input parameterization.

    Attributes:
    ----------
    TODO
     add self parameters here

    Methods:
    --------
    TODO
     functions go here
    """

    def __init__(self, params, input_dim):
        self.model = None
        self.input_dim = input_dim

        self.LSTM_p = params["LSTM"]
        self.conv_p = params["conv"]
        self.dense_p = params["dense"]
        self.embedding_p = params["embedding"]
        self.learning_p = params["learning"]
        self.output_p = params["output"]
        self.siamese_p = params["siamese"]

    def build_model(self, single_task="None", siamese=False):
        inlayer, net = self._build_base_network()

        if siamese:
            # for crosslinks
            base_network = Model(inlayer, net, name="siamese")

            input_a = Input(shape=self.input_dim)
            input_b = Input(shape=self.input_dim)

            # init the base network with shared parameters
            processed_a = base_network(input_a)
            processed_b = base_network(input_b)

            # merge the lower and upper part of the network
            if self.siamese_p["merge_type"].lower() == "add":
                merge_func = Add

            elif self.siamese_p["merge_type"].lower() == "multiply":
                merge_func = Multiply

            elif self.siamese_p["merge_type"].lower() == "average":
                merge_func = Average

            elif self.siamese_p["merge_type"].lower() == "concatenate":
                merge_func = Concatenate

            elif self.siamese_p["merge_type"].lower() == "maximum":
                merge_func = Maximum
            else:
                raise KeyError("Merging operation not supported ({})".
                               format(self.siamese_p["merge_type"]))
            merger_layer = merge_func()([processed_a, processed_b])

            net = Model([input_a, input_b], merger_layer)
            # create the individual prediction net works
            act_conf = self.output_p
            hsax_task = self._add_task_dense_layers(merger_layer, None)
            hsax_task = Dense(act_conf["hsax-dimension"], activation=act_conf["hsax-activation"],
                              name="hSAX")(hsax_task)

            scx_task = self._add_task_dense_layers(merger_layer, None)
            scx_task = Dense(act_conf["scx-dimension"], activation=act_conf["scx-activation"],
                             name="SCX")(scx_task)

            rp_task = self._add_task_dense_layers(merger_layer, None)
            rp_task = Dense(act_conf["rp-dimension"], activation=act_conf["rp-activation"],
                            name="RP")(rp_task)

            model_full = Model(inputs=net.input, outputs=[hsax_task, scx_task, rp_task])
        else:
            model_full = self._build_task_network(inlayer, input_meta=None, net=net,
                                                  single_task=single_task)
        self.model = model_full

    def _build_base_network(self):
        """
        Construct a simple network that consists of an input, embedding, and recurrent-layers.

        Function can be used to  build a scaffold for siamese networks.

        Returns:

        """
        # init the input layer
        inlayer = Input(shape=self.input_dim, name="main_input")

        # translate labels into continuous space
        net = Embedding(input_dim=self.input_dim,
                        output_dim=self.embedding_p["length"],
                        embeddings_initializer="he_normal", name="main_embedding")(inlayer)

        # sequence layers (LSTM-type) + batch normalization if in config
        for i in np.arange(self.LSTM_p["nlayers"]):
            if self.LSTM_p["nlayers"] > 1:
                net = self._add_recursive_layer(net, i, name="shared{}_".format(i))
            else:
                # return only sequence when there are more than 1 recurrent layers
                net = self._add_recursive_layer(net, 1, name="shared{}_".format(i))
        return inlayer, net

    def _build_task_network(self, inlayer, input_meta, net, single_task):
        """
        Build task specific, dense layers in the xiRT architecture.

        Parameters:
        ----------
        inlayer: layer, previous input layers
        input_meta:
        net:
        single_task:

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
        act_conf = self.output_p
        hsax_task = self._add_task_dense_layers(net, net_meta)
        hsax_task = Dense(act_conf["hsax-dimension"], activation=act_conf["hsax-activation"],
                          name="hSAX")(hsax_task)

        if single_task.lower() == "hsax":
            model = Model(inputs=inlayer, outputs=[hsax_task])
            return model

        scx_task = self._add_task_dense_layers(net, net_meta)
        scx_task = Dense(act_conf["scx-dimension"], activation=act_conf["scx-activation"],
                         name="SCX")(scx_task)
        if single_task.lower() == "scx":
            model = Model(inputs=inlayer, outputs=[scx_task])
            return model

        rp_task = self._add_task_dense_layers(net, net_meta)
        rp_task = Dense(act_conf["rp-dimension"], activation=act_conf["rp-activation"],
                        name="RP")(rp_task)

        if single_task.lower() == "rp":
            model = Model(inputs=inlayer, outputs=[rp_task])
            return model

        if input_meta is None:
            model = Model(inputs=inlayer, outputs=[hsax_task, scx_task, rp_task])
        else:
            model = Model(inputs=[net.input, net_meta.input],
                          outputs=[hsax_task, scx_task, rp_task])

        return model

    def _add_recursive_layer(self, prev_layer, n_layer=0, name=""):
        """
        Add recursive layers to network.

        Depending on the parameters adds GRU/LSTM/Cu*** layers to the network architecture.
        Regularization parameters are taken from the initiliazed options.

        Parameters:
        ----------
        prev_layer: keras model, a base model that should be extended
        n_layer: int, current number of layer to add
        name: str, name of the layer

        Return:
        -------

        """
        # adjust parameter for 1  or more recurrent layers
        return_seqs = True if n_layer == 0 else False

        # add regularizer
        reg_kernel = self._init_regularizer(regularizer=self.LSTM_p["kernel_regularization"],
                                            reg_value=self.LSTM_p["kernelregularizer_value"])
        reg_act = self._init_regularizer(regularizer=self.LSTM_p["activity_regularization"],
                                         reg_value=self.LSTM_p["activityregularizer_value"])

        # set the RNN Function to be used
        if self.LSTM_p["type"] == "GRU":
            f_rnn = GRU
            f_name = "Gru"
        elif self.LSTM_p["type"] == "LSTM":
            f_rnn = LSTM
            f_name = "LSTM"
        elif self.LSTM_p["type"] == "CuDNNGRU":
            f_rnn = CuDNNGRU
            f_name = "CuGRU"
        elif self.LSTM_p["type"] == "CuDNNLSTM":
            f_rnn = CuDNNLSTM
            f_name = "CuLSTM"
        else:
            raise KeyError("Recurrent type option not found ({})".format(
                self.LSTM_p["activity_regularization"]))

        if self.LSTM_p["bidirectional"]:
            lstm = Bidirectional(f_rnn(self.LSTM_p["units"], activation=self.LSTM_p["activation"],
                                       activity_regularizer=reg_act,
                                       kernel_regularizer=reg_kernel, return_sequences=return_seqs),
                                 name=name + "Bi" + f_name)(prev_layer)
        else:
            lstm = f_rnn(self.LSTM_p["units"], activation=self.LSTM_p["activation"],
                         activity_regularizer=reg_act,
                         kernel_regularizer=reg_kernel, return_sequences=return_seqs,
                         name=name + "Bi" + f_name)(prev_layer)

        # add batch normalization
        if self.LSTM_p["lstm_bn"]:
            lstm = BatchNormalization(name=name + "lstm_bn_" + str(n_layer))(lstm)
        return lstm

    def _add_task_dense_layers(self, net, net_meta=None):
        """
        Adds task specific dense layers.

        If net_meta is set also adds the meta information as input for each individual layer.

        Parameters:
        -----------
        TODO
        """
        task = None
        for i in np.arange(self.dense_p["nlayers"]):
            # the first layer requires special handling, it takes the input from the shared
            # sequence layers
            if i == 0:
                if net_meta is not None:
                    task = concatenate([net.output, net_meta.output])
                    task = self._add_dense_layer(i, task)
                else:
                    task = self._add_dense_layer(i, net)
            else:
                task = self._add_dense_layer(i, task)
        return task

    def _add_dense_layer(self, idx, prev_layer):
        """
        Adds a dense layer.

        Parameters:
        ----------
        idx: int,
                integer indicating the idx'th layer that was added
        prev_layer: keras layer,
                    Functional API object from the definition of the network.

        """
        # add regularizer
        reg_ = self._init_regularizer(self.dense_p["kernel_regularizer"][idx],
                                      self.dense_p["regularizer_value"][idx])
        # dense layer
        dense = Dense(self.dense_p["neurons"][idx], kernel_regularizer=reg_)(prev_layer)

        if self.dense_p["dense_bn"][idx]:
            dense = BatchNormalization()(dense)
        # this can be used for uncertainty estimation
        # dense = Dropout(tmp_conf["dropout"][idx])(dense, training=True)
        dense = Dropout(self.dense_p["dropout"][idx])(dense)
        return dense

    @staticmethod
    def _init_regularizer(regularizer, reg_value):
        """
        Create a regularizer (l1, l2, l1l2) to be used in an layer of choice.

        Args:
            regularizer: regularizers, type of regularizer to be used
            reg_value: float, lambda value

        Returns:
            function, regularizer object to be used in model building.
        """
        if regularizer == "l1":
            regularizer_tmp = regularizers.l1(reg_value)

        elif regularizer == "l2":
            regularizer_tmp = regularizers.l2(reg_value)

        elif regularizer == "l1l2":
            regularizer_tmp = regularizers.l1_l2(reg_value, reg_value)

        else:
            raise KeyError("Regularizer not defined ({})".format(regularizer))
        return regularizer_tmp

    def export_model_visualization(self, fig_path):
        try:
            plot_model(self.model, to_file=fig_path + "Network_Model.pdf", show_shapes=True,
                       show_layer_names=True, dpi=300, expand_nested=True)
        except ValueError as err:
            print("Encountered an ValueError, PDF is still written. ({})".format(err))

    def compile(self, loss=None, metric=None, loss_weights=None):
        # compile optimizer
        adam = optimizers.Adam(lr=self.learning_p["learningrate"])

        # overwrite parameters from config, if specified
        if not loss:
            loss = self.output_p["loss"]
        if not metric:
            metric = [self.output_p["metric"]]

        self.model.compile(loss=loss, optimizer=adam, metrics=metric, loss_weights=loss_weights)

    def fit(X_train, y_train, X_val, y_val, X_train_meta=None, X_val_meta=None):
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

        history = model.fit(train_data, y_train, batch_size=comp_params["batch_size"],
                            epochs=comp_params["epochs"], verbose=v, callbacks=callbacks,
                            validation_data=validation_data, validation_split=validation_split)

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
