"""Module to build the xiRT-network."""
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard, \
    ReduceLROnPlateau
from tensorflow.keras.layers import Embedding, GRU, BatchNormalization, \
    Input, concatenate, Dropout, Dense, LSTM, Bidirectional, Add, Maximum, Multiply, Average, \
    Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.layers import CuDNNGRU, CuDNNLSTM
from tqdm.keras import TqdmCallback


# pragma: no cover
def loss_ordered(y_true, y_pred):  # pragma: not covered
    """
    Compute the loss for ordered logistic regression for neural networks.

    Args:
        y_true: ar-like, observed
        y_pred: ar-like, predictions

    Returns:
        float, loss value
    """
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))
                     / (K.int_shape(y_pred)[1] - 1), dtype='float32')  # pragma: no cover
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)  # pragma: no cover


gpus = tf.config.experimental.list_physical_devices('GPU')  # pragma: no cover
if gpus:  # pragma: no cover
    # Currently, memory growth needs to be the same across GPUs
    try:  # pragma: no cover
        for gpu in gpus:  # pragma: no cover
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:  # pragma: no cover
        print(e)  # pragma: no cover


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
        """
        Construct the xiRTNET.

        Args:
            params: dict, parsed yaml file
            input_dim: int, number of input dimenions for the first layer

        Returns:
            None
        """
        self.model = None
        self.input_dim = input_dim

        self.LSTM_p = params["LSTM"]
        self.dense_p = params["dense"]
        self.embedding_p = params["embedding"]
        self.learning_p = params["learning"]
        self.output_p = params["output"]
        self.siamese_p = params["siamese"]
        self.callback_p = params["callbacks"]

        self.tasks = np.concatenate([sorted(params["predictions"]["fractions"]),
                                     sorted(params["predictions"]["continues"])])
        self.tasks = [i.lower() for i in self.tasks]

    def build_model(self, siamese=False):
        """
        Build xiRTNET.

        Function can either be used to build the Siamese network, Psedolinear (concatenated cross
        links) or a normal network with a single input for linear peptides.

        Args:
            siamese: bool, if True siamese architecture is used.

        Returns:
            None
        """
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
            merge_func = self._add_siamese_connector()
            merger_layer = merge_func()([processed_a, processed_b])

            net = Model([input_a, input_b], merger_layer)
            # create the individual prediction net works
            # shortcut to acces output parameters
            act_conf = self.output_p

            tasks_ar = []
            for task_i in self.tasks:
                tmp_task = self._add_task_dense_layers(merger_layer, None)
                tmp_task = Dense(act_conf[task_i + "-dimension"],
                                 activation=act_conf[task_i + "-activation"],
                                 name=task_i)(tmp_task)
                tasks_ar.append(tmp_task)

            model_full = Model(inputs=net.input, outputs=tasks_ar)
        else:
            model_full = self._build_task_network(inlayer, input_meta=None, net=net)
        self.model = model_full

    def _add_siamese_connector(self):
        """
        Add the siamese layer to connect the individual data from the two crosslink branches.

        Returns:
            layer, merging layer, e.g. add, multiply, average, concatenate, maximum, minimum
        """
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
        return merge_func

    def _build_base_network(self):
        """
        Construct a simple network that consists of an input, embedding, and recurrent-layers.

        Function can be used to  build a scaffold for siamese networks.

        Returns:
            tuple, (input, network): the input data structure and the network structure from tf 2.0
        """
        # init the input layer
        inlayer = Input(shape=self.input_dim, name="main_input")

        # translate labels into continuous space
        net = Embedding(input_dim=self.input_dim,
                        output_dim=self.embedding_p["length"],
                        embeddings_initializer="he_normal", name="main_embedding",
                        mask_zero=True)(inlayer)

        # sequence layers (LSTM-type) + batch normalization if in config
        for i in np.arange(self.LSTM_p["nlayers"]):
            if self.LSTM_p["nlayers"] > 1:
                net = self._add_recursive_layer(net, i, name="shared{}_".format(i))
            else:
                # return only sequence when there are more than 1 recurrent layers
                net = self._add_recursive_layer(net, 1, name="shared{}_".format(i))
        return inlayer, net

    def _build_task_network(self, inlayer, input_meta, net):
        """
        Build task specific, dense layers in the xiRT architecture.

        Parameters:
            inlayer: layer, previous input layers
            input_meta: df, meta features
            net: keras model, network so far

        Returns
            model

        """
        # if desired the sequence input can be supplemented by precomputed features
        if input_meta is not None:
            in_meta = Input(shape=(input_meta,))
            net_meta = Model(inputs=in_meta, outputs=in_meta)
            net = Model(inputs=inlayer, outputs=net)
        else:
            net_meta = None

        # create the individual prediction networks
        act_conf = self.output_p

        tasks_ar = []
        for task_i in self.tasks:
            tmp_task = self._add_task_dense_layers(net, None)
            tmp_task = Dense(act_conf[task_i + "-dimension"],
                             activation=act_conf[task_i + "-activation"],
                             name=task_i)(tmp_task)
            tasks_ar.append(tmp_task)

        if input_meta is None:
            model = Model(inputs=inlayer, outputs=tasks_ar)
        else:
            model = Model(inputs=[net.input, net_meta.input],
                          outputs=tasks_ar)

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
            f_name = "GRU"

        elif self.LSTM_p["type"] == "LSTM":
            f_rnn = LSTM
            f_name = "LSTM"

        elif self.LSTM_p["type"] == "CuDNNGRU":  # pragma: no cover
            f_rnn = CuDNNGRU  # pragma: no cover
            f_name = "CuGRU"  # pragma: no cover

        elif self.LSTM_p["type"] == "CuDNNLSTM":  # pragma: no cover
            f_rnn = CuDNNLSTM  # pragma: no cover
            f_name = "CuLSTM"  # pragma: no cover
        else:
            raise KeyError("Recurrent type option not found ({})".format(
                self.LSTM_p["activity_regularization"]))

        if self.LSTM_p["bidirectional"]:
            # GRU implementations do not support activiation
            # activation = self.LSTM_p["activation"], disabled fo rnow
            lstm = Bidirectional(f_rnn(self.LSTM_p["units"], activity_regularizer=reg_act,
                                       kernel_regularizer=reg_kernel, return_sequences=return_seqs),
                                 name=name + "Bi" + f_name)(prev_layer)
        else:
            lstm = f_rnn(self.LSTM_p["units"], activation=self.LSTM_p["activation"],
                         kernel_regularizer=reg_kernel, return_sequences=return_seqs,
                         name=name + "Bi" + f_name)(prev_layer)

        # add batch normalization
        if self.LSTM_p["lstm_bn"]:
            lstm = BatchNormalization(name=name + "lstm_bn_" + str(n_layer))(lstm)
        return lstm

    def _add_task_dense_layers(self, net, net_meta=None):
        """
        Add task specific dense layers.

        If net_meta is set also adds the meta information as input for each individual layer.

        Parameters:
            net, keras network model

            net_meta: None or df, if df features should be stored there
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
        Add a dense layer.

        Parameters:
        idx: int,
                integer indicating the idx'th layer that was added
        prev_layer: keras layer,
                    Functional API object from the definition of the network.

        Returns:
            layer, a densely connected layer with dropout
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
        """
        Visualize model architecture in pdf.

        Args:
            fig_path: str, file location where the model should be stored.

        Returns:
            None
        """
        try:
            plot_model(self.model, to_file=fig_path + "xiRT_model.pdf", show_shapes=True,
                       show_layer_names=True, dpi=300, expand_nested=True)
        except ValueError as err:
            print("Encountered an ValueError, PDF is still written. ({})".format(err))

    def compile(self):
        """
        Wrappaer to compile xiRTNETwork.

        Loss, Metrics, Weights are all retrieved from the parameter file together with the
        optimizer and the network is prepared (compiled) for training using the standard
        keras / tf procedure.

        Returns:
            None
        """
        # set optimizer
        if self.learning_p["optimizer"] == "adam":
            opt = optimizers.Adam(lr=self.learning_p["learningrate"])

        elif self.learning_p["optimizer"].lower() == "sgd":
            opt = optimizers.SGD(learning_rate=self.learning_p["learningrate"],
                                 momentum=0.0, nesterov=False, name='SGD')

        elif self.learning_p["optimizer"].lower() == "rmsprob":
            opt = optimizers.RMSprop(learning_rate=self.learning_p["learningrate"])

        elif self.learning_p["optimizer"].lower() == "nadam":
            opt = optimizers.Nadam(learning_rate=self.learning_p["learningrate"])

        # get parameters from config file
        loss = {i: self.output_p[i + "-loss"] for i in self.tasks}
        metric = {i: self.output_p[i + "-metrics"] for i in self.tasks}
        loss_weights = {i: self.output_p[i + "-weight"] for i in self.tasks}

        self.model.compile(loss=loss, optimizer=opt, metrics=metric, loss_weights=loss_weights)

    def get_callbacks(self, suffix=""):
        """
        Create a list of callbacks to be passed to the fit function from the neural network model.

        Args:
            suffix: str, suf to use for the models / weights during callback savings

        Returns:
            ar-like, list of callbacks
        """
        # collect callbacks here
        callbacks = []
        prefix_path = self.callback_p["callback_path"]

        # pragma: not covered
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)

        if self.callback_p["progressbar"]:
            callbacks.append(TqdmCallback(verbose=0))

        if self.callback_p["reduce_lr"]:
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                               factor=self.callback_p["reduce_lr_factor"],
                                               patience=self.callback_p["reduce_lr_patience"],
                                               verbose=1, min_delta=1e-4, mode='min')
            callbacks.append(reduce_lr_loss)

        if self.callback_p["early_stopping"]:
            # if the val_loss does not improve, stop
            # also does load the best weights!
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                               patience=self.callback_p["early_stopping_patience"],
                               restore_best_weights=True)
            callbacks.append(es)

        if self.callback_p["check_point"]:
            # save the best performing model
            mc = ModelCheckpoint(os.path.join(prefix_path, 'xirt_model_{}.h5'.format(suffix)),
                                 monitor='val_loss', mode='min', verbose=0,
                                 save_best_only=True)

            mc2 = ModelCheckpoint(os.path.join(prefix_path, 'xirt_weights_{}.h5'.format(suffix)),
                                  monitor='val_loss', mode='min', verbose=0,
                                  save_best_only=True, save_weights_only=True)
            callbacks.append(mc)
            callbacks.append(mc2)

        if self.callback_p["log_csv"]:
            # log some stuff (what?)
            csv_logger = CSVLogger(
                os.path.join(prefix_path, 'xirt_epochlog_{}.log'.format(suffix)))
            callbacks.append(csv_logger)

        if self.callback_p["tensor_board"]:
            # use tensorboard logger
            tb = TensorBoard(log_dir=os.path.join(prefix_path, "tensorboard"),
                             embeddings_freq=0)
            callbacks.append(tb)

        return callbacks

    def print_layers(self):
        """
        Print layers form the current model.

        Returns:
            None
        """
        print([i.name for i in self.model.layers])

    def get_param_overview(self):
        """
        Print parameters from the model.

        Returns:
            None
        """
        trainable_count = np.sum([K.count_params(w) for w in self.model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in self.model.non_trainable_weights])
        print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))

    def load_weights(self, location):
        """
        Load weights.

        Args:
            location: str, location of weights

        Returns:
            None
        """
        self.model.load_weights(location)


def params_to_df(yaml_file, out_file):
    """
    Give a list of parameters as dictionary entries and transform it into a dataframe.

    Args:
        yaml_file: str, file to load the parameters from
        outfile: str, file to store the parameters to

    Returns:
        df, parameter dictionary
    """
    df_params = pd.json_normalize(yaml.load(open(yaml_file), Loader=yaml.FullLoader)).transpose()
    df_params.to_csv(out_file)
    return df_params


def reshapey(values):
    """
    Flattens the arrays that were stored in a single data frame cell.

    Formatting is needed to pass the data to the neural network input.

    Args:
        values:

    Returns:
        ar-like, formatted array
    """
    # len(values) = rows, values[0].size = ncols
    return np.array([y for x in values for y in x]).reshape(len(values), values[0].size)
