#FIX2022forpubrep: ready for publication
# Code based in https://github.com/tnolle/binet
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(1, "E:\\EXPERIMENTOS2020\\anomaly_detection_artigo\\")
from anomaly_detection.general.global_variables import *
from anomaly_detection.approaches.anomaly_detection_algorithm import anomaly_detection_algorithm


class DAE(anomaly_detection_algorithm):

    def __init__(self, params):

        super().__init__(params)

        self.no = params.pop('no')  # number of neurons in each hidden layers
        self.nitmax = params.pop('nitmax')  # maximum number of epochs for training
        self.modelo = []
        self.nro_camadas_ocultas = params.pop('nro_camadas_ocultas')
        self.gaussian_noise_std = params.pop('gaussian_noise_std')
        self.funcao_f = params.pop('funcao_f')
        self.funcao_g = params.pop('funcao_g')
        self.alfa = params.pop('alfa')
        self.batch_size = params.pop('batch_size')
        self.early_stopping_patiente = params.pop('early_stopping_patiente')
        self.dropout = params.pop('dropout')  # 0.5
        self.optimizer_beta2 = params.pop('optimizer_beta2')  # 0.99
        self.early_stopping_metric = params.pop('early_stopping_metric')
        self.input_rep_path = None
        self.version_approach = params.pop('version_approach', 'original')  # Threshold for
        self.version_outlierness_calculus = params.pop('version_outlierness_calculus',
                                                       'using_mismatches')  # Threshold for
        if (self.early_stopping_metric is None):
            self.early_stopping_metric = 'val_loss'
        self.set_dataset_paths()

    def set_dataset_paths(self):
        self.train_log.input_rep_path = os.path.join(LOGS_DIR, "one_hot_encoding", self.train_log.input_rep_name)
        self.train_log.log_of_cases_path = os.path.join(LOGS_DIR,
                                                        self.train_log.input_rep_name.replace("ohe_log", "logofcases"))

    @staticmethod
    def model_fn(self, features):

        # Import Keras
        from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam

        # Number of inputs or numbr of neurons in the input layer (ne)
        ne = features.shape[1]

        # Input layer
        input = Input(shape=(ne,), name='input')
        x = input

        # Noise layer
        # if noise is not None:
        x = GaussianNoise(self.gaussian_noise_std)(x)  # Just active in training

        # Hidden layer
        for i in range(self.nro_camadas_ocultas):
            x = Dense(int(self.no[i]), activation=self.funcao_f, name=f'hid{i + 1}')(x)
            x = Dropout(self.dropout)(x)

        # Output layer
        output = Dense(ne, activation=self.funcao_g, name='output')(x)

        # Set model
        modelo = Model(inputs=input, outputs=output)

        # Compile Model
        modelo.compile(
            optimizer=Adam(lr=self.alfa, beta_2=self.optimizer_beta2),
            loss='mean_squared_error',
        )
        # modelo.summary()
        return modelo

    def train_model(self, dataset_tr, folder_output, results, index, p_modelit_id, j, **kwargs):
        """
        # Xtr: treining set
        # Ytr: labels of training set
        # XVal: validation set
        # YVal: labels of validation set
        """

        Xtr = dataset_tr.X_ohe
        dataset_val = kwargs.get('dataset_val')
        Xval = dataset_val.X_ohe
        Yval = Xval
        from tensorflow.keras.callbacks import EarlyStopping

        # Create model
        self.modelo = self.model_fn(self, Xtr)
        nit = 0

        if (self.early_stopping_patiente is ''):  # If we do not want early stopping
            # Train model
            history = self.modelo.fit(Xtr, Xtr, epochs=self.nitmax, validation_data=(Xval, Yval),
                                      batch_size=self.batch_size)
        else:
            # Early Stopping
            early_stopping = EarlyStopping(
                # The validation function loss
                monitor=self.early_stopping_metric,

                # If improvement is not seen in last 10 epochs, training will stop

                patience=self.early_stopping_patiente,

                # restore_best_weights=True,
                verbose=True
            )

            # Train modelo
            if (type(Xval) == int):  # If we do not have validation set
                if (Xval == 0):
                    history = self.modelo.fit(Xtr, Xtr, epochs=self.nitmax, batch_size=self.batch_size,
                                              callbacks=[early_stopping])
            else:  # If we do have validation set
                history = self.modelo.fit(Xtr, Xtr, epochs=self.nitmax, validation_data=(Xval, Yval),
                                          batch_size=self.batch_size, callbacks=[early_stopping])
            nit = early_stopping.stopped_epoch  # numero de epocas iteradas no treinamento antes de parar

        if (nit == 0):
            nit = self.nitmax

        self.modelo.save(os.path.join(folder_output, OUT_DIRNAME, "exp%s_iter%s_modelo_treinado.h5" % (p_modelit_id, j)))

        # EQM figure
        from anomaly_detection.general import utils_da
        utils_da.create_loss_figure(folder_output, history, p_modelit_id, j)

        # Save some data about training
        train_loss_last = history.history['loss'][-1]

        try:
            val_loss_last = history.history['val_loss'][-1]
        except:
            # If val_loss does not exist,  val_loss_last will be -1.
            # It means that there are not an validation set
            pass

        # Autoencoder parameters after training
        nexp = len(results[results['p_modelit_id'] == p_modelit_id])

        # nit is the epoch number in which the neural network stopped the training
        results.loc[index + 1:index + nexp, 'nit'] = nit

        # The last loss obtained in training set
        results.loc[index + 1:index + nexp, 'train_loss_last'] = train_loss_last

        # The last loss obtained in validation set
        results.loc[index + 1:index + nexp, 'val_loss_last'] = val_loss_last

    def get_outlierness_scores(self, dataset_tst, folder_output, p_modelit_id, params, index_iteration_cv, use_mask=False,
                               level='case'):

        # Retrieve variavels
        Xtst = dataset_tst.X_ohe
        Ytst = dataset_tst.Y_ohe

        # Try reconstruct inputs
        Yout_test = self.modelo.predict(Xtst)  # output of model 324x600

        # Get outlierness scores
        [EQMs, scores] = self.get_scores(Yout_test, Ytst, dataset_tst, use_mask)  # EQMs é matriz ndarray de EQM de dimensao 1xN (N,)
        outlierness_scores = EQMs

        # Round outlierness_scores
        outlierness_scores = np.round(outlierness_scores, 4)

        # Save
        path_filename=os.path.join(folder_output, OUT_DIRNAME, "exp%s_iter%s_scores.csv" % (p_modelit_id,index_iteration_cv))
        np.savetxt(path_filename,
                  scores,
                  delimiter=",")
        np.savetxt(path_filename.replace("scores","outlierness_scores"),
                   outlierness_scores,
                   delimiter=",")

        return [outlierness_scores,None,None]

    def convert_square_errors_to_scores(self, square_errors, dataset_tst):

        split = np.cumsum(np.tile([dataset_tst.attr_dim], [dataset_tst.max_case_len]), dtype=int)[:-1]

        square_errors = np.split(square_errors, split, axis=1)  # it returns a list. len(errors)=32

        # np.mean gets the mean of square_errors in each event
        square_errors = np.array([np.mean(a, axis=1) if len(a) > 0 else 0.0 for a in square_errors])

        scores = square_errors.T  # 5000x16x2 # if just activities: 5000x16
        return scores


    def get_scores(self, Yout_test, Y_test, dataset_tst, use_mask=False):
        import numpy as np
        erro = Yout_test - Y_test
        N = len(Yout_test)
        ns = Yout_test.shape[1]  # number of outpus ( number of neurons in the output layer)
        errors_quadrado=erro * erro

        # scores are the MSE for each event. scores will be: #number_of_cases x max_nr_events
        scores = self.convert_square_errors_to_scores(errors_quadrado, dataset_tst)
        if use_mask:
            mask = self.create_mask(scores, dataset_tst.case_lens) # for not considering padding into scores
            scores= np.ma.array(scores, mask=mask)

        EQMs2=np.mean(scores,axis=1)

        return [EQMs2,scores.data]

    def create_mask(self,errors,case_lens):
        mask = np.ones(errors.shape, dtype=bool) # 5000x16 #initialize using ones
        for m, j in zip(mask, case_lens):
            m[:j] = False
        return mask