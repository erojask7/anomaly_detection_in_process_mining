#FIX2022forpubrep: ready for publication
# Code based in https://github.com/tnolle/binet
import numpy as np
import os
import sys
sys.path.insert(1, "E:\\EXPERIMENTOS2020\\anomaly_detection_artigo\\")
from anomaly_detection.general.global_variables import *
from anomaly_detection.approaches.anomaly_detection_algorithm import anomaly_detection_algorithm

class Binet1(anomaly_detection_algorithm):

    def __init__(self, params):

        super().__init__(params)

        self.alfa = params.pop('alfa')
        self.optimizer_beta2 = params.pop('optimizer_beta2')
        self.batch_size = params.pop('batch_size')
        self.nitmax = params.pop('nitmax')
        self.early_stopping_patiente = params.pop('early_stopping_patiente')
        self.set_dataset_paths()
        self.n_gru_units = None
        self.decoder = params.pop('decoder')
        self.version_approach = params.pop('version_approach', 'original')
        self.version_outlierness_calculus = params.pop('version_outlierness_calculus',
                                                       'original')  # options: 'original', 'eqm'

    def set_dataset_paths(self):
        self.train_log.input_rep_path = os.path.join(LOGS_DIR, "stideplus_input", self.train_log.input_rep_name)
        self.train_log.log_of_cases_path = os.path.join(LOGS_DIR, (
            self.train_log.input_rep_name.replace("tstideplus_log", "logofcases")).replace(".pkl", ".csv"))

    @staticmethod
    def model_fn(self, dataset):

        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        from tensorflow.keras.layers import Embedding
        from tensorflow.keras.layers import GRU
        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras import losses

        attr_dim = dataset.attr_dim  # vocabulary size
        categorical_loss = losses.categorical_crossentropy
        loss_map = {}

        features = dataset.X_for_tstideplus

        # Build inputs (and encoders if enabled) for past events
        embeddings = []
        inputs = []
        past_outputs = []
        feature, attr_key = features, 'name'

        i = Input(shape=(None,), name='past_name')
        inputs.append(i)

        # Set embeddings
        voc_size = int(attr_dim + 1)
        emb_size = np.clip(int(voc_size / 10), 2,
                           16)
        embedding = Embedding(input_dim=voc_size,
                              output_dim=emb_size,
                              input_length=feature.shape[1],
                              mask_zero=True)
        embeddings.append(embedding)

        x = embedding(i)

        # Create encoder
        x, _ = GRU(self.n_gru_units,
                   return_sequences=True,
                   return_state=True,# Whether to return the last state in addition to the output
                   name='past_encoder_name')(x)
        x = BatchNormalization()(x)

        past_outputs.append(x)

        # Build output layers for each attribute to predict
        outputs = []

        x = past_outputs  # past_outputs have two models. One for activit and other for user

        # if we have just one attribute (name)
        x = x[0]

        # Set decoder
        if self.decoder: # in experiments it was set False
            x = GRU(self.n_gru_units,
                    return_sequences=True,
                    name='decoder_name')(x)
            x = BatchNormalization()(x)

        # Output activation function
        activation = 'softmax'

        o = Dense(int(attr_dim), activation=activation, name='out_name')(x)

        outputs.append(o)

        # Combine features and build model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(
            optimizer=Adam(lr=self.alfa, beta_2=self.optimizer_beta2),
            loss=categorical_loss
        )

        return model

    def train_model(self, dataset_tr, folder_output, results, index, p_modelit_id, j, **kwargs):  # Implement

        Ytr = dataset_tr.Y_for_binet
        validation_split = 0.1

        self.n_gru_units = max(dataset_tr.case_lens) * 2
        self.modelo = self.model_fn(self, dataset_tr)

        from tensorflow.keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(
            monitor="val_loss",
            # If improvement is not seen in last 10 epochs, training will stop
            patience=self.early_stopping_patiente,

            verbose=True
        )
        history = self.modelo.fit(
            dataset_tr.X_for_tstideplus,
            Ytr,
            batch_size=self.batch_size,
            epochs=int(float(self.nitmax)),
            validation_split=validation_split,
            verbose=1,
            callbacks=[early_stopping],
            **kwargs
        )

        self.modelo.save(os.path.join(folder_output, OUT_DIRNAME,"exp%s_iter%s_modelo_treinado.h5" % (p_modelit_id, j)))

        # Grafica do EQM
        from anomaly_detection.general import utils_da
        utils_da.create_loss_figure(folder_output, history, p_modelit_id, j)

        # Save some data about training
        train_loss_last = history.history['loss'][-1]

        # Autoencoder parameters after training
        nexp = len(results[results['p_modelit_id'] == p_modelit_id])
        nit = len(history.history['loss'])
        results.loc[index + 1:index + nexp, 'nit'] = nit  # salvar numero de epoca na qual a rede parou de treinar
        results.loc[index + 1:index + nexp, 'train_loss_last'] = train_loss_last
        results.loc[index + 1:index + nexp, 'n_gru_units'] = self.n_gru_units


    def get_outlierness_scores(self, dataset_tst, folder_output, p_modelit_id, params, index_iteration_cv, use_mask=False,
                               level='case'):
        """
        Primary function

        Parameters
        ----------
        dataset_tst:
        folder_output: folder to save results
        p_modelit_id: id of model
        level   :  case (one score by case). event ( one score by event). Attribute ( one score by atribute- Nolle method for stide)

        Returns
        -------
        outlierness_scores  : np.array nx1

        """
        Xtst = dataset_tst.X_for_tstideplus
        Ytst = dataset_tst.Y_for_binet
        scores = self.modelo.predict(Xtst)  # output of model
        # Get outlierness scores

        if self.version_outlierness_calculus == 'original':
            # Transform outputs(probabilities) to outlierness scores by each event               
            scores = self.get_scores(dataset_tst, scores)  # 827x18x1

        outlierness_scores = self.get_outlierness_scores_from_scores(dataset_tst, scores)  # 827x18x18 ()

        # Round outlierness_scores
        outlierness_scores=np.round(outlierness_scores,4)

        # Save
        path_filename=os.path.join(folder_output, OUT_DIRNAME, "exp%s_iter%s_scores.csv" % (p_modelit_id,index_iteration_cv))
        np.savetxt(path_filename,
                   scores[:,:,0],
                   delimiter=",")
        np.savetxt(path_filename.replace("scores","outlierness_scores"),
                   outlierness_scores,
                   delimiter=",")

        return [outlierness_scores,None,None]

    def get_outlierness_scores_from_scores(self, dataset_tst, scores):  # Implement
        # Get outlierness scores
        if self.version_outlierness_calculus == 'original':
            outlierness_scores = np.amax(scores, axis=(1, 2))  # 827x1

        elif self.version_outlierness_calculus == 'eqm':
            outlierness_scores = self.calculo_erros_reproducao(scores, dataset_tst.Y_for_binet)

        return outlierness_scores

    def get_scores(self, dataset_tst, Yout_test):  # Implement
        Xtst = dataset_tst.X_for_tstideplus
        Ytst = dataset_tst.Y_for_binet

        # START Extract naive
        predictions = [Yout_test]

        onehot_features = [Ytst]

        # Add perfect prediction for start symbol
        from tensorflow.keras.utils import to_categorical
        onehot_features_full = [to_categorical(Xtst)[:, :, 1:]]  # incluidng frst event
        perfect = [f[0, :1] for f in onehot_features_full]
        for i in range(len(predictions)):
            predictions[i][:, 1:] = predictions[i][:, :-1]
            predictions[i][:, 0] = perfect[i]

        # Send parameters to function
        features = dataset_tst.X_for_tstideplus.reshape(
            (dataset_tst.X_for_tstideplus.shape[0], dataset_tst.X_for_tstideplus.shape[1], 1))
        predictions
        scores = self.binet_scores_fn(features, predictions)
        return scores

    def binet_scores_fn(self, features, predictions):
        ps = [np.copy(p) for p in predictions]
        args = [p.argsort(axis=-1) for p in ps]  # It returns the indexes than will be order this array
        [p.sort(axis=-1) for p in ps]
        sums = [1 - np.cumsum(p, axis=-1) for p in ps]
        indices = [np.argmax(a + 1 == features[:, :, i:i + 1], axis=-1).astype(int) for i, a in
                   enumerate(args)]

        scores = np.zeros(features.shape, dtype=np.float64)
        for (i, j, k), f in np.ndenumerate(features):
            if f != 0 and k < len(predictions):
                scores[i, j, k] = sums[k][i, j][indices[k][i, j]]
        if 'p2p' in self.train_log.input_rep_path:
            scores[:, 0, :] = 0
        return scores

    def calculo_erros_reproducao(self, Yout_test, Y_test):
        import numpy as np
        erro = Yout_test - Y_test
        N = len(Yout_test)
        ns = Yout_test.shape[1]  # numero de saidas(numero de neuronios de saida)
        EQMs = np.sum(np.sum(erro * erro, axis=2), axis=1) / ns
        return EQMs
