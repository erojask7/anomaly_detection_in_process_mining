#FIX2022forpubrep: ready for publication
# Code based in https://github.com/tnolle/binet
import numpy as np
import sys

sys.path.insert(1, "E:\\EXPERIMENTOS2020\\anomaly_detection_artigo\\")
from anomaly_detection.general.global_variables import *
import os
from anomaly_detection.approaches.anomaly_detection_algorithm import anomaly_detection_algorithm


class TStidePlus(anomaly_detection_algorithm):
    abbreviation = 't-stide+'
    name = 't-STIDE+'

    def __init__(self, params):
        """Initialize sliding window anomaly detector.

        """
        super().__init__(params)
        self.k = params.pop('k')  # windows size
        self.thd = params.pop('thd')
        self.version_approach = params.pop('version_approach', 'original')  # Threshold for
        self.version_outlierness_calculus = params.pop('version_outlierness_calculus',
                                                       'using_mismatches')  # Threshold for
        self.modelo = None
        self.score = None
        self.set_dataset_paths()

    def get_anomaly_scores(self,ngrams):
        out=np.ones(ngrams.shape)
        for z in range(ngrams.shape[2]):
            for x in range(ngrams.shape[0]):
                for y in range(ngrams.shape[1]):
                    hash=ngrams[x,y,z]
                    if(hash in self.score.keys()):
                        out[x, y, z]=self.score[hash]
                    else:
                        out[x, y, z]=np.infty
        return out


    def set_dataset_paths(self):
        self.train_log.input_rep_path = os.path.join(LOGS_DIR, "stideplus_input", self.train_log.input_rep_name)
        self.train_log.log_of_cases_path = os.path.join(LOGS_DIR, (
            self.train_log.input_rep_name.replace("tstideplus_log", "logofcases")).replace(".pkl", ".csv"))

    def train_model(self, dataset_tr, folder_output, results, index, p_modelit_id, j, **params_training):
        versao = 'original'

        dataset = dataset_tr
        n = self.get_ngrams(dataset.X_for_tstideplus)
        num_casos = len(n)

        # Add ngrams per attribute, e.g. "0:1,1:2|0:2" and ""0:1,1:2|1:2""
        ngrams = [n]
        if dataset.num_attributes > 1:
            for i in range(dataset.num_attributes):  # i=0,1
                m = np.copy(n)
                m[:, :, i] = -1
                ngrams.append(m)
            ngrams = np.hstack(ngrams)
        else:
            ngrams = n

        ngrams = ngrams.reshape(np.product(ngrams.shape[:-2]),
                                np.product(ngrams.shape[2:]))

        if (versao == 'original'):

            ngrams1 = np.copy(ngrams)

            # It removes all ngrams that have "users" or [0 0 0.. 0]
            ngrams1 = ngrams1[ngrams1[:, 3] == -1]

            # It removes all ngrams created just for be padding elements
            max_size_of_case = dataset.case_lens.max()
            offsets_inicio = np.array([max_size_of_case * i for i in range(num_casos)])
            offsets_fim = offsets_inicio + dataset.case_lens
            x = []
            for i in range(num_casos):
                x.append(list(range(offsets_inicio[i], offsets_fim[i])))
            x1 = np.hstack(x)
            ngrams1 = ngrams1[x1]
            ngrams = ngrams1

        ngrams = np.apply_along_axis(lambda x: hash(x.tobytes()), -1,
                                     ngrams)

        if (self.version_approach == 'nolle'):

            keys, counts = np.unique(ngrams, return_counts=True,
                                     axis=0)

            counts = -np.log(counts / num_casos)

        if (self.version_approach == 'original') | (self.version_approach == 'original_normalized') | (self.version_approach=='original_percentiles'):
            # A sum of all counts is the same value than the number of ngrams

            keys, counts = np.unique(ngrams, return_counts=True,
                                     axis=0)

            nr_windows = len(ngrams)
            from sklearn import preprocessing
            counts = counts / nr_windows

        if (self.version_approach == 'original_normalized'):
            min_max_scaler = preprocessing.MinMaxScaler()
            counts = min_max_scaler.fit_transform(counts.reshape((-1, 1)))
            counts=counts.reshape((-1,))

        self.modelo = dict(k=self.k, score=dict(zip(keys, counts)))

        import pickle
        path_file_name = os.path.join(folder_output, OUT_DIRNAME,"exp%s_iter%s_modelo_treinado.pkl" % (p_modelit_id, j))
        pickle.dump(self.modelo, open(path_file_name, "wb"))


    def get_outlierness_scores(self, dataset_tst, folder_output, p_modelit_id, params, index_iteration_cv,use_mask=False,
                               level='case'):
        '''
        levels: case (one score by case). event ( one score by event). attribute ( one score by atribute- Nolle method for stide)
        '''

        # Get outlierness scores
        [scores, outlierness_scores,thd_value] = self.detect_detailed(dataset_tst)

        # Round outlierness_scores
        outlierness_scores=np.round(outlierness_scores,10)

        # Save
        path_filename=os.path.join(folder_output, OUT_DIRNAME, "exp%s_iter%s_scores.csv" % (p_modelit_id,index_iteration_cv))
        np.savetxt(path_filename,
                   scores[:,:,1],
                   delimiter=",")
        np.savetxt(path_filename.replace("scores","outlierness_scores"),
                   outlierness_scores,
                   delimiter=",")

        return [outlierness_scores,thd_value,None]

    def detect_detailed(self, dataset):
        n = self.get_ngrams(
            dataset.X_for_tstideplus)

        ngrams = []
        if dataset.num_attributes > 1:
            for i in range(dataset.num_attributes):  # 0,1
                m = np.copy(n)
                m[:, :, i] = -1

                m = m.reshape(*m.shape[:-2], np.product(m.shape[
                                                        2:]))

                m2 = np.apply_along_axis(lambda x: str(x[0]) + ',' + str(x[1]) + ',' + str(x[2]) + ',' + str(x[3]), -1,m)

                m = np.apply_along_axis(lambda x: hash(x.tobytes()), -1, m)  # m is 5000x16. A operacao hash será aplicada sobre o ultimo axis. #é feito de maneira similar do que no .fit().hash function will be apply on the axis 1 (filas. fila1, fila2,etc) [-1 -1 0 140000]
                ngrams.append(m)
            ngrams = np.dstack(ngrams)
        else:
            n.reshape(*n.shape[:-2], np.product(n.shape[2:]))
            n = np.apply_along_axis(lambda x: hash(x.tobytes()), -1, n)
            ngrams = n
        self.score = self.modelo['score']

        scores = self.get_anomaly_scores(ngrams)
        scores[:, :, 0] = 0

        # outlierness case level
        outlierness_scores = []
        if (self.version_approach == 'original_percentiles'):
            thd1 = self.predict_for_an_scaling_factor_(scores[:, :, 1], "percentiles", self.thd)

        else:
            thd1=self.thd

        for i in range(scores.shape[0]):  # case i
            scores[i, :, 1]  # a case
            number_of_windows = dataset.case_lens[i]

            number_mismatch1 = sum(
                scores[i, :number_of_windows, 1] < thd1)  # windows that has a score less than threshold #

            number_mismatch2 = sum(
                scores[i, :number_of_windows, 1] == np.inf)  # windows that does not appear in training
            outlierness_score = (number_mismatch1 + number_mismatch2) / number_of_windows

            # Porcentage of windows mismatch inside a case. Number between [0,1]
            outlierness_scores.append(
                outlierness_score)
        outlierness_scores = np.array(outlierness_scores)
        return [scores, outlierness_scores,thd1]

    def get_ngrams(self, features):
        if self.k > 1:

            pad_features = np.pad(features, ((0, 0), (self.k - 1, 0), (0, 0)),
                                  mode='constant')

            ngrams = [pad_features]

            for i in np.arange(1, self.k):  # for i=1
                ngrams.append(np.pad(pad_features[:, i:], ((0, 0), (0, i), (0, 0)),
                                     mode='constant'))
            ngrams = np.stack(ngrams, axis=-1)[:, :-(self.k - 1)]
            return ngrams
        else:
            raise Exception("k parameter should be higher than 1")

    def predict_for_an_scaling_factor_(self, vector_of_values, method_for_explore_thresholds, scaling_factor):
        if method_for_explore_thresholds == "scaling_factor_x_mean":  # scaling_factor_x_mean
            outlierness_score_mean = vector_of_values.mean()
            limiar_heuristica = scaling_factor * outlierness_score_mean

        elif method_for_explore_thresholds == None:  # scaling_factors_by_interval
            print("Not implemented yet")

        elif method_for_explore_thresholds == "percentiles":
            # If scaling_factor=0.5, it will be used  the median
            # Problem: If most cases have outlierness_score=0, it is possible than median will be zero, and several percentiless wil be zero, so
            # limiar_heuristica will be zero the most of time
            limiar_heuristica = np.percentile(vector_of_values, scaling_factor * 100)

        elif (method_for_explore_thresholds == "percentiles_unique"):
            # It resolves problem of 'percentiles' because it uses percentiles without duplicated values of percentiles
            # se scaling_factor=0.5, utilizaremos a mediana
            limiar_heuristica = np.percentile(np.unique(vector_of_values), scaling_factor * 100)

        limiar_heuristica = round(limiar_heuristica, 10)
        return limiar_heuristica