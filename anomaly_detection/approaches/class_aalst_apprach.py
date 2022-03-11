#FIX2022forpubrep: rady for publication
import numpy as np
import sys

sys.path.insert(1, "E:\\EXPERIMENTOS2020\\anomaly_detection_artigo\\")
from anomaly_detection.general.global_variables import *
import os
from anomaly_detection.approaches.anomaly_detection_algorithm import anomaly_detection_algorithm

class AalstApproach(anomaly_detection_algorithm):

    def __init__(self, params):
        """Initialize sliding window anomaly detector.
        """
        super().__init__(params)
        self.kk = params.pop('kk')  # subsequence max size
        self.thdd = params.pop('thdd')  # Threshold
        self.version_approach = params.pop('version_approach', 'original')
        self.version_outlierness_calculus = params.pop('version_outlierness_calculus',
                                                       'using_mismatches')
        self.modelo = None
        self.score = None
        self.set_dataset_paths()

    def set_dataset_paths(self):
        self.train_log.input_rep_path = os.path.join(LOGS_DIR, "aalst_input", self.train_log.input_rep_name)
        self.train_log.log_of_cases_path = os.path.join(LOGS_DIR, (
            self.train_log.input_rep_name.replace("ohe_log", "logofcases")).replace(".pkl", ".csv"))

    def train_model(self, dataset_tr, folder_output, results, index, p_modelit_id, j, **params_training):
        """
        Primary function. This function sets self.model, wich will be used by get_outlierness_scores

        """

        from py4j.java_gateway import JavaGateway
        gateway = JavaGateway()

        o_vaalst_java = gateway.entry_point

        xes_input_path = dataset_tr.X_for_aalst
        thdd = self.thdd  # Threshold
        kk = self.kk  # Subsequence max length kk=self.kk

        o_vaalst_java.train_model(xes_input_path, folder_output, str(p_modelit_id), thdd, kk)
        self.modelo = o_vaalst_java

    def get_outlierness_scores(self, dataset_tst, folder_output, p_modelit_id, params, index_iteration_cv,use_mask=False,
                               level='case'):
        """
        Primary function for get outlierness scores

        Parameters
        ----------
        dataset_tst: dataset where model will be applied
        folder_output: folder to save results
        p_modelit_id: id of model
        level   :  case (one score by case). event ( one score by event). Attribute ( one score by atribute- Nolle method for stide)

        Returns
        -------
        outlierness_scores  : np.array nx1

        """
        # Get outlierness scores
        scores = self.get_scores(dataset_tst)
        [outlierness_scores,cases_ids] = self.get_outlierness_scores_from_scores(dataset_tst, scores)

        # Round outlierness_scores
        outlierness_scores=np.round(outlierness_scores,10)

        # Save
        path_filename=os.path.join(folder_output, OUT_DIRNAME, "exp%s_iter%s_scores.csv" % (p_modelit_id,index_iteration_cv))
        np.savetxt(path_filename.replace("scores.csv","scores0.csv"),
                   scores[:,0,:],
                   delimiter=",")
        np.savetxt(path_filename.replace("scores.csv","scores1.csv"),
                   scores[:,1,:],
                   delimiter=",")
        np.savetxt(path_filename.replace("scores","outlierness_scores"),
                   outlierness_scores,
                   delimiter=",")

        return [outlierness_scores,None,cases_ids]

    def get_outlierness_scores_from_scores(self, dataset_tst, scores1):
        # Scores represent the Conditional Probability of each window
        outlierness_scores = []

        # cases, k(size of subsequence), index da janela (a último coluna é do cases_ids)
        scores = scores1[:, :, :-1]
        cases_ids=scores1[:,1,-1].astype(int)

        if (self.version_outlierness_calculus == 'using_mismatches'):
            for i in range(scores.shape[0]):  # case i,scors
                number_of_windows = dataset_tst.case_lens[i]  # It seem that it is not necessary to apply Masks

                # Windows that has a score (CP) less than threshold
                number_mismatch1 = np.sum(np.any(scores[i, :, :number_of_windows] < self.thdd, axis=0))

                outlierness_score = number_mismatch1 / number_of_windows
                outlierness_scores.append(
                    outlierness_score)  # Porcentage of windows mismatch inside a case. Number betwenn [0,1]
        outlierness_scores = np.array(outlierness_scores)
        return [outlierness_scores,cases_ids]

    def get_scores(self, dataset_tst):
        if (self.version_approach == 'original'):  # It calculates CP just for more frequent windows
            scores = np.array(self.modelo.get_scores(dataset_tst.X_for_aalst, 0))
        if (self.version_approach == 'original_mod'):  # It calculates CP for every window
            scores = np.array(self.modelo.get_scores(dataset_tst.X_for_aalst, 1))

        return scores