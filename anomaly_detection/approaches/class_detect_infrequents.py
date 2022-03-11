#FIX2022forpubrep: ready for publication
import numpy as np
import sys

sys.path.insert(1, "E:\\EXPERIMENTOS2020\\anomaly_detection_artigo\\")
from anomaly_detection.general.global_variables import *
import os
from anomaly_detection.approaches.anomaly_detection_algorithm import anomaly_detection_algorithm


class DetectInfrequents(anomaly_detection_algorithm):

    def __init__(self, params):
        """Initialize sliding window anomaly detector.

        """
        super().__init__(params)
        self.version_approach = params.pop('version_approach', 'original')  # original_mod e original
        self.version_outlierness_calculus = params.pop('version_outlierness_calculus', 'original')  # Threshold for
        self.modelo = None
        self.score = None
        self.set_dataset_paths()

    def set_dataset_paths(self):
        self.train_log.input_rep_path = os.path.join(LOGS_DIR, "stideplus_input", self.train_log.input_rep_name)
        self.train_log.log_of_cases_path = os.path.join(LOGS_DIR, (
            self.train_log.input_rep_name.replace("tstideplus_log", "logofcases")).replace(".pkl", ".csv"))

    def train_model(self, dataset_tr, folder_output, results, index, p_modelit_id, j, **params_training):

        raise NotImplementedError()

    def get_outlierness_scores(self, dataset_tst, folder_output, p_modelit_id, params, index_iteration_cv, use_mask=False,
                               level='case'):
        '''
        levels: case (one score by case). event ( one score by event). attribute ( one score by atribute- Nolle method for stide)
        '''

        # Get outlierness scores
        scores = self.get_scores(dataset_tst)

        # We are not doing any transformation over scores
        outlierness_scores = self.get_outlierness_scores_from_scores(dataset_tst, scores)

        # Round outlierness_scores
        outlierness_scores=np.round(outlierness_scores,10)

        # Save
        path_filename=os.path.join(folder_output, OUT_DIRNAME, "exp%s_iter%s_scores.csv" % (p_modelit_id,index_iteration_cv))
        np.savetxt(path_filename,
                   scores,
                   delimiter=",")
        np.savetxt(path_filename.replace("scores","outlierness_scores"),
                   outlierness_scores,
                   delimiter=",")
        self.modelo = None
        return [outlierness_scores,None,None]

    def get_scores(self, dataset_tst):
        # Create a hash by each case
        tst_hashes = np.apply_along_axis(lambda x: hash(x.tobytes()), -1, dataset_tst.X_for_tstideplus[:, :, 0])

        if (self.modelo is None):  # If modelo was not created
            # Count how many cases exists by each trace
            traces_hashes_unq, counts = np.unique(tst_hashes, return_counts=True, axis=-1)
            num_cases = len(dataset_tst.X_for_tstideplus[:, :, 0])  # 5000 n_grams #numero de casos

            # Calulate 1-frequency
            scores = counts / num_cases
            self.modelo = dict(score_table=dict(zip(traces_hashes_unq, scores)))

        scores = self.get_anomaly_scores(
            tst_hashes)  #o scores[:,:,0] é de "usuarios". o 1 é de "nomes de atividades"
        return scores

    def get_anomaly_scores(self, tst_hashes):
        score_table = self.modelo['score_table']
        scores = []
        for tst_hash in tst_hashes:
            if tst_hash in score_table.keys():
                scores.append(score_table[tst_hash])
            else:
                scores.append(np.infty)
        return np.array(scores)

    def get_outlierness_scores_from_scores(self, dataset_tst, scores1):
        scores1=1-scores1
        return scores1
