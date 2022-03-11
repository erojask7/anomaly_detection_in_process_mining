# FIX2022forpubrep: ready for publication
import sys

sys.path.insert(1, "E:\\EXPERIMENTOS2020\\anomaly_detection_artigo\\")
from anomaly_detection.general.class_dataset import Dataset
import pandas as pd
import os
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from anomaly_detection.general.global_variables import *


class anomaly_detection_algorithm(object):
    def __init__(self, params):
        self.input_rep_path = None
        self.train_log = Dataset(params.pop('input_rep_name'))
        self.modelo = None

    def create_input_representation(self, **params):
        raise NotImplementedError()

    def get_input_representation_path(self, log_name):
        raise NotImplementedError()

    def train_model(self, dataset_tr, folder_output, results, index, p_modelit_id, j, **kwargs):
       # This function returns the model

        raise NotImplementedError()


    def predict_using_several_thresholds(self, dataset_tst, folder_output,
                                         p_modelit_id, index_iteration_cv,
                                         results, Ytst_labels, index, scaling_factors, method_for_explore_thresholds,
                                         exp_group_name, params,test_caseids,save=False):
        """
        Create results from make predictions over dataset_tst

        Parameters
        ----------
        dataset_tst: it will be used for test trained model

        Returns
        -------
        results: list
            Contain the results of predictions. result[0] is Y_pred for one threshold

        """

        # Test
        use_mask=results["use_mask"].unique()[0]
        [outlierness_scores,thd_value,cases_ids] = self.get_outlierness_scores(dataset_tst, folder_output, p_modelit_id, params,
                                                         index_iteration_cv,use_mask, level='case')

        # Verify cases ids
        if cases_ids is not None:
            # If not all are Identical ( if there are any different)
            if not(all(cases_ids==test_caseids.values)):
                raise Exception("CasesIDs are differents. They should be identical.")

        # Yd to dataframe
        Yd = pd.DataFrame(Ytst_labels)  # labels
        Yd.to_csv(os.path.join(folder_output,OUT_DIRNAME, "exp%s_iter%s_test_y_true.csv" % (p_modelit_id, index_iteration_cv)),
                  sep=',',
                  encoding='utf-8', index=False, header=False)
        Ytst_labels_true = Yd

        # Evaluate an specific model using AP, AUC_ROC
        [ap, auc_roc,prc_thresholds,prc_precisions,prc_recalls] = self.evaluate_model(Ytst_labels_true, outlierness_scores, p_modelit_id, index_iteration_cv,
                                            folder_output)

        # Save results
        nexp = len(results[results['p_modelit_id'] == p_modelit_id])
        index_start = index + 1
        index_end = index + nexp
        results.loc[index_start:index_end, 'iteracao_cv'] = index_iteration_cv
        results.loc[index_start:index_end, 'exp_auc_pr'] = ap
        results.loc[index_start:index_end, 'exp_auc_roc'] = auc_roc
        results.loc[index_start:index_end, 'exp_auc_pr_thresholds'] = ' ,'.join([str(th) for th in prc_thresholds])
        results.loc[index_start:index_end, 'exp_auc_pr_precisions'] = ' ,'.join([str(th) for th in prc_precisions])
        results.loc[index_start:index_end, 'exp_auc_pr_recalls'] = ' ,'.join([str(th) for th in prc_recalls])

        # Track time
        import time
        time_tst_start = time.time()
        for scaling_factor in scaling_factors:
            index = index + 1
            self.predict_and_evaluate_for_an_scaling_factor(scaling_factor, outlierness_scores,
                                                            method_for_explore_thresholds, folder_output, p_modelit_id,
                                                            index_iteration_cv, results, Ytst_labels_true, index,
                                                            exp_group_name)
        time_tst_end = time.time()
        time_tst = time_tst_end - time_tst_start
        results.loc[index_start:index_end, 'time_tst_all_sf'] = time_tst

        return [results, index]


    def get_outlierness_scores(self, dataset_tst, folder_output, p_modelit_id, params, index_iteration_cv,
                               level='case'):

        """
        Get outlierness scores for predictions

        Parameters
        ----------
            level: It could be 'case' or 'event'

        Returns
        -------
            Array containing an outlierness score by each case. Without considering scaling factors (thresholds)

        """
        raise NotImplementedError()

    def evaluate_model(self, y_true, outlierness_scores, p_modelit_id, j, folder_output):
        """
        It calculates quality measures from outlierness_scores. Outlierness_scores is a matrix containing one score by each case


        Returns
        -------
            ap: average precision
            auc_roc: ara under roc curve
            prc_thresholds: thresholds used for precision and recall calculations
            prc_precisions: precision obtained when an specific threshold in prc_threshold is used
            prc_recalls: recall obtained when an specific threshold in prc_threshold is used
        """

        # Calculate AP
        ap = average_precision_score(y_true[0].values, outlierness_scores, pos_label='a')
        #ap = {"a": average_precision_score(y_true[0].values, outlierness_scores, pos_label='a'),
        #      "n": average_precision_score(y_true[0].values, outlierness_scores, pos_label='n')}

        # Precision-Recall Curve
        prc_precisions, prc_recalls, prc_thresholds = precision_recall_curve(y_true[0].values, outlierness_scores, pos_label='a')
        self.precision_recall_graphic(prc_precisions, prc_recalls, folder_output, p_modelit_id, j, iplot=False)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true[0].values, outlierness_scores, pos_label='a')
        auc_roc = auc(fpr, tpr)
        return [ap, auc_roc,prc_thresholds,prc_precisions,prc_recalls]

    def predict_and_evaluate_for_an_scaling_factor(self, scaling_factor, outlierness_scores,
                                                   method_for_explore_thresholds, folder_output, p_modelit_id,
                                                   index_iteration_cv,
                                                   results, Ytst_labels_true, index, exp_group_name):
        import time

        name_file = "exp%s_iter%s_sf%s_test_y_pred.csv" % (p_modelit_id, index_iteration_cv, scaling_factor)
        time_tst_start = time.time()
        [Y, limiar_heuristica] = self.predict_for_an_scaling_factor(outlierness_scores, method_for_explore_thresholds,
                                                                    scaling_factor, results,
                                                                    index, folder_output, name_file)
        time_tst_end = time.time()
        time_tst = time_tst_end - time_tst_start

        # Generate confusion matrix
        Ytst_labels_pred = Y

        [conf_matrix, pre_rec_fsc] = self.evaluate_anomaly_detection_algorithm(Ytst_labels_pred, Ytst_labels_true)

        results.loc[index, 'scaling_factor'] = scaling_factor
        results.loc[index, 'TP'] = conf_matrix[0, 0]
        results.loc[index, 'FP'] = conf_matrix[1, 0]
        results.loc[index, 'FN'] = conf_matrix[0, 1]
        results.loc[index, 'TN'] = conf_matrix[1, 1]
        results.loc[index, 'PP'] = results.loc[index, 'TP'] + results.loc[index, 'FP']
        results.loc[index, 'PN'] = results.loc[index, 'TN'] + results.loc[index, 'FN']

        # Evaluation metrics for positive class
        results.loc[index, 'precision_p'] = pre_rec_fsc[0][0]
        results.loc[index, 'recall_p'] = pre_rec_fsc[1][0]
        results.loc[index, 'fscore_p'] = pre_rec_fsc[2][0]

        # Evaluation metrics for negative class
        results.loc[index, 'precision_n'] = pre_rec_fsc[0][1]
        results.loc[index, 'recall_n'] = pre_rec_fsc[1][1]
        results.loc[index, 'fscore_n'] = pre_rec_fsc[2][1]

        # Macro average
        results.loc[index, 'precision_macro'] = np.mean([pre_rec_fsc[0][0], pre_rec_fsc[0][1]])
        results.loc[index, 'recall_macro'] = np.mean([pre_rec_fsc[1][0], pre_rec_fsc[1][1]])
        results.loc[index, 'fscore_macro'] = np.mean([pre_rec_fsc[2][0], pre_rec_fsc[2][1]])

        # Save time
        results.loc[index, 'time_tst_sec'] = time_tst
        return limiar_heuristica

    def predict_for_an_scaling_factor(self, outlierness_score, method_for_explore_thresholds, scaling_factor, results,
                                      index, folder_output, name_file):

        if method_for_explore_thresholds == "scaling_factor_x_mean":
            outlierness_score_mean = outlierness_score.mean()
            limiar_heuristica = scaling_factor * outlierness_score_mean

        elif method_for_explore_thresholds == None:  # scaling_factors_by_interval
            print("not implemented yet")

        elif method_for_explore_thresholds == "percentiles":  # percentiles
            # If scaling_factor=0.5, the threshold will be the percentil 50, it means the median
            # Problem: If most cases have outlierness_score=0, it is possible than median will be zero, and several percentiless wil be zero, so
            # limiar_heuristica will be zero the most of time
            limiar_heuristica = np.percentile(outlierness_score, scaling_factor * 100)

        elif method_for_explore_thresholds == "percentiles_unique":
            # It resolves problem of 'percentiles' because it uses percentiles without duplicated values of percentiles
            limiar_heuristica = np.percentile(np.unique(outlierness_score), scaling_factor * 100)

        limiar_heuristica = round(limiar_heuristica, 10)
        results.loc[index, 'threshold_value'] = limiar_heuristica

        # Generation of predictions of model (Y)
        Y_labels = pd.Series(outlierness_score > limiar_heuristica)
        Y_labels[outlierness_score > limiar_heuristica] = 'a'
        Y_labels[outlierness_score <= limiar_heuristica] = 'n'

        # Save y_pred
        Y_labels_ = pd.DataFrame(Y_labels)
        Y_labels_.to_csv(
            os.path.join(folder_output, OUT_DIRNAME, name_file),
            sep=',', encoding='utf-8', index=False, header=False)
        return [Y_labels, limiar_heuristica]

    def precision_recall_graphic(self, precision, recall, folder_output, p_modelit_id, j, iplot=False,
                                 print_figure=True):
        import os
        import plotly.graph_objs as go

        trace1 = go.Scatter(x=recall,
                            y=precision,
                            )
        data = [trace1]

        layout = go.Layout(title='Precision Recall Curve',
                           yaxis={'title': 'Precision',
                                  'range': [0, 1]},
                           xaxis={'title': 'Recall',
                                  'range': [0, 1]}
                           )

        # Create figure that will be presented
        fig = go.Figure(data=data, layout=layout)

        # Show figure
        # if(iplot==True):
        #     py.iplot(fig)
        # else:
        #     py.plot(fig)

        if (print_figure == True):
            fig.write_image(os.path.join(folder_output, OUT_DIRNAME,"exp%s_iter%s_pr.png" % (p_modelit_id, j)))
            # fig.write_image("fig4.jpeg")

    def evaluate_anomaly_detection_algorithm(self, y_pred, y_true):
        import numpy as np
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=np.array(['a', 'n']))
        prfs = precision_recall_fscore_support(y_true, y_pred, average=None)

        return [cm, prfs]