#FIX2022forpubrep: ready for publication
import numpy as np
from cytoolz.itertoolz import _outer_join
from numpy import savetxt
import os
from anomaly_detection.general.global_variables import *
import pandas as pd


class Dataset(object):
    def __init__(self, input_rep_name=None):
        # Public properties
        self.num_attributes = None
        self.go_backwards = None
        self.flat_features = None
        self.features = dict([])
        self.case_lens = None
        self.input_rep_name = input_rep_name
        self.input_rep_path = None
        self.log_of_cases_path = None
        self.ohe_rep = None  #
        self.label_ = None
        self.X_and_Y_for_tstideplus = None  # inputs for model and labels
        self.X_ohe = None
        self.X_for_tstideplus = None
        self.X_and_Y_for_aalst = None
        self.X_for_aalst = None
        self.Y_for_binet = None
        self.attr_dim = None  # number of activities from log. Incluindo teste e treinamento
        self.max_case_len=None

    def create_cases_x_activities_order(self, log_name, input_path, save=True):

        dataset_log_view = pd.read_csv(os.path.join(input_path, log_name))

        # Encode activities names
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        activities_list = np.array(list(np.sort(dataset_log_view['activity'].unique())))

        # Each activities will be encoded by a number
        encoded_activities = encoder.fit_transform(activities_list) + 1
        encoded_activities_detail = np.array([encoded_activities, activities_list]).T

        # Set some variables
        case_lens = np.array(dataset_log_view.groupby('traceid')['activity'].count())
        offsets = np.concatenate(([0], np.cumsum(case_lens)[:-1]))
        max_case_lens = case_lens.max()
        number_of_cases = case_lens.shape[0]

        # create table of encoded features in 2d (cases x order of activities).
        encoded_features = np.zeros((number_of_cases, max_case_lens, 2))

        # create table of features in 2d (cases x order of activities)
        features = np.zeros((number_of_cases, max_case_lens, 2)).astype(str)

        entire_log_w_encoded_activities = encoder.transform(dataset_log_view['activity']) + 1
        for i, (offset, case_len) in enumerate(zip(offsets, case_lens)):
            encoded_features[i, 0:case_len, 0] = entire_log_w_encoded_activities[offset:offset + case_len]
            features[i, 0:case_len, 0] = encoder.inverse_transform(
                entire_log_w_encoded_activities[offset:offset + case_len] - 1)

        # Add aditional columns (label and traceid)
        log_of_cases = pd.read_csv(os.path.join(LOGS_DIR, log_name.replace("log", "logofcases")))
        output_encoded_features = pd.DataFrame(encoded_features[:, :, 0])
        rotulos = log_of_cases['label'].apply(lambda x: x[0]).values  # rotulo
        tracesid = log_of_cases['traceid'].values  # traceid
        output_encoded_features['label'] = rotulos
        output_encoded_features['traceid'] = tracesid

        if (save == True):
            projeto_origem = TSTIDE_DIR

            # Create folder if it does not exist
            if not (os.path.isdir(projeto_origem)):
                os.mkdir(projeto_origem)

            output_encoded_features.to_csv(projeto_origem / ("encoded_features_%s" % (log_name)), sep=',', index=False,
                                           header=None)

        self.features = features
        self.X_for_tstideplus = encoded_features
        self.case_lens = case_lens
        self.X_and_Y_for_tstideplus = output_encoded_features

    def create_xes_for_aalst_approach(self, df_log_view, log_view_name, train_caseids, test_caseids, idx_iter):
        """
        Function that creates a .xes file that will be used by aalst_approach

        Parameters
        ----------
        df_log_view: dataframe containing events of the log

        Returns
        -------
        path_xes_train: String
        path_xes_test: String

        """

        from pm4py.objects.log.util import dataframe_utils
        from pm4py.objects.conversion.log import converter as log_converter
        from pm4py.objects.log.exporter.xes import exporter as xes_exporter

        # Set filenames
        path_xes_train = os.path.join(AALST_DIR,
                                      log_view_name.replace('log', 'iter' + str(idx_iter) + '_train_log').replace(
                                          '.csv', '.xes'))
        path_xes_test = os.path.join(AALST_DIR,
                                     log_view_name.replace('log', 'iter' + str(idx_iter) + '_test_log').replace('.csv',
                                                                                                                '.xes'))

        # If does not exists
        if (not (os.path.isfile(path_xes_test))) | (not (os.path.isfile(path_xes_test))):
            df_log_view = df_log_view[['traceid', 'activity']]
            df_log_view = df_log_view.rename(columns={'traceid': 'case:concept:name', 'activity': 'concept:name'})

            # Create log of training and test for that iteration
            df_log_view_train = df_log_view[df_log_view['case:concept:name'].isin(train_caseids)]
            df_log_view_test = df_log_view[df_log_view['case:concept:name'].isin(test_caseids)]

            # Convert Csv to lOg object
            event_log_object_train = log_converter.apply(df_log_view_train)
            event_log_object_test = log_converter.apply(df_log_view_test)

            xes_exporter.apply(event_log_object_train, path_xes_train)
            xes_exporter.apply(event_log_object_test, path_xes_test)

        return [path_xes_train, path_xes_test]
