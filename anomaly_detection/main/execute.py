#FIX2022forpubrep: ready for publication
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from anomaly_detection.general.global_variables import *
from anomaly_detection.general.class_dataset import Dataset
from anomaly_detection.approaches.class_stide import TStidePlus
from anomaly_detection.approaches.class_aalst_apprach import AalstApproach
from anomaly_detection.approaches.class_binet1 import Binet1
from anomaly_detection.approaches.class_detect_infrequents import DetectInfrequents
from anomaly_detection.approaches.class_random import Random


def execute_exp(experiment, scaling_factors, folder_output, fast_debug=False, save=True):
    # Import libraries
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from anomaly_detection.approaches.class_autoencoder_denoising_nolle import DAE as DAE
    from numpy import savetxt

    pd.set_option('display.max_columns', 8)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)

    # parameters
    p_modelit_id = experiment['p_modelit_id']
    exp_group_name = experiment['exp_group_name']
    log_name = experiment['log_name']
    k = experiment['s_k_cv']
    model_type = experiment['p_modelit_type']
    method_for_explore_thresholds = experiment['s_method_for_explore_thresholds']
    s_strategy_tr_tst = experiment['s_strategy_tr_tst']

    if (experiment['p_modelit_type'] == 'autoencoder_nolle'):
        funcao_f = experiment['funcao_f']
        funcao_g = experiment['funcao_g']
        nitmax = experiment['nitmax']
        batch_size = experiment['batch_size']
        early_stopping_patiente = experiment['early_stopping_patiente']
        early_stopping_metric = experiment['early_stopping_metric']
        nro_camadas_ocultas = experiment['nro_camadas_ocultas']
        gaussian_noise_std = experiment['gaussian_noise_std']
        dropout = experiment['dropout']
        optimizer_beta2 = experiment['optimizer_beta2']
        alfa = experiment['alfa']
        no = experiment['no']

    # Set dataframe for results
    results = experiment.to_frame().T
    results = pd.concat([results] * len(scaling_factors) * k, ignore_index=True)

    # Add columns for results of evaluation
    results[['iteracao_cv',
             'scaling_factor',
             'threshold_value',
             'TP',
             'FN', 'FP', 'TN',
              'precision_p', 'recall_p', 'fscore_p','precision_n', 'recall_n', 'fscore_n', 'precision_macro', 'recall_macro', 'fscore_macro',
              'time_tr_sec', 'time_tst_sec', 'time_tst_all_sf','PP', 'PN',
             ]]=-1

    if (experiment['p_modelit_type'] == 'autoencoder_nolle') | (experiment['p_modelit_type'] == 'binet1'):
        results[['nit', 'val_loss_last', 'train_loss_last']] = -1
    if experiment['p_modelit_type'] == 'binet1':
        results['n_gru_units'] = -1
    if experiment['p_modelit_type']== 'tstideplus':
        results['p_modelit_thd_value']=-1

    # Set test percentage
    if ('holdout' in s_strategy_tr_tst):
        test_percentage = 0.3
    else:
        test_percentage = None

    validation_percentage = 0
    common_params = dict(
        version_approach=experiment['p_modelit_vapproach'],
        version_outlierness_calculus=experiment['p_modelit_voutlierness_calc'],
    )
    if (experiment['p_modelit_type'] == 'autoencoder_nolle'):
        validation_percentage = 0.3

        list_of_number_of_neurons_in_hidden_layers = [no, no]  # number of hidden neurons in the first and second layer respectively
        params = dict(no=list_of_number_of_neurons_in_hidden_layers,
                      nitmax=nitmax,
                      nro_camadas_ocultas=nro_camadas_ocultas,
                      gaussian_noise_std=gaussian_noise_std,
                      funcao_f=funcao_f,
                      funcao_g=funcao_g,
                      alfa=alfa,
                      batch_size=batch_size,
                      early_stopping_patiente=early_stopping_patiente,
                      dropout=dropout,
                      optimizer_beta2=optimizer_beta2,
                      early_stopping_metric=early_stopping_metric,
                      input_rep_name=log_name
                      )
        oADA = DAE({**common_params, **params})

    elif (experiment['p_modelit_type'] == 'binet1'):
        validation_percentage = 0.3

        params = dict(nitmax=experiment['nitmax'],
                      alfa=experiment['alfa'],
                      batch_size=int(experiment['batch_size']),
                      optimizer_beta2=experiment['optimizer_beta2'],
                      early_stopping_patiente=experiment['early_stopping_patiente'],
                      decoder=experiment['decoder'],
                      input_rep_name=log_name,
                      )
        oADA = Binet1({**common_params, **params})

    elif experiment['p_modelit_type'] == 'tstideplus':

        params = dict(k=int(experiment['p_modelit_windows_size']),
                      thd=float(experiment['p_modelit_thd']),
                      input_rep_name=log_name,
                      )
        oADA = TStidePlus({**common_params, **params})
    elif experiment['p_modelit_type'] == 'aalst_approach':

        params = dict(kk=int(experiment['p_modelit_kk']),
                      thdd=float(experiment['p_modelit_thdd']),
                      input_rep_name=log_name,
                      )
        oADA = AalstApproach({**common_params, **params})

    elif experiment['p_modelit_type'] == 'detect_infrequents':

        params = dict(input_rep_name=log_name)
        oADA = DetectInfrequents({**common_params, **params})

    elif experiment['p_modelit_type'] == 'random':

        params = dict(input_rep_name=log_name)
        oADA = Random({**common_params, **params})

    if ('.pkl' in log_name):
        dataset = pickle.load(open(oADA.train_log.input_rep_path, "rb"))
        X_for_model_and_Y_labels = dataset.X_and_Y_for_tstideplus
        case_lens = dataset.case_lens
        attr_dim = np.max([X_for_model_and_Y_labels.iloc[:, :-2]])
        log_of_cases = pd.read_csv(os.path.join(LOGS_DIR, oADA.train_log.log_of_cases_path), sep=',')
    else:
        X_for_model_and_Y_labels = pd.read_csv(oADA.train_log.input_rep_path, header=None)
        log_of_cases = pd.read_csv(os.path.join(LOGS_DIR, oADA.train_log.log_of_cases_path), sep=',')
        log = pd.read_csv(os.path.join(LOGS_DIR, oADA.train_log.log_of_cases_path.replace("logofcases","log")), sep=',')
        case_lens = np.array(log_of_cases['nr_events'])
        attr_dim = len(log["activity"].unique())

    X_for_model = np.array(X_for_model_and_Y_labels.iloc[:, :-2])  # Dados do input_rep
    Y_labels = np.array(X_for_model_and_Y_labels.iloc[:, -2])  # Rotulos do input_rep

    import pickle
    import os
    # If file already exist, we load it.
    path_iteracoes_cv_filename = os.path.join(LOG_ITERACOES_CV, ("cv_%s" % (log_name.replace(".csv",".pkl").replace("ohe_","").replace("tstideplus_",""))))
    if (s_strategy_tr_tst == 'cv_without_traces_duplicated_tst_tr') & (os.path.isfile(path_iteracoes_cv_filename)):
        iteracao = pickle.load(open(path_iteracoes_cv_filename, 'rb'))
    else: # Otherwise, we get iteracao and save it
        [iteracao,iteracao_info] = get_idx_for_training_and_test_dataset(X_for_model_and_Y_labels, X_for_model, Y_labels, log_name,
                                                                         exp_group_name, k,
                                                                         s_strategy_tr_tst,
                                                                         test_percentage, oADA)
        if s_strategy_tr_tst == 'cv_without_traces_duplicated_tst_tr':
            # Save iteracao to be reused later
            if not (os.path.isdir(LOG_ITERACOES_CV)):
                os.makedirs(LOG_ITERACOES_CV)
            pickle.dump(iteracao, open(path_iteracoes_cv_filename, "wb"))

            # Save
            df_iteracao_info = pd.DataFrame()
            for i in range(len(iteracao_info)):
                df_iteracao_info = df_iteracao_info.append(iteracao_info[i], ignore_index=True)
            df_iteracao_info.to_csv(path_iteracoes_cv_filename.replace(".pkl",".csv"))

    # Variable Initialization
    iteracao_EQMs_nit = pd.DataFrame(
        columns=['iteracao', 'EQM', 'nit'])  # matriz para almacenar os erros de cada iteracao

    index = -1

    # Cross-validation
    for idx_iter, (train_idx, test_idx) in enumerate(iteracao):
        # Parameter initialitation
        if save == True:
            savetxt(os.path.join(folder_output, OUT_DIRNAME, "exp%s_iter%s_test_idx.csv" % (p_modelit_id, idx_iter)), test_idx,
                    fmt="%d",
                    delimiter=',')
            savetxt(os.path.join(folder_output,OUT_DIRNAME, "exp%s_iter%s_train_idx.csv" % (p_modelit_id, idx_iter)), train_idx,
                    fmt="%d",
                    delimiter=',')

        test_caseids = X_for_model_and_Y_labels.iloc[test_idx, -1]
        train_caseids = X_for_model_and_Y_labels.iloc[train_idx, -1]

        if save == True:
            savetxt(os.path.join(folder_output, OUT_DIRNAME,"exp%s_iter%s_test_caseids.csv" % (p_modelit_id, idx_iter)),
                    test_caseids,
                    fmt="%d", delimiter=',')
            savetxt(os.path.join(folder_output, OUT_DIRNAME,"exp%s_iter%s_train_caseids.csv" % (p_modelit_id, idx_iter)),
                    train_caseids,
                    fmt="%d", delimiter=',')

        # Get data about fold j
        if (experiment['p_modelit_type'] == 'aalst_approach'):
            # For aalst_approach, Xtr e Xtst will be set in a different way than other approaches
            # for aalst_approach, Xtr  Xtsts will be strings and not arrays
            dataset_ = Dataset()
            log_view_name = log_name.replace('ohe_', '')
            dataset_log_view = pd.read_csv(os.path.join(LOGS_DIR, log_view_name))
            [Xtr, Xtst] = dataset_.create_xes_for_aalst_approach(dataset_log_view, log_view_name, train_caseids,
                                                                 test_caseids, idx_iter)
        else:
            Xtr = np.array(X_for_model[train_idx])
            Xtst = np.array(X_for_model[test_idx])

        Ytr = Xtr
        Ytr_labels = np.array(Y_labels[train_idx])  # labels of this iteration

        Ytst_labels = np.array(Y_labels[test_idx])
        Ytst = Xtst

        if ((model_type == 'autoencoder_nolle') | (model_type == 'autoencoder_keras')):
            if (validation_percentage > 0):
                # Divide training set in order to get training and test set
                Xtr, Xval, Ytr, Yval = train_test_split(Xtr, Ytr, stratify=Ytr_labels,
                                                        test_size=validation_percentage)

            elif (validation_percentage == 0): #Validation will be not executed when training the model
                Xval = 0
                Yval = 0

            dataset_val = Dataset()
            dataset_val.X_ohe = Xval
            params_training = {'dataset_val': dataset_val}
        else:
            params_training = {}

        if (model_type == 'autoencoder_nolle'):
            dataset_tr = Dataset()
            dataset_tr.X_ohe = Xtr

            dataset_tst = Dataset()
            dataset_tst.X_ohe = Xtst
            dataset_tst.Y_ohe = Ytst
            dataset_tst.attr_dim = attr_dim
            dataset_tst.case_lens = case_lens[test_idx]
            dataset_tst.max_case_len = max(case_lens)

        if ((model_type == 'tstideplus') | (model_type == 'detect_infrequents') | (model_type == 'random')):
            dataset_tr = Dataset()
            dataset_tr.X_for_tstideplus = np.dstack((Xtr, np.zeros(Xtr.shape)))
            dataset_tr.num_attributes = 2
            dataset_tr.case_lens = case_lens[train_idx]
            dataset_tr.log_of_cases = log_of_cases.iloc[list(train_idx), :]

            dataset_tst = Dataset()
            dataset_tst.X_for_tstideplus = np.dstack((Xtst, np.zeros(Xtst.shape)))
            dataset_tst.num_attributes = 2
            dataset_tst.case_lens = case_lens[test_idx]
            dataset_tst.log_of_cases = log_of_cases.iloc[list(test_idx), :]

        if (model_type == 'aalst_approach'):
            dataset_tr = Dataset()
            dataset_tr.X_for_aalst = Xtr
            dataset_tr.case_lens = case_lens[train_idx]

            dataset_tst = Dataset()
            dataset_tst.X_for_aalst = Xtst
            dataset_tst.case_lens = case_lens[test_idx]

        if (model_type == 'binet1'):
            from tensorflow.keras.utils import to_categorical
            dataset_tr = Dataset()
            dataset_tr.X_for_tstideplus = Xtr
            tmp = to_categorical(dataset_tr.X_for_tstideplus)[:, :, 1:]

            dataset_tr.Y_for_binet = np.pad(tmp[:, 1:], ((0, 0), (0, 1), (0, 0)), mode='constant')
            dataset_tr.attr_dim = attr_dim
            dataset_tr.case_lens = case_lens[train_idx]

            dataset_tst = Dataset()
            dataset_tst.X_for_tstideplus = Xtst
            tmp = to_categorical(dataset_tst.X_for_tstideplus)[:, :, 1:]
            dataset_tst.Y_for_binet = np.pad(tmp[:, 1:], ((0, 0), (0, 1), (0, 0)), mode='constant')
            dataset_tst.attr_dim = attr_dim

        if (fast_debug == False):
            # We wont create a model ( Its not possible) with detect_infrequents and cv_without_traces_duplicated_tst_tr
            import copy
            oADAtr = copy.deepcopy(oADA)
            if (not ((s_strategy_tr_tst == 'cv_without_traces_duplicated_tst_tr') & (
                    model_type == 'detect_infrequents'))) & (not (model_type == 'random')):
                # Track time
                import time
                time_tr_start = time.time()

                # Train model
                oADAtr.train_model(dataset_tr, folder_output, results, index, p_modelit_id, idx_iter, **params_training)

                # Track total time used in training model
                time_tr_end = time.time()
                time_tr=time_tr_end-time_tr_start
                results['time_tr_sec']=time_tr

            params = dict({})
            [results, index] = oADAtr.predict_using_several_thresholds(dataset_tst, folder_output,
                                                                     p_modelit_id, idx_iter,
                                                                     results, Ytst_labels, index, scaling_factors,
                                                                     method_for_explore_thresholds,
                                                                     exp_group_name, params,test_caseids)


        else:
            print("nothing")

        plt.close()
    return results


def show_some_statistics(log_of_cases_without_repetition, log_of_cases):
    # Some statistics
    print("Distribution of anomalous and normal in log_of_cases_without_repetition")
    import plotly.express as px
    fig = px.histogram(log_of_cases_without_repetition, x="label")
    fig.show()

    dict_1 = {'number_of_cases': log_of_cases_without_repetition['label'].count(),
              'number_of_traces': log_of_cases_without_repetition['label'].count(),
              'number_of_activities': 'nao muda',
              'number_of_anomalous_cases':
                  log_of_cases_without_repetition[log_of_cases_without_repetition["label"] == 'anomaly'][
                      'label'].count(),
              'number_of_normal_cases':
                  log_of_cases_without_repetition[log_of_cases_without_repetition["label"] == 'normal']['label'].count()
              }
    print(dict_1)
    log_of_cases_without_repetition['label'].describe()

    log_of_cases_without_repetition2 = log_of_cases.groupby(by=["trace"])["traceid"].count().reset_index()
    log_of_cases["traceid"] = log_of_cases["traceid"].astype(str)

    log_of_cases_without_repetition2 = log_of_cases_without_repetition2.rename(columns={"traceid": "nr_cases"})
    dictionary_replace = dict(
        zip(log_of_cases_without_repetition2["trace"], log_of_cases_without_repetition2["nr_cases"]))

    log_of_cases_without_repetition["nr_cases"] = log_of_cases_without_repetition["trace"].map(dictionary_replace)

    print("Normal traces ordered by frequency. Frequency is defined by the number of cases that execute one trace")
    print(log_of_cases_without_repetition[log_of_cases_without_repetition["label"] == 'normal'][
              ['label', 'nr_cases', 'traceid']].sort_values(by=['nr_cases'], ascending=False))

    print("Anomalous traces ordered by frequency")
    print(log_of_cases_without_repetition[log_of_cases_without_repetition["label"] == 'anomaly'][
              ['label', 'nr_cases', 'traceid']].sort_values(by=['nr_cases'], ascending=False))


def get_idx_for_training_and_test_dataset(X_for_model_and_Y_labels, X_for_model, Y_labels, input_rep_name, tipo_experimento, k,
                                          s_strategy_tr_tst, test_percentage, oADA):
    """
    This functions returns 'iteracao', a list of tuples (train_idx,test_idx) containing the idx for each iteration or fold
    """

    iteracao_info=[]

    # Strategy (s_strategy_tr_tst) can be one of the following:
    if ("train_with_all_without_repetitions" in s_strategy_tr_tst):
        # This startegy train using all data, tst with all data. Just one time
        if k == 1:
            all_idx = np.array(range(0, len(X_for_model)))
            iteracao = [(all_idx, all_idx)]  # Lista com apenas 1 elemento. Esse elemento e uma tupla
        else:
            raise Exception('k_cv should be 1')

    elif ("train_with_all_with_repetitions" in s_strategy_tr_tst):
        # This startegy train using all data, tst with all data. k times
        if (k > 1):
            # Train and test with all dataset.Repite that process [k] times
            all_idx = np.array(range(0, len(X_for_model)))
            iteracao = []
            for i in range(k):
                iteracao.append((all_idx, all_idx))
        else:
            raise Exception('k_cv should be > 1')
    elif (s_strategy_tr_tst == 'cv_cases'):
        # This strategy divides set in training and test set
        # Folds contain stratified set for training and test
        # Problem: Cases with the same trace could be in several folds. So, it is possible to test the model using the same trace that was used for training.
        iteracao = list(StratifiedKFold(n_splits=k, shuffle=True).split(X_for_model, Y_labels))

    elif (s_strategy_tr_tst == 'cv_without_traces_duplicated_tst_tr'):
        # This strategy solves the problem happening in the previous strategy

        # Get dataset of training without repetitions
        log_of_cases = pd.read_csv(oADA.train_log.log_of_cases_path)
        log_of_cases = log_of_cases.rename(columns={'traceid': 'caseid'})
        log_of_cases_without_repetition = log_of_cases.drop_duplicates(subset='trace', keep='first')

        # traces_and_frequency is a dataframe that saves nr_occurrences by each trace.
        traces_and_frequency = log_of_cases.groupby(by=['trace'])['caseid'].count().sort_values(
            ascending=False).reset_index()
        traces_and_frequency = traces_and_frequency.rename(columns={'caseid': 'nr_ocurrences'})

        # Insert nr_ocurrences in log_of_cases
        list1 = list(traces_and_frequency['trace'])  # old
        list2 = list(traces_and_frequency['nr_ocurrences'])  # new
        dictionary_replace = dict(zip(list1, list2))
        log_of_cases_without_repetition['nr_ocurrences'] = log_of_cases_without_repetition['trace'].map(
            dictionary_replace)

        # Initialize dicts of folds
        from collections import defaultdict
        folds_traces = defaultdict(dict)
        nfolds = k #number of folds
        for i in range(nfolds):
            folds_traces[i]['traces'] = []
            folds_traces[i]['nr_ocurrences'] = []
        sum_cases_in_fold1 = pd.Series([0] * nfolds)

        # First: normal, order by nr_ocurrences
        # First: anomaly, order by nr_ocurrences
        log_of_cases_without_repetition = log_of_cases_without_repetition.sort_values(by=['label','nr_ocurrences'],
                                                                                      ascending=False)

        # Create folder if does not exist
        if not (os.path.isdir(LOG_ITERACOES_CV)):
            os.makedirs(LOG_ITERACOES_CV)

        # Save
        log_of_cases_without_repetition.to_csv(LOG_ITERACOES_CV/ ("method_table_"+input_rep_name.replace(".pkl",".csv").replace("ohe_","").replace("tstideplus_","")))

        for label in ['normal','anomaly']:
            log_of_cases_without_repetition_ = log_of_cases_without_repetition[
                log_of_cases_without_repetition['label'] == label]

            index = 0
            # While exists cases without analysis
            while index < len(log_of_cases_without_repetition_):
                # Find next fold to fill. We have to fill all 5. order they by nr_ocurrences
                # Order fold by number of cases in each (less to more)
                sum_cases_in_fold1 = sum_cases_in_fold1.sort_values()

                # We are going to send the most frequent traces to the fold which has the lower number of cases in it
                for idx_fold, item in sum_cases_in_fold1.items():
                    if (index < len(log_of_cases_without_repetition_)):
                        row = log_of_cases_without_repetition_.iloc[index]

                        # Fill each fold. It starts allocating traces in folds that have the lower number of cases.
                        folds_traces[idx_fold]['traces'].append(row['trace'])
                        folds_traces[idx_fold]['nr_ocurrences'].append(row['nr_ocurrences'])
                        sum_cases_in_fold1[idx_fold] = sum_cases_in_fold1[idx_fold] + row['nr_ocurrences']
                        index = index + 1
        # Save
        pickle.dump(folds_traces, open(LOG_ITERACOES_CV / ("traces_in_folds_"+input_rep_name.replace(".csv",".pkl").replace("ohe_","").replace("tstideplus_","")), "wb"))

        # Fill each fold using cases
        folds_cases = {}
        iteracao = []
        for idx_fold in range(nfolds):
            fold_traces = folds_traces[idx_fold]['traces']  # list of traces in fold
            log_of_cases_fold = log_of_cases[log_of_cases['trace'].isin(fold_traces)].sort_values(by=['trace'])
            folds_cases[idx_fold] = log_of_cases_fold

            # Assign cases in the fold idx_fold1 to test_idx
            test_idx = X_for_model_and_Y_labels[
                X_for_model_and_Y_labels.iloc[:, -1].isin(log_of_cases_fold['caseid'])].index

            # Assign rest of cases  to train_idx
            train_idx = X_for_model_and_Y_labels[
                ~X_for_model_and_Y_labels.iloc[:, -1].isin(log_of_cases_fold['caseid'])].index
            iteracao.append((train_idx, test_idx))

        # Show some statistics
        iteracao_info={}
        for idx_fold in range(nfolds):
            folds_cases[idx_fold]['label'].describe()
            iteracao_info[idx_fold]= {
                'n_normal_cases': len(folds_cases[idx_fold][folds_cases[idx_fold]['label'] == 'normal']),
                'n_anomalous_cases': len(folds_cases[idx_fold][folds_cases[idx_fold]['label'] == 'anomaly']),
                'n_traces': len(
                    folds_cases[idx_fold].drop_duplicates(
                        subset='trace', keep='first')),
                'n_cases': len(folds_cases[idx_fold]),
                'n_normal_traces': len(
                    folds_cases[idx_fold][folds_cases[idx_fold]['label'] == 'normal'].drop_duplicates(
                        subset='trace', keep='first')),
                'n_anomalous_traces': len(
                    folds_cases[idx_fold][folds_cases[idx_fold]['label'] == 'anomaly'].drop_duplicates(
                        subset='trace', keep='first')),
            }

    return [iteracao,iteracao_info]