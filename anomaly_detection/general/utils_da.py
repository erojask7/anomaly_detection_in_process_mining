#FIX2022forpubrep: ready for publication
import sys

sys.path.insert(1, "E:\\EXPERIMENTOS2020\\anomaly_detection_artigo\\")
from anomaly_detection.general.global_variables import *


def get_list_of_activities_in_process(control_flow, process_name):
    activities_list = []
    if process_name == "ptall":
        if control_flow == "AND":
            activities_list = ["N", "O", "P", "Q"]

        if control_flow == "AND_extended":
            activities_list = ["N", "O", "P", "Q", "M", "R"]

        if control_flow == "XOR":
            activities_list = ["G", "H", "I", "J", "K", "L"]

        if control_flow == "XOR_extended":
            activities_list = ["G", "H", "I", "J", "K", "L", "F", "M"]

        if control_flow == "group1":
            activities_list = ["F", "G", "H", "I", "J", "K", "L"]

        if control_flow == "group2":
            activities_list = ["M", "N", "O", "P", "Q", "R", "S", "T", "U"]
    return activities_list


def create_loss_figure(folder_output, history, p_modelit_id, j):  # Create figure of EQM
    import matplotlib.pyplot as plt
    import os
    try:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(os.path.join(folder_output, OUT_DIRNAME, "exp%s_iter%s_evolucao_lossfunc.png" % (p_modelit_id, j)),
                    bbox_inches='tight')
        # plt.show()
        # plt.close()
    except KeyError:  # val_loss not found
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig(os.path.join(folder_output, OUT_DIRNAME, "exp%s_iter%s_evolucao_lossfunc.png" % (p_modelit_id, j)),
                    bbox_inches='tight')
        # plt.show()
        # plt.close()


def create_log_of_cases(log_filename, logs_path, output_path="", save=True):
    """
    This funtion creates a log of cases.
    """
    import pandas as pd
    log = pd.read_csv(logs_path / log_filename)

    # Create set of cases
    set_cases = log.groupby('traceid')['activity'].apply(lambda x: ' ,'.join(x))

    if "label" in log.columns:
        labels = log.groupby('traceid')['label'].first()

    set_cases = set_cases.rename("trace")
    set_cases = pd.DataFrame(set_cases)
    set_cases['nr_events'] = list(log.groupby(['traceid'])['activity'].count())

    if "label" in log.columns:
        set_cases['label'] = labels

    if ("type_anomaly" in log.columns):
        anomaly_type = log.groupby('traceid')['type_anomaly'].first()
        set_cases['type_anomaly'] = anomaly_type

    set_cases = set_cases.reset_index()
    if (save == True):
        logofcases_filename = log_filename.replace("log", "logofcases")
        set_cases.to_csv(output_path / logofcases_filename, sep=',', encoding='utf-8', index=False)
    return set_cases


def create_traces_and_frequency(log_of_cases, column_name_trace, column_name_to_count, log_name, path_output,
                                save=True):
    """
    This function creates traces_and_frequency. Traces_and_frequency is a dataframe that
    saves number of ocurrences by ech trace.

    Returns
    -------
    traces_and_frequency

    """
    import os
    import pandas as pd

    traces_and_frequency = pd.DataFrame(
        log_of_cases.groupby(by=[column_name_trace])[column_name_to_count].count().sort_values(
            ascending=False))
    traces_and_frequency = traces_and_frequency.rename(columns={column_name_to_count: 'nr_ocurrences'})

    if ("label" in log_of_cases.columns):
        type_anomaly = log_of_cases.groupby(by=[column_name_trace])["label"].first()
        traces_and_frequency['label'] = type_anomaly

    traces_and_frequency = traces_and_frequency.reset_index()

    if save == True:
        traces_and_frequency.to_csv(os.path.join(path_output, "traces_and_frequency_%s" % log_name), sep=',',
                                    encoding='utf-8',
                                    index=False)
    return traces_and_frequency


def create_informative_csvs_about_logs(logs_path, filenames, types):
    """
    This function create log_of_cases format and traces_and_frequency format of logs.
    We consider three formats for event logs:

    a. classic event log: each line of the log contains an event and their properties:
    traceid,label,activity

    b. log_of_cases: ach line represent an case and thei properties:
    traceid, trace ,nr_events, label

    c. traces_and_frequency: is a format that saves number of ocurrences by each trace.
    """
    from anomaly_detection.general import utils_da
    for log_filename in filenames:
        # Criar log_of_cases para cada log
        if "logofcases" in types:
            set_of_cases = utils_da.create_log_of_cases(log_filename, logs_path, logs_path)
        if "traces_and_frequency" in types:
            traces_and_frequency = utils_da.create_traces_and_frequency(set_of_cases, "trace", "traceid", log_filename,
                                                                        logs_path)


def get_abreviation_of(full_name_approach):
    abbrev = ''
    if (full_name_approach == 'autoencoder_nolle'):
        abbrev = 'daen'
    elif (full_name_approach == 'tstideplus'):
        abbrev = 'stp'
    elif (full_name_approach == 'aalst_approach'):
        abbrev = 'aalst'
    elif (full_name_approach == 'binet1'):
        abbrev = 'binetn'
    elif (full_name_approach == 'detect_infrequents'):
        abbrev = 'infreq'
    elif (full_name_approach == 'random'):
        abbrev = 'rndm'
    elif (full_name_approach == 'p_modelit_windows_size'):
        abbrev = 'ws'
    return abbrev


def get_pretty_names_for_params(param, language='PT'):
    abbrev = param
    if (param == 'p_modelit_windows_size'):
        abbrev = 'Windows Size' if language == 'EN' else 'Tamanho de janela'
    if (param == 'exp_auc_pr'):
        abbrev = 'AP' if language == 'EN' else 'AP'
    if (param == 'exp_auc_roc'):
        abbrev = 'AUC ROC' if language == 'EN' else 'Area baixo a curva ROC'
    if (param == 'p_modelit_thd'):
        abbrev = 'Thd'
    if (param == 'p_modelit_thdd'):
        abbrev = 'Thd'
    if (param == 'p_modelit_kk'):
        abbrev = 'K'
    if (param == 'log_name_abbrev'):
        abbrev = 'Log'
    if (param == 'actv_functions'):
        abbrev = 'Activation functions' if language == 'EN' else 'Funções de ativação'
    if (param == 'alfa'):
        abbrev = 'Learning Rate' if language == 'EN' else 'Taxa de aprendizado'
    if (param == 'log_anomaly_intuition'):
        abbrev = 'Anomaly intuition' if language == 'EN' else 'Anomalia'
    if (param == 'p_modelit_type'):
        abbrev = 'Approach' if language == 'EN' else 'Abordagem'
    if (param == 'p_modelit_vapproach'):
        abbrev = 'Type' if language == 'EN' else 'Variação'
    return abbrev


def get_specific_params(full_name_approach):
    if (full_name_approach == 'autoencoder_nolle'):
        output = ['alfa', 'funcao_f', 'funcao_g']
    elif (full_name_approach == 'tstideplus'):
        output = ['p_modelit_windows_size', 'p_modelit_thd']
    elif (full_name_approach == 'aalst_approach'):
        output = ['p_modelit_kk', 'p_modelit_thdd', 'p_modelit_vapproach']
    elif (full_name_approach == 'binet1'):
        output = ['alfa', 'nitmax']
    elif (full_name_approach == 'detect_infrequents') | (full_name_approach == 'random'):
        output = []
    return output


def get_params_all_results(approach_name):

    # Commom parameters for all approaches
    exp_group_params = ['exp_group_id', 'exp_group_name', 'p_modelit_type', 's_strategy_tr_tst', 'scaling_factors',
                        's_k_cv']

    dataset_params = ['log_bimp', 'log_control_flow_type', 'log_anomaly_intuition', 'log_name', 'log_name_abbrev']
    approach_params = ['p_modelit_id', 'p_modelit_vapproach', 'p_modelit_voutlierness_calc', 'use_mask']
    anomaly_detection_model_params = ['s_tipo_heuristica', 'scaling_factor', 'threshold_value',
                                      's_method_for_explore_thresholds']
    # Parameters for avaliation
    avaliation_params = ['iteracao_cv', 's_avaliation_gran_level', 's_avaliation_type', 'TP', 'FN', 'FP', 'TN', 'PP',
                         'PN',
                         'precision_p', 'recall_p', 'fscore_p', 'precision_n', 'recall_n', 'fscore_n',
                         'precision_macro', 'recall_macro', 'fscore_macro',
                         'time_tr_sec', 'time_tst_sec', 'time_tst_all_sf',
                         'exp_auc_pr', 'exp_auc_roc',
                         'exp_auc_pr_thresholds', 'exp_auc_pr_precisions', 'exp_auc_pr_recalls']

    # Specific parameters of approach setting
    if approach_name == 'tstideplus':
        specific_approach_params_input = ['p_modelit_windows_size', 'p_modelit_thd', 'p_modelit_thd_value']

    elif (approach_name == 'autoencoder_nolle'):
        specific_approach_params_input = ['batch_size', 'alfa', 'optimizer_beta2',
                                          'nitmax', 'funcao_g', 'ne', 'no',
                                          'gaussian_noise_std', 'funcao_f', 'early_stopping_metric',
                                          'early_stopping_patiente', 'dropout',
                                          'nro_camadas_ocultas']

    elif approach_name == 'aalst_approach':
        specific_approach_params_input = ['p_modelit_kk', 'p_modelit_thdd']

    elif approach_name == 'binet1':
        specific_approach_params_input = ['batch_size', 'alfa', 'optimizer_beta2',
                                          'nitmax',
                                          'gaussian_noise_std',
                                          'early_stopping_patiente',
                                          'decoder'
                                          ]
    else:
        specific_approach_params_input = []

    # Specific avaliation params of approach (during creation of model)
    if (approach_name == 'autoencoder_nolle') | (approach_name == 'binet1'):
        specific_avaliation_params = ['nit', 'train_loss_last', 'val_loss_last']
        if approach_name == 'binet1':
            specific_avaliation_params.append('n_gru_units')
    else:
        specific_avaliation_params = []

    # Join list of parameters in order
    commom_params = exp_group_params + dataset_params + approach_params + anomaly_detection_model_params + avaliation_params
    specific_params = specific_avaliation_params + specific_approach_params_input

    params = commom_params + specific_params
    return params


def join_all_results_files(folder_output, approach_name, all_results_name):
    import pandas as pd
    import os
    params_ordered_list = get_params_all_results(approach_name)

    # Concat all results files
    results_files = [f for f in os.listdir(folder_output) if f.endswith(ALL_RESULTS_DETAILS_END_STRING)]
    all_results_details = pd.concat([pd.read_csv(os.path.join(folder_output, f), sep=',') for f in results_files])

    # Order columns
    all_results_details = all_results_details[params_ordered_list]

    # Save file
    all_results_details.to_csv(os.path.join(folder_output, all_results_name), sep=',', encoding='utf-8',
                               index=False)


def filter_dataframes(df, conditions):
    """
    Filter dataframe according to some conditions.

    Parameters
    ----------
    df: dataFrame
        Dataframe that will b filtered using conditions
    conditions: dict
        Conditions will be used for filtering the dataframe.
        It is a dictionary where key is the dataframe column name and val is the possible list of values
        Example:{'funcao_f':'relu','funcao_g':['relu','sigmoid']}

    Returns
    -------
    dataFrame
        the returned dataFrame only contain rows that passes the stablished conditions

    """

    cond = None
    for key, val in conditions.items():
        if cond is None:
            # If we specified a list of values we will use isin
            if (type(val) is list):
                cond = df[key].isin(val)
            # If not we will use ==
            else:
                cond = (df[key] == val)
        else:
            if (type(val) is list):
                cond = cond & (df[key].isin(val))
            else:
                cond = cond & (df[key] == val)
    return [cond, df[cond]]


def prettyfy_dataframe(df):
    if "p_modelit_type" in df.columns:
        df["p_modelit_type"] = df["p_modelit_type"].map({'detect_infrequents': 'ABRI',
                                                         'aalst_approach': 'ABCS',
                                                         'random': 'Randômica',
                                                         'tstideplus': 't-STIDE+',
                                                         'autoencoder_nolle': 'DAE',
                                                         'binet1': 'Binet'
                                                         })
    if "log_anomaly_intuition" in df.columns:
        df["log_anomaly_intuition"] = df["log_anomaly_intuition"].str.upper()

    return df


def prettyfy_dataframe2(df):
    if "p_modelit_type" in df.columns:
        df["p_modelit_type"] = df["p_modelit_type"].map({'detect_infrequents': 'ABRI',
                                                         'aalst_approach': 'ABCS',
                                                         'random': 'Randômica',
                                                         'tstideplus': 't-STIDE+',
                                                         'autoencoder_nolle': 'DAE',
                                                         'binet1': 'Binet'
                                                         })
    if "log_anomaly_intuition" in df.columns:
        df["log_anomaly_intuition"] = df["log_anomaly_intuition"].str.upper()

    if "pasa_loop" in df.columns:
        df["pasa_loop"] = df["pasa_loop"].map({True: "Loop",
                                               False: "Não Loop",
                                               })

    if "anomaly_creates_duplicates" in df.columns:
        df["anomaly_creates_duplicates"] = df["anomaly_creates_duplicates"].map({True: "Duplicatas",
                                                                                 False: "Não Duplicatas",
                                                                                 })
    # if "log_name_abbrev" in df.columns:
    #    df["log_name_abbrev"]=df["log_name_abbrev"].str.upper()
    return df


def create_dir_if_doesnt_exist(full_path):
    import os
    if not (os.path.isdir(full_path)):  # If folder doesnt exist
        os.mkdir(full_path)  # Creat folder


def execute_all_histogram_tests_in_df(dfg, facet_col, iplot, alpha=0.05):
    import pandas as pd
    # 3.1 Print Histograms
    histograms(dfg, facet_col, iplot)

    # 3.2 Normality Test
    # Initialize DataFrame where results of normality tests will be saved
    df_rslt_normality = pd.DataFrame(
        columns=['test_name', 'param', 'alpha', 'p_value', 'test_statistic', 'fail_to_reject_H0', 'reject_H0'])

    for column in dfg.columns:
        dict_rslt = execute_normality_test(dfg[[column]], alpha)
        df_rslt_normality = df_rslt_normality.append(dict_rslt, ignore_index=True)
    return df_rslt_normality


def execute_all_median_tests_in_df(dfg, limiar_exp_auc_pr, parametric=False):
    import pandas as pd
    from itertools import combinations

    # Initialize DataFrame where results of tests of medians will be saved
    df_rslt_tests = pd.DataFrame(
        columns=['test_name', 'limiar_AP', 'params_values', 'params_nr', 'alpha', 'p_value', 'fail_to_reject_H0',
                 'reject_H0'])

    for nr_approaches in range(len(dfg.columns), 1, -1):
        indexes_columns_list = combinations(range(0, len(dfg.columns)), nr_approaches)
        for indexes_columns in indexes_columns_list:
            indexes_columns = list(indexes_columns)

            # Create filtered dataframe
            df = dfg.iloc[:, indexes_columns]

            # Execute test
            if parametric:
                dict_rslt = execute_anova_test(df)
            else:
                dict_rslt = execute_friedman_test(df)
            df_rslt_tests = df_rslt_tests.append(dict_rslt, ignore_index=True)
        df_rslt_tests['limiar_AP'] = limiar_exp_auc_pr

    return df_rslt_tests


def execute_normality_test(df, alpha):
    from scipy.stats import shapiro
    stat, p_value = shapiro(df.iloc[:, 0].values)

    # print('Statistics=%.3f, p=%.3f' % (stat, p_value))

    # interpret
    alpha = 0.05
    if p_value > alpha:
        # print('Sample looks Gaussian (fail to reject H0)')
        fail_to_reject_H0 = True
    else:
        # print('Sample does not look Gaussian (reject H0)')
        fail_to_reject_H0 = False

    dict_rslt = {
        'test_name': 'Shapiro-Wilk',
        'param': df.columns[0],
        'p_value': p_value,
        'alpha': alpha,
        'test_statistic': stat,
        'fail_to_reject_H0': fail_to_reject_H0,  # same_distributions
        'reject_H0': not (fail_to_reject_H0)  # different distributions
    }
    return dict_rslt


def execute_anova_test(df):
    from statsmodels.stats.anova import AnovaRM
    import pandas as pd

    df1 = pd.DataFrame(df.unstack()).reset_index()
    df1 = df1.rename(columns={0: "AP"})
    var_blocks = "p_modelit_type"
    var_treatmeant = "log_name_abbrev"
    var_value = "AP"

    rslts = AnovaRM(data=df1, depvar=var_value, subject=var_blocks, within=[var_treatmeant]).fit().anova_table
    chi_squared = rslts["F Value"].values[0]
    p_value = rslts["Pr > F"].values[0]
    alpha = 0.05
    if p_value > alpha:
        fail_to_reject_H0 = True
    #     print('Same distributions (fail to reject H0)')
    else:
        fail_to_reject_H0 = False

    dict_rslt = {
        'test_name': 'AnovaRM',
        'params_values': list(df.columns),
        'params_nr': len(list(df.columns)),
        'p_value': p_value,
        'alpha': alpha,
        'chi_squared': chi_squared,
        'fail_to_reject_H0': fail_to_reject_H0,  # same_distributions
        'reject_H0': not (fail_to_reject_H0)  # different distributions
    }
    return dict_rslt


def execute_friedman_test(df):
    import os
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    import rpy2.robjects as robjects
    from rpy2.robjects.conversion import localconverter
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_from_pd_df = robjects.conversion.py2rpy(df)
    os.environ['R_HOME'] = "C:/Program Files/R/R-4.1.0/"
    base = importr('base')  # origem do erro UnicodeDecodeError

    r = robjects.r

    matrix1 = base.as_matrix(r_from_pd_df)
    friedman_test = r['friedman.test'](matrix1)
    chi_squared = friedman_test[0]
    p_value = friedman_test[2]
    alpha = 0.05
    if p_value[0] > alpha:
        fail_to_reject_H0 = True
    #     print('Same distributions (fail to reject H0)')
    else:
        fail_to_reject_H0 = False
    #     print('Different distributions (reject H0)')

    dict_rslt = {
        'test_name': 'Friedman_R',
        'params_values': list(df.columns),
        'params_nr': len(list(df.columns)),
        'p_value': p_value[0],
        'alpha': alpha,
        'chi_squared': chi_squared[0],
        'fail_to_reject_H0': fail_to_reject_H0,  # same_distributions
        'reject_H0': not (fail_to_reject_H0)  # different distributions
    }
    consolewrite_warnerror = None
    return dict_rslt


def histograms(df, facet_col, iplot=False):
    import pandas as pd
    # all columns will be included as histograms
    # df.unstack()
    df1 = pd.DataFrame(df.unstack()).reset_index()
    df1 = df1.rename(columns={0: "AP"})
    import plotly.express as px
    import plotly.offline as py
    fig = px.histogram(df1,
                       x="AP",
                       title="Histogram",
                       facet_col=facet_col,
                       facet_col_wrap=2,
                       )

    if (iplot == True):
        py.iplot(fig)
    else:
        py.plot(fig)
