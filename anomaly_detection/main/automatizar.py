# %matplotlib inline #FIX2022forpubrep: ready for publication
import sys
from pathlib import Path

ROOT_DIR = Path("E:\\EXPERIMENTOS2020\\anomaly_detection\\")
sys.path.insert(1, str(ROOT_DIR))
from anomaly_detection.general.utils_da import get_abreviation_of
from anomaly_detection.main.tasks import *
from anomaly_detection.main.get_experiment_parameterization import *


def get_experiments_parameterization(nome_grupo_experimentos, main_path, phase_number, dataset_values,
                                     bimp_log_filename,
                                     approach_name, enum_p_modelit_id_from=1, start_in_p_modelit_id=1):
    ################### General Parameters #################
    import time
    exp_group_id = 'gexp_' + get_abreviation_of(approach_name) + '_' + str(int(round(time.time() - 1584060000, 3) * 1000))  # identificador usando o tempo

    ################### Dataset Parameters #################

    # Set Datasets
    dataset_values = set(dataset_values)

    # Set Scaling Factors
    if (nome_grupo_experimentos == "relatorio1-question1"):
        min_sf = 0
        max_sf = 1
        sf_number_of_values = 5
    elif (nome_grupo_experimentos == "relatorio1-question2"):
        min_sf = 0
        max_sf = 1
        sf_number_of_values = 11
    elif (nome_grupo_experimentos == "test_case"):
        min_sf = 0
        max_sf = 1
        sf_number_of_values = 2

    if approach_name == 'autoencoder_nolle':
        params_list = ['funcao_f_values', 'funcao_g_values', 'alfa_min', 'alfa_max', 'alfa_number_of_values',
                       'vapproach_values', 'voutlierness_calc_values']

        if phase_number == 1:
            if nome_grupo_experimentos == "relatorio1-question1":
                # Set Activation Functions
                funcao_f_values = {'sigmoid', 'relu', 'tanh', 'softmax'}
                funcao_g_values = {'sigmoid', 'softmax'}

                # Set Learning rate
                alfa_min = 0.001
                alfa_max = 1
                alfa_number_of_values = 5

                vapproach_values = {'original'}
                voutlierness_calc_values = {'original'}

            if nome_grupo_experimentos == "test_case":
                # Set Activation Functions
                funcao_f_values = {'sigmoid'}
                funcao_g_values = {'sigmoid', 'softmax'}

                # Set Learning rate
                alfa_min = 0.001
                alfa_max = 1
                alfa_number_of_values = 2

                vapproach_values = {'original'}
                voutlierness_calc_values = {'original'}

        if phase_number == 2:
            if nome_grupo_experimentos == "relatorio1-question2":  # best parameters
                funcao_f_values = {'relu'}
                funcao_g_values = {'sigmoid'}

                # Set Learning rate
                alfa_min = 0.001
                alfa_max = 0.001
                alfa_number_of_values = 1

                vapproach_values = {'original'}
                voutlierness_calc_values = {'original'}

            if nome_grupo_experimentos == "test_case":  # best parameters
                funcao_f_values = {'relu'}
                funcao_g_values = {'sigmoid'}

                # Set Learning rate
                alfa_min = 0.001
                alfa_max = 0.001
                alfa_number_of_values = 1

                vapproach_values = {'original'}
                voutlierness_calc_values = {'original'}

    elif approach_name == 'binet1':
        params_list = ['nit_min', 'nit_max', 'nit_number_of_values', 'alfa_min', 'alfa_max', 'alfa_number_of_values',
                       'vapproach_values', 'voutlierness_calc_values']
        if phase_number == 1:
            if nome_grupo_experimentos == "relatorio1-question1":
                # Set Learning rate
                nit_min = 20
                nit_max = 200
                nit_number_of_values = 10

                # Set Learning rate
                alfa_min = 0.001
                alfa_max = 1
                alfa_number_of_values = 3

                vapproach_values = {'original'}
                voutlierness_calc_values = {'original'}

            if nome_grupo_experimentos == 'test_case':
                params_list = ['nit_min', 'nit_max', 'nit_number_of_values', 'alfa_min', 'alfa_max',
                               'alfa_number_of_values','vapproach_values', 'voutlierness_calc_values']
                # Set Learning rate
                nit_min = 20
                nit_max = 40
                nit_number_of_values = 2

                # Set Learning rate
                alfa_min = 0.001
                alfa_max = 1
                alfa_number_of_values = 2

                vapproach_values = {'original'}
                voutlierness_calc_values = {'original'}

        if phase_number == 2:
            if nome_grupo_experimentos == "relatorio1-question2":
                # Set Learning rate
                nit_min = 100
                nit_max = 100
                nit_number_of_values = 1

                # Set Learning rate
                alfa_min = 0.001
                alfa_max = 0.001
                alfa_number_of_values = 1

                vapproach_values = {'original'}
                voutlierness_calc_values = {'original'}

            if nome_grupo_experimentos == 'test_case':
                params_list = ['nit_min', 'nit_max', 'nit_number_of_values', 'alfa_min', 'alfa_max',
                               'alfa_number_of_values','vapproach_values', 'voutlierness_calc_values']
                # Set Learning rate
                nit_min = 2
                nit_max = 2
                nit_number_of_values = 1

                # Set Learning rate
                alfa_min = 0.001
                alfa_max = 0.001
                alfa_number_of_values = 1

                vapproach_values = {'original'}
                voutlierness_calc_values = {'original'}

        if approach_name == 'aalst_approach':
            params_list = ['thdd_min', 'thdd_max', 'thdd_number_of_values', 'min_kk', 'max_kk', 'vapproach_values',
                       'voutlierness_calc_values']
        if phase_number == 1:
            if nome_grupo_experimentos == "relatorio1-question1":
                thdd_min = 0
                thdd_max = 0.4
                thdd_number_of_values = 11

                min_kk = 1
                max_kk = 4
                vapproach_values = {'original', 'original_mod'}
                voutlierness_calc_values = {'using_mismatches'}

            if nome_grupo_experimentos == "test_case":
                thdd_min = 0
                thdd_max = 0.4
                thdd_number_of_values = 3

                min_kk = 1
                max_kk = 2
                vapproach_values = {'original', 'original_mod'}
                voutlierness_calc_values = {'using_mismatches'}

        if phase_number == 2:
            if nome_grupo_experimentos == "relatorio1-question2":
                thdd_min = 0.04
                thdd_max = 0.04
                thdd_number_of_values = 1

                min_kk = 1
                max_kk = 1
                vapproach_values = {'original_mod'}
                voutlierness_calc_values = {'using_mismatches'}

    elif approach_name == 'tstideplus':
        params_list = ['thd_min', 'thd_max', 'thd_number_of_values', 'min_windows_size', 'max_windows_size',
                       'vapproach_values', 'voutlierness_calc_values']
        if phase_number == 1:
            if nome_grupo_experimentos == "relatorio1-question1":
                thd_min = 0
                thd_max = 0.01
                thd_number_of_values = 11

                min_windows_size = 2  # should be higher than one
                max_windows_size = 10
                vapproach_values = {'original'}
                voutlierness_calc_values = {'using_mismatches'}

            if nome_grupo_experimentos == 'test_case':
                thd_min = 0
                thd_max = 0.01
                thd_number_of_values = 3

                min_windows_size = 2  # should be higher than one
                max_windows_size = 3
                vapproach_values = {'original'}
                voutlierness_calc_values = {'using_mismatches'}

        if phase_number == 2:

            if nome_grupo_experimentos == "relatorio1-question2":
                thd_min = 0.003
                thd_max = 0.003
                thd_number_of_values = 1

                min_windows_size = 3  # should be higher than one
                max_windows_size = 3
                vapproach_values = {'original'}
                voutlierness_calc_values = {'using_mismatches'}

            if nome_grupo_experimentos == 'test_case':
                thd_min = 0.001
                thd_max = 0.001
                thd_number_of_values = 1

                min_windows_size = 2  # should be higher than one
                max_windows_size = 2
                vapproach_values = {'original'}
                voutlierness_calc_values = {'using_mismatches'}

    elif approach_name == 'detect_infrequents':
        params_list = ['vapproach_values',
                       'voutlierness_calc_values']

        vapproach_values = {'original'}
        voutlierness_calc_values = {'original'}

    elif approach_name == 'random':
        params_list = ['vapproach_values',
                       'voutlierness_calc_values']

        vapproach_values = {'original'}
        voutlierness_calc_values = {'original'}

    params = {}
    for variable in params_list:
        params[variable] = eval(variable)

    # Create combination of parameteres values
    [number_of_models_it, d_dynamic_parameters_model] = get_combinated_dynamic_parameters(dataset_values, min_sf,
                                                                                          max_sf, sf_number_of_values,
                                                                                          approach_name, params,
                                                                                          enum_p_modelit_id_from)
    dataset_tipo_processo = []
    dataset_intuicao_anomalias = []
    for log_name in d_dynamic_parameters_model['log_name']:
        process = log_name[log_name.find("_pt") + 1:].split("_")[0].replace("pt", "")
        anomaly_type = log_name[log_name.find("_at") + 1:].split("_")[0].replace("at", "")

        dataset_tipo_processo.append(process)
        dataset_intuicao_anomalias.append(anomaly_type)

    # Set parameters for all approaches
    d_general_fixed_parameters = {
        # Dataset-related parameters
        'log_bimp': bimp_log_filename,
        'log_anomaly_intuition': dataset_intuicao_anomalias,
        'log_control_flow_type': dataset_tipo_processo,

        # Experiments setting (Principal)
        's_k_cv': 5,
        's_tipo_heuristica': None,
        's_method_for_explore_thresholds': 'percentiles_unique',  # 'scaling_factor_x_mean','percentiles_unique'
        's_strategy_tr_tst': 'train_with_all_with_repetitions' if nome_grupo_experimentos == 'reproduce_binet1_p2p' else 'cv_without_traces_duplicated_tst_tr',

        # Experiments setting (Secondary)
        's_avaliation_type': 'q_classification',  # q_model or q_classification
        's_avaliation_gran_level': 'anomalous_cases',

        # Experiments group parameters
        'exp_group_name': nome_grupo_experimentos,  # experiment group description
        'exp_group_id': exp_group_id,  # experiment group ID

        # Mask
        'use_mask': 'default'
    }
    d_fixed_parameters_model = get_fixed_parameterization_model(approach_name, nome_grupo_experimentos)
    d_derived_parameters = get_derived_parameters_set_automatically(approach_name, d_general_fixed_parameters)

    dict_parameterization = {**d_general_fixed_parameters, **d_dynamic_parameters_model, **d_fixed_parameters_model,
                             **d_derived_parameters}

    experiments_parameterization = pd.DataFrame.from_dict(dict_parameterization)
    experiments_parameterization["log_name_abbrev"] = experiments_parameterization["log_name"].apply(
        lambda x: x[x.rfind('_at') + 3:-4].replace("_", ""))

    if (start_in_p_modelit_id > 1):
        experiments_parameterization = experiments_parameterization[
            (experiments_parameterization['p_modelit_id'] >= start_in_p_modelit_id)].reset_index(drop=True)

    return experiments_parameterization


def train_test_and_evaluate(input_rep_filenames, bimp_log_filename, nome_grupo_experimentos, main_path, question,
                            approach_name, first_p_modelit_id=1, start_in_p_modelit_id=1):
    """ Function that execute training and testing in batch"""

    from anomaly_detection.main.execute_experiments_and_save_results import execute_experiments_and_save_results

    # Get parameterization
    experiments_parameterization = get_experiments_parameterization(nome_grupo_experimentos, main_path, question,
                                                                    input_rep_filenames, bimp_log_filename,
                                                                    approach_name,
                                                                    first_p_modelit_id, start_in_p_modelit_id)

    # Execute experiments: Train and test
    [experiments_group_path, results_data_filename] = execute_experiments_and_save_results(experiments_parameterization,
                                                                                           main_path)

    return [experiments_group_path, results_data_filename]


def execute_actions(phase_number, list_actions, bimp_log_filename, nome_grupo_experimentos, approach_name_list,
                    experiments_group_path=None, results_data_filename=None, enum_p_modelit_id_from=1,
                    start_in_p_modelit_id=1):
    # for action= "qualittv_analysis_create_d_analysis_files" the approach_name_list should start with aalst_approach

    import glob
    import os

    # set dirs:
    main_path = ROOT_DIR
    bimp_logs_path = BIMP_LOGS_DIR
    reference_logs_path = REFERENCE_LOGS_DIR
    logs_path = LOGS_DIR  # ( Original logs)

    # 0. Create logs
    if ("create_logs" in list_actions):
        if phase_number == 1:
            anomalies_list = ['allinone']  # vai inserir todas as anomalias em cada log
        if phase_number == 2:
            # Only skipp, only insert, only swit in every log
            anomalies_list = ['skipping_activity', 'activity_insertion', 'activity_switching']
        if (phase_number == 1) | (phase_number == 2):

            print("Not included yet")
    else:
        # If logs already were created
        original_logs_filenames = list(
            pd.read_csv(logs_path / ("original_logs_filenames_" + bimp_log_filename), header=None)[0])

    log_of_cases_all = pd.DataFrame()
    for approach_name in approach_name_list:
        if (("create_input_representations" in list_actions)
                | ("train_test_and_evaluate" in list_actions)
                | ("quantittv_analysis_by_approach_ap" in list_actions)
                | ("quantittv_analysis_by_approach_pr_rec" in list_actions)
                | ("qualittv_analysis_create_d_analysis_files" in list_actions)):
            print("\n ========================== Approach %s ==========================" % approach_name)

        # 1. Create ohe from logs
        if ("create_input_representations" in list_actions):
            input_rep_filenames = create_input_representation_from_logs(logs_path, original_logs_filenames,
                                                                        approach_name)

        if (("create_input_representations" not in list_actions) | (
                len(input_rep_filenames) < 1)):  # We assume, files are in hard disk
            if (approach_name == "autoencoder_nolle"):
                input_rep_filenames = "ohe_" + pd.Series(original_logs_filenames)

            if ((approach_name == "tstideplus") | (approach_name == 'binet1') | (
                    approach_name == 'detect_infrequents') | (
                    approach_name == 'random')):
                input_rep_filenames = "tstideplus_" + pd.Series(original_logs_filenames).apply(
                    lambda x: x.replace('.csv', '.pkl'))

            if (approach_name == "aalst_approach"):
                input_rep_filenames = "ohe_" + pd.Series(original_logs_filenames)
            input_rep_filenames = input_rep_filenames.tolist()

        # 2. Train and test in batch
        if ("train_test_and_evaluate" in list_actions):
            [experiments_group_path, results_data_filename] = train_test_and_evaluate(input_rep_filenames,
                                                                                      bimp_log_filename,
                                                                                      nome_grupo_experimentos,
                                                                                      main_path,
                                                                                      phase_number, approach_name,
                                                                                      enum_p_modelit_id_from,
                                                                                      start_in_p_modelit_id)

        # 3. Evaluate anomaly detection
        if ( "quantittv_analysis_by_approach_ap" in list_actions
            or "quantittv_analysis_by_approach_pr_rec" in list_actions
            or "qualittv_analysis_create_d_analysis_files" in list_actions):

            if experiments_group_path is None:

                all_result = glob.glob(os.path.join(RESULTS_DIR, 'resultados_gexp_' + get_abreviation_of(approach_name) + "*"))
                if len(all_result) > 0:
                    experiments_group_path_ = Path(all_result[0])
                    results_data_filename = os.path.basename(glob.glob(os.path.join(experiments_group_path_, "*_all_results_details.csv"))[0])
                else:
                    experiments_group_path_ = None
                    print("\n It was not found any folder named resultados_gexp_%s*" % get_abreviation_of(approach_name))
            else:
                experiments_group_path_ = Path(experiments_group_path)
        ##3.1 Quantitative analysis
        if ("quantittv_analysis_by_approach_pr_rec" in list_actions or "quantittv_analysis_by_approach_ap" in list_actions):

            from anomaly_detection.pos_processamento import p_q1_create_figures_from_results_files as figures_ad

            if (experiments_group_path_ is not None):
                if ("quantittv_analysis_by_approach_ap" in list_actions):
                    if phase_number == 1:
                        dict_limiar_test = {
                            "autoencoder_nolle": 0.78,
                            "tstideplus": 0.9,
                            "binet1": 0.9,
                            "aalst_approach": 0.88
                        }
                        figures_ad.create_analysis_q1(experiments_group_path_,
                                                      results_data_filename,
                                                      dict_limiar_test,
                                                      statistic_analysis=True,
                                                      iplot=True,
                                                      print_figure=False,
                                                      # figure_numbers=[12])
                                                      figure_numbers=[11, 12])

                    if phase_number == 2:
                        figures_ad.create_analysis_q1(experiments_group_path_,
                                                      results_data_filename,
                                                      dict_limiar_test=None,
                                                      statistic_analysis=True,
                                                      iplot=True,
                                                      print_figure=False,
                                                      figure_numbers=[13])

                if ("quantittv_analysis_by_approach_pr_rec" in list_actions):
                    figures_ad.create_analysis_q1(experiments_group_path_,
                                                  results_data_filename,
                                                  dict_limiar_test=None,
                                                  statistic_analysis=True,
                                                  iplot=True,
                                                  print_figure=False,
                                                  figure_numbers=[17])


    if ("quantittv_analysis_all_approaches_in_one" in list_actions) & (phase_number == 2):
        from anomaly_detection.pos_processamento import p_q1_create_figures_from_results_files as figures_ad
        print("\n ========================== All Approaches in list ==========================")

        df_full_name = "full.csv"
        # If full file does not exist, we create it
        if not (os.path.isfile(RESULTS_DIR / df_full_name)):
            print("full.csv was created")
            df_full = pd.DataFrame()
            for approach_name in approach_name_list:
                # experiments_group_paths= glob.glob(os.path.join(RESULTS_DIR, 'resultados_gexp_' + "*"))
                experiments_group_path = \
                glob.glob(os.path.join(RESULTS_DIR, 'resultados_gexp_' + get_abreviation_of(approach_name) + "*"))[0]
                experiments_group_path = Path(experiments_group_path)
                results_data_filename = glob.glob(os.path.join(experiments_group_path, "*_all_results_details.csv"))[0]
                df_full = df_full.append(pd.read_csv(results_data_filename))
            df_full.to_csv(RESULTS_DIR / df_full_name, sep=',', encoding='utf-8', index=False)
        else:
            print("full.csv already exists. It will be used for analisys")

        figures_ad.create_analysis_q1(RESULTS_DIR,
                                      df_full_name,
                                      dict_limiar_test=None,
                                      statistic_analysis=True,
                                      iplot=True,
                                      print_figure=False,
                                      figure_numbers=[14, 15, 16, 18])
        #  figure_numbers = [18]) # for dissertation


def get_params_by_research_questions(phase_number):
    """ Definition of parameters that will be executed used according to phase_number selected.

    :param
        phase_number: integer. Possible values:
            1 (Look for the best parameterizations)
            2 (Analyze robustness of approaches,
            3 (Analyze approaches to process discovery improvement).
    :return: list
    """
    if (phase_number == 1) | (phase_number == 2) | (phase_number == 3):
        bimp_log_filename = "log_bimp_ptall.csv"

    if (phase_number == 1):
        nome_grupo_experimentos = "relatorio1-question1"
        task_list = ["create_logs", "create_input_representations", "train_test_and_evaluate",
                     "quantittv_analysis_by_approach_ap"]

    if (phase_number == 2):
        # Possible tasks:
        #   A. Tasks for log creation and training: ["create_logs","create_input_representations","train_test_and_evaluate"]
        #   B. Tasks for quantitative analysis  : ["quantittv_analysis_by_approach_ap","quantittv_analysis_all_approaches_in_one"]
        #   C. Tasks for creation of files for qualitative analysis: ["qualittv_analysis_create_d_analysis_files"]
        #   D. Opcional Task : ["quantittv_analysis_by_approach_pr_rec"]
        nome_grupo_experimentos = "relatorio1-question2"
        task_list = ["create_input_representations", "train_test_and_evaluate",
                     "quantittv_analysis_all_approaches_in_one"]

    if (phase_number == 3):
        nome_grupo_experimentos = "relatorio1-question3"
        task_list = ["generate_filtered_logs_and_labeled", "generate_pmodels_and_quality_measures"]

    return [bimp_log_filename, nome_grupo_experimentos, task_list]


def main(phase_number):
    approach_name_list = ["tstideplus", "autoencoder_nolle", "binet1", "detect_infrequents", "aalst_approach"]
    approach_name_list = ["tstideplus", "autoencoder_nolle", "binet1", "detect_infrequents"]
    [bimp_log_filename, nome_grupo_experimentos, list_actions] = get_params_by_research_questions(phase_number)
    nome_grupo_experimentos = "test_case"
    import time
    time_tst_start = time.time()
    execute_actions(phase_number, list_actions, bimp_log_filename, nome_grupo_experimentos, approach_name_list,
                    experiments_group_path=None, results_data_filename=None, enum_p_modelit_id_from=1,
                    start_in_p_modelit_id=0)
    time_tst_end = time.time()
    time_tst = time_tst_end - time_tst_start
    print("\n time for execute actions:" + str(time_tst) + "\n")

main(phase_number=2)
