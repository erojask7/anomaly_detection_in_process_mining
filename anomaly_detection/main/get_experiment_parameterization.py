#FIX2022forpubrep: ready for publication
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(1, "E:\\EXPERIMENTOS2020\\anomaly_detection_artigo\\")
from anomaly_detection.general.global_variables import *


def verify_params(s_method_for_explore_thresholds, s_tipo_heuristica, scaling_factors_values):

    if (s_tipo_heuristica == None):  # If no one heuristic was set
        if (scaling_factors_values == '') | (s_method_for_explore_thresholds == ''):
            raise Exception(
                "ERROR, if no one heuristic was set, we should set scaling_Factors_values and s_method_for_explore_thresholds")
    else:  # If any heuristic was set
        if (scaling_factors_values != '') | (s_method_for_explore_thresholds != ''):  # if scaling factor was not set
            raise Exception(
                "ERROR, if one heuristic was set,  set scaling_Factors_values='' and s_method_for_explore_thresholds=''")

def get_fixed_parameterization_model(approach_name,nome_grupo_experimentos=None):
    """
    Function that get parameters that will be fixed for all experiments, according to approach.
    It means that the returned dictionary will set the parameter values for all experiments for the specified approach.

    Parameters
    ----------
    approach_name
    nome_grupo_experimentos

    Returns
    -------
    dictionary

    """

    if approach_name == 'autoencoder_nolle':
        if nome_grupo_experimentos =='test_case': #not especified
            dict_parameterization = {
                # Model_ignoring_threshold parameters
                'p_modelit_type': approach_name,
                'nitmax': 5,  # Maximum number of epochs #paper:200
                'batch_size': 500,
                'early_stopping_patiente': 10,

                'nro_camadas_ocultas': 2,
                'gaussian_noise_std': 0.1,
                'dropout': 0.5,
                'optimizer_beta2': 0.99,
            }
        else:
            dict_parameterization = {
                # Model_ignoring_threshold parameters
                'p_modelit_type': approach_name,
                'nitmax': 200,  # Maximum number of epochs #paper:200
                'batch_size': 500,
                'early_stopping_patiente': 10,

                'nro_camadas_ocultas': 2,
                'gaussian_noise_std': 0.1,
                'dropout': 0.5,
                'optimizer_beta2': 0.99,
            }

    elif approach_name == 'binet1':
        dict_parameterization = {
            # Model_ignoring_threshold parameters
            'p_modelit_type': approach_name,
            'batch_size': 500,
            'early_stopping_patiente': 10,

            'gaussian_noise_std': 0.1,
            'optimizer_beta2': 0.999,
            'decoder': False,
        }
    else:
        dict_parameterization = {
            # Model_ignoring_threshold parameters
            'p_modelit_type': approach_name,
        }
    return dict_parameterization


def get_combinated_dynamic_parameters(dataset_values, min_sf, max_sf, sf_number_of_values, approach_name, params,
                                      enum_p_modelit_id_from=1):  # IMPORTANT SET
    """
    This function create a combination of several parameters. That combination creates experiments with different
    parameter values

    Parameters
    ----------
    min_sf: minimum value for scaling factor
    max_sf: maximum value for scaling factor
    sf_number_of_values : number of scaling factor values
    enum_p_modelit_id_from: Default 1. id from wich we will start the counting of experiments. It could be used when
        we want execute a new set of experiments but we want the ids will be start in enum_p_modelit_id_from and not in 1.
        When we have a previous set off executions and we dont want to have several experiments using the same id [p_modelit_id]

    """

    from itertools import product

    # Set scalign factors
    sfs_values = list(
        np.round(np.linspace(min_sf, max_sf, sf_number_of_values), 3))
    sfs_values = ' '.join(map(str, sfs_values))

    vapproach_values = params.get('vapproach_values')
    voutlierness_calc_values = params.get('voutlierness_calc_values')

    if approach_name == 'aalst_approach':
        thdd_min = params.get('thdd_min')
        thdd_max = params.get('thdd_max')
        thdd_number_of_values = params.get('thdd_number_of_values')
        threshold_values = set(np.round(np.linspace(thdd_min, thdd_max, thdd_number_of_values), 3))  # Tc

        min_kk = params.get('min_kk')
        max_kk = params.get('max_kk')
        subsequence_values = set(range(min_kk, max_kk + 1, 1))  # K

        params_experiments = np.array(sorted(list(product(subsequence_values, threshold_values,
                                                          dataset_values, vapproach_values,
                                                          voutlierness_calc_values))))  # size of params_experiments= size of experimentos

        dynamic_parameters = {
            'p_modelit_id': range(enum_p_modelit_id_from, enum_p_modelit_id_from + len(params_experiments)),
            # nro_experimento,
            'p_modelit_kk': params_experiments[:, 0],  # funcao_g  # funcao de ativacao da camada de saida
            'p_modelit_thdd': params_experiments[:, 1],  # funcao_f # funcao de ativacao da camada oculta
            'log_name': params_experiments[:, 2],  # log_name
            'scaling_factors': sfs_values,
            'p_modelit_vapproach': params_experiments[:, 3],
            'p_modelit_voutlierness_calc': params_experiments[:, 4],
        }
    if approach_name == 'tstideplus':
        thd_min = params.get('thd_min')
        thd_max = params.get('thd_max')
        thd_number_of_values = params.get('thd_number_of_values')
        threshold_values = set(np.round(np.linspace(thd_min, thd_max, thd_number_of_values), 3))  # Tc

        min_windows_size = params.get('min_windows_size')
        max_windows_size = params.get('max_windows_size')
        subsequence_values = set(range(min_windows_size, max_windows_size + 1, 1))  # K

        params_experiments = np.array(sorted(list(product(subsequence_values, threshold_values,
                                                          dataset_values, vapproach_values,
                                                          voutlierness_calc_values))))
        # len(params_experiments) will be the total numbr of experiments

        dynamic_parameters = {
            'p_modelit_id': range(enum_p_modelit_id_from, enum_p_modelit_id_from + len(params_experiments)),
            'p_modelit_windows_size': params_experiments[:, 0],
            'p_modelit_thd': params_experiments[:, 1],
            'log_name': params_experiments[:, 2],
            'scaling_factors': sfs_values,
            'p_modelit_vapproach': params_experiments[:, 3],
            'p_modelit_voutlierness_calc': params_experiments[:, 4],
        }
    if (approach_name == 'detect_infrequents') | (approach_name == 'random'):
        params_experiments = np.array(sorted(list(product(dataset_values, vapproach_values,
                                                          voutlierness_calc_values))))

        dynamic_parameters = {
            'p_modelit_id': range(enum_p_modelit_id_from, enum_p_modelit_id_from + len(params_experiments)),
            'log_name': params_experiments[:, 0],  # log_name
            'scaling_factors': sfs_values,
            'p_modelit_vapproach': params_experiments[:, 1],
            'p_modelit_voutlierness_calc': params_experiments[:, 2],
        }
    if approach_name == 'autoencoder_nolle':
        alfa_min, alfa_max, alfa_number_of_values = params.get('alfa_min'), params.get('alfa_max'), params.get(
            'alfa_number_of_values'),
        # Set learning rates
        alfa_values = set(np.round(np.linspace(alfa_min, alfa_max, alfa_number_of_values), 3))  # Tc

        # Set batch_size
        # batch_size={500}    

        # Multiplication of parameters ( All against alls)
        funcao_f_values, funcao_g_values = params.get('funcao_f_values'), params.get('funcao_g_values')
        params_experiments = np.array(sorted(list(product(funcao_f_values, funcao_g_values, alfa_values,
                                                          dataset_values, vapproach_values,
                                                          voutlierness_calc_values))))

        # Set hidden number of neurons automatically
        ne_list = []
        no_list = []
        for i in range(len(params_experiments)):
            ne = len(pd.read_csv(os.path.join(OHE_DIR, params_experiments[i, 3]), header=None).iloc[1, :-2])
            ne_list.append(ne)
            no_list.append(ne / 2)  # no=ne/2 ( number of hidden neurons is the half of the input neurons

        dynamic_parameters = {
            'p_modelit_id': range(enum_p_modelit_id_from, enum_p_modelit_id_from + len(params_experiments)),
            'funcao_f': params_experiments[:, 0],  # activation function of input layer
            'funcao_g': params_experiments[:, 1],  # activation function of output layer
            'alfa': params_experiments[:, 2].astype(float),  # learning rate
            'log_name': params_experiments[:, 3],  # log_name
            'scaling_factors': sfs_values,
            'p_modelit_vapproach': params_experiments[:, 4],
            'p_modelit_voutlierness_calc': params_experiments[:, 5],

            # Dynamic but set automatically
            'ne': ne_list,
            'no': no_list,  # number of neurons in hidden layers

        }
    if approach_name == 'binet1':
        alfa_min, alfa_max, alfa_number_of_values = params.get('alfa_min'), params.get('alfa_max'), params.get(
            'alfa_number_of_values'),
        # Set learning rates
        alfa_values = set(np.round(np.linspace(alfa_min, alfa_max, alfa_number_of_values), 3))  # Tc

        # Set batch_size
        nitmax_values = set(
            np.round(np.linspace(params.get('nit_min'), params.get('nit_max'), params.get('nit_number_of_values')), 3))

        # Multiplication of parameters ( All against alls)
        # funcao_f_values, funcao_g_values = params.get('funcao_f_values'), params.get('funcao_g_values')
        params_experiments = np.array(sorted(list(product(alfa_values,
                                                          dataset_values, nitmax_values, vapproach_values,
                                                          voutlierness_calc_values))))  # size of params_experiments= size of experimentos

        dynamic_parameters = {
            'p_modelit_id': range(enum_p_modelit_id_from, enum_p_modelit_id_from + len(params_experiments)),
            'alfa': params_experiments[:, 0].astype(float),  # learning rate
            'log_name': params_experiments[:, 1],  # log_name
            'scaling_factors': sfs_values,
            'p_modelit_vapproach': params_experiments[:, 3],
            'p_modelit_voutlierness_calc': params_experiments[:, 4],
            'nitmax': params_experiments[:, 2],

        }

    number_of_models_it = len(params_experiments)
    return [number_of_models_it, dynamic_parameters]


def get_derived_parameters_set_automatically(approach_name, d_general_fixed_parameters):  # Pode ir em outro py
    parameters = {}
    if approach_name == 'autoencoder_nolle':

        # Set early stopping ON / OFF
        if ('train_with_all' in d_general_fixed_parameters.get('s_strategy_tr_tst')):
            early_stopping_metric = 'loss'
        else:
            early_stopping_metric = 'val_loss'

        parameters = {
            # Model_with_threshold parameteres
            'early_stopping_metric': early_stopping_metric,
        }
    return parameters
