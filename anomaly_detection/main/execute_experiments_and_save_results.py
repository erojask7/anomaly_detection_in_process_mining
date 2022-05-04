#FIX2022forpubrep: ready for publication
import numpy as np
from anomaly_detection.general.global_variables import *


def execute_experiments_and_save_results(experiments, main_path):
    # Create folder for results
    import os
    import pandas as pd
    from anomaly_detection.general.utils_da import get_params_all_results
    exp_group_id = experiments['exp_group_id'][experiments.index[0]]
    folder_output = os.path.join(RESULTS_DIR, "resultados_%s" % (exp_group_id))
    if not (os.path.isdir(folder_output)):  # If folder doesnt exist
        os.makedirs(folder_output)  # Create folder

    # Create OUT DIR
    if not (os.path.isdir(os.path.join(folder_output, OUT_DIRNAME))):  # If folder doesnt exist
        os.makedirs(os.path.join(folder_output, OUT_DIRNAME))  # Create folder
    print('folder_output: ')
    print(folder_output)
    print('\n')

    # Save experiments settings
    # Each line is a parameterization of the model. Every line has an specific range of thresholds wrote as string
    experiments.to_csv(os.path.join(folder_output, "%s_parameterization.csv" % exp_group_id), sep=',',
                       encoding='utf-8', index=False)

    # Execute experiments
    approach_name = experiments['p_modelit_type'].iloc[0]
    params_ordered_list = get_params_all_results(approach_name)
    for index, experimento in experiments.iterrows():

        # Execute an experiment
        results = execute_experiment(experimento, approach_name, folder_output)
        results.to_csv(os.path.join(folder_output, OUT_DIRNAME, "exp%s_results_details.csv" % experimento['p_modelit_id']), sep=',',
                       encoding='utf-8', index=False)
        if results is None:
            continue
        all_results_name = "%s_all_results_details.csv" % experimento['exp_group_id']

        # Verify if columns match with expected params for that approach
        difference1 = set(results.columns).difference(params_ordered_list)
        difference2 = set(params_ordered_list).difference(results.columns)
        if (len(difference1) > 0) | (len(difference2) > 0):
            raise Exception("Error. Columns do not match. \n You should include in params_ordered_list: "+str(difference1)+
                            "\n You should include in results.columns: "+str(difference2))

        if not(os.path.isfile(os.path.join(folder_output, all_results_name))): # If does not exist #first append
            all_results_details = results
        else: # if already exists
            all_results_details = pd.read_csv(os.path.join(folder_output, all_results_name), sep=',',
                                                         encoding='utf-8', dtype=results.dtypes.to_dict())
            all_results_details=all_results_details.append(results)

        # Order columns
        all_results_details = all_results_details[params_ordered_list]

        # Save file
        all_results_details.to_csv(os.path.join(folder_output, all_results_name), sep=',', encoding='utf-8',
                                   index=False)

    return [folder_output, all_results_name]


def execute_experiment(experiment, approach, folder_output):
    """
    Create ONE anomaly detection model and test over one log

    Parameters
    ----------
    experiment: parameter values of experiment
    approach:
    folder_output : folder where results will be saved

    Returns
    -------

    """

    from anomaly_detection.main.execute import execute_exp
    print('\n')
    print("=================Experiment%s=====================" % experiment['p_modelit_id'])
    print('\n')
    print('Data about experiment:')
    print(experiment)
    print('\n')

    # Convert scaling_factors to list
    scaling_factors = np.array(list(map(float, experiment['scaling_factors'].split())))
    results = None

    results = execute_exp(experiment, scaling_factors, folder_output, fast_debug=False)
    return results