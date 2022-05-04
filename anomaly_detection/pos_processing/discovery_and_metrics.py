"""
Created on Thu Jun 11 16:39:48 2020

@author: Ana Rocio Cardenas Maita & Marcelo Fantinato
@updatedby: Esther Rojas
"""

import csv
import os
import pandas as pd
from time import time
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.evaluation.replay_fitness import evaluator as replay_fitness
from pm4py.evaluation.precision import evaluator as calc_precision
from pm4py.evaluation.simplicity import evaluator as calc_simplic
from pm4py.algo.conformance.alignments import algorithm as alignments
from pm4py.objects.conversion.log import converter as log_converter
from anomaly_detection.general.global_variables import *


def calculate_quality_metrics(model_log_csv, metric_log_csv, model_base, log_base, test_ids, batch_dir,filename_filtered_log,full_log_name=None):
    log = "other"
    start_time = time()

    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'traceid'}
    model_log = log_converter.apply(model_log_csv, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
    metric_log = log_converter.apply(metric_log_csv, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
    parameters = {inductive_miner.Variants.DFG_BASED.value.Parameters.CASE_ID_KEY: 'traceid',
                  inductive_miner.Variants.DFG_BASED.value.Parameters.ACTIVITY_KEY: 'activity',
                  alignments.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters.ACTIVITY_KEY: 'activity'}

    petrinet, initial_marking, final_marking = inductive_miner.apply(model_log, parameters=parameters)
    gviz = pn_visualizer.apply(petrinet, initial_marking, final_marking)

    if not (os.path.isdir(batch_dir/"petrinets")):  # If folder doesnt exist
        os.mkdir(batch_dir/"petrinets")  # Create folder

    # model_name_in_disk is the name of model (.png) in disk
    if("filter_model" in model_base):
        model_name_in_disk=filename_filtered_log.replace("filtered_log.csv","filtered_model")
    elif("full_model" in model_base):
        model_name_in_disk = full_log_name

    if not(os.path.isfile(batch_dir/'petrinets' /model_name_in_disk)):
        gviz.render(batch_dir/'petrinets' /model_name_in_disk)

    # Calculate quality metrics
    alignments_res = alignments.apply_log(metric_log, petrinet, initial_marking, final_marking, parameters=parameters)
    fitness = replay_fitness.evaluate(alignments_res, variant=replay_fitness.Variants.ALIGNMENT_BASED,
                                      parameters=parameters)
    precision = calc_precision.apply(metric_log, petrinet, initial_marking, final_marking, parameters=parameters)
    generaliz = 0
    simplic = calc_simplic.apply(petrinet)
    f_score = 2 * ((fitness['averageFitness'] * precision) / (fitness['averageFitness'] + precision))

    # Print progress in console
    end_time = time()
    m, s = divmod(end_time - start_time, 60)
    h, m = divmod(m, 60)
    print(model_base + '/' + log_base + ' F:',
          '%.10f' % fitness['averageFitness'],
          ' P:', '%.10f' % precision,
          ' FS:', '%.10f' % f_score,
          ' G:', '%.10f' % generaliz,
          ' S:', '%.10f' % simplic, ' T:',
          '%02d:%02d:%02d' % (h, m, s))

    # Format quality metrics
    metrics = pd.Series(['%.10f' % fitness['averageFitness'],
                         '%.10f' % precision,
                         '%.10f' % f_score,
                         '%.10f' % generaliz,
                         '%.10f' % simplic,
                         '%02d:%02d:%02d' % (h, m, s)])
    return [model_base,metrics]


def discover_and_calculate_metrics_batch(exp_group_id, batch_dir_name, batch_filename):
    """
    Function that discover models from logs registered in *_all_results_details_batch1.csv
    Discovered models will be saved in "petrinets" folder
    Quality metrics quality will be saved in *_all_results_details_dscvry_batch1.csv, inside folder "batch_filename"

    Parameters
    ----------
    exp_group_id: ID of experiment group. It will be used to find *_all_results_details_batch1.csv in disk
    batch_dir_name: folder where *_all_results_details_batch1.csv is located
    batch_filename: complete name of *_all_results_details_batch1.csv.
        Example: gexp_stp_62918022799_all_results_details_batch1.csv


    """

    batch_number = int(batch_dir_name[len(BATCHES_FOLDER_NAME):])
    gexp = pd.read_csv(
        os.path.join(RESULTS_DIR, "resultados_"+exp_group_id, batch_dir_name, batch_filename + '.csv'))
    gexp = gexp[gexp['batch_nro'] == batch_number]

    # Folder where results will be saved
    batch_dir=RESULTS_DIR / ("resultados_"+exp_group_id) / batch_dir_name
    # Folder name where results will be saved
    results_filename= batch_filename.replace("all_results_details","all_results_details_dscvry") + '.csv'

    # Writing header of .csv
    csv_results = open(batch_dir/ results_filename, 'w', newline='')
    results = csv.writer(csv_results, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\\')

    results.writerow(
        list(gexp.columns)+['model_base',
                            'F_o', 'P_o', 'FS_o', 'G_o', 'S_o', 'time_o',
                            'F_f', 'P_f', 'FS_f', 'G_f', 'S_f', 'time_f',
                            'F', 'P', 'FS', 'G', 'S', 'time'])

    del results
    csv_results.close()
    gexp=gexp.sort_values(by='log_name_abbrev')

    # Write results in excel
    for index, row in gexp.iterrows():
        csv_results = open(batch_dir / results_filename, 'a', newline='')
        results = csv.writer(csv_results, delimiter=',')
        log_name = row['log_name'].replace('tstideplus_', '')\
                                    .replace('ohe_', '')\
                                    .replace('.pkl', '.csv')

        full_path_full_log = os.path.join(LOGS_DIR,log_name )
        filename_filtered_log = row['filename_log_filtrado']

        experiment_number=str(row['p_modelit_id'])
        iteration_cv=str(row['iteracao_cv'])
        train_or_test='test'
        test_ids = pd.read_csv( RESULTS_DIR / ("resultados_"+exp_group_id)/ OUT_DIRNAME / (
                    "exp" + experiment_number + "_iter" + iteration_cv + "_" + train_or_test + "_caseids.csv"), header=None)[0]


        full_path_filtered_log = os.path.join(batch_dir, 'filtered_logs', filename_filtered_log)


        if os.path.isfile(full_path_filtered_log):
            # Open Sublog
            full_log = pd.read_csv(full_path_full_log, ',')  # it will be used to create the model
            full_log = full_log[full_log['traceid'].isin(test_ids.tolist())]
            full_log_name = row["log_name_abbrev"] + "_iter" + iteration_cv

            # Open filtered sub log
            filtered_log=pd.read_csv(full_path_filtered_log, ',')
            print("\n"+filename_filtered_log)

            # Calculate quality metrics for filtered model versus full log
            [model_base,metrics_filter_full] = calculate_quality_metrics(filtered_log, full_log,
                                                            'filter_model_' + exp_group_id[:-12] + '_' + batch_dir_name[:-14] + '_' + batch_filename[:-20],
                                                            'full_log',test_ids,batch_dir,filename_filtered_log)

            # Calculate quality metrics for filtered model versus filtered log
            [_,metrics_filter_filter] = calculate_quality_metrics(filtered_log, filtered_log,
                                                              'filter_model_' + exp_group_id[:-12] + '_' + batch_dir_name[:-14] + '_' + batch_filename[:-20],
                                                              'filter_log_' + exp_group_id[:-12] + '_' + batch_dir_name[:-14] + '_' + batch_filename[:-20],
                                                                  test_ids,batch_dir,filename_filtered_log)

            # Calculate quality metrics for full model versus full log
            [_,metrics_full_full] = calculate_quality_metrics(full_log, full_log,
                                                              'full_model_' + exp_group_id[:-12] + '_' + batch_dir_name[:-14] + '_' + batch_filename[:-20],
                                                              'full_log_' + exp_group_id[:-12] + '_' + batch_dir_name[:-14] + '_' + batch_filename[:-20],
                                                                  test_ids,batch_dir,filename_filtered_log,full_log_name)

            results.writerow(pd.concat([row, pd.Series([model_base]),metrics_filter_full,metrics_filter_filter,metrics_full_full], ignore_index=True))

            del results
            csv_results.close()

        else:
            print('*arquivo no encontrado: ', full_path_filtered_log)