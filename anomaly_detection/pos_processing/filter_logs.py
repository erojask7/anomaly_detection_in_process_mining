from anomaly_detection.general.global_variables import *

def generate_semantic_result_and_filtered_logs_and_labeled(scaling_factor, results_path, folder_output,
                                                           experiment_number, train_or_test, iteration_cv,
                                                           log, table_id_encodings, log_original,log_type):
    """
    Function that create filtered logs according to detection made by an anomaly detection model.
    Information about detection using experiment_number and scaling_factor will be searched in "results_path/OUT_DIRNAME"

    The filtered event log will be saved as *_filtered_log.csv

    """
    import pandas as pd
    print('dataset: '+train_or_test)
    print('==============================')
    print('\n')
    print('scaling_factor: '+scaling_factor)
    print('\n')

    # A) Create detection.csv file

    # Read test idx
    test_idx= pd.read_csv(results_path / OUT_DIRNAME/
                          ("exp"+experiment_number+"_iter"+iteration_cv+"_"+train_or_test+"_idx.csv"),
                          header=None)

    # Create semantic labels
    set_cases=log.groupby('traceid')['activity'].apply(lambda x: ' ,'.join(x))
    set_cases=pd.DataFrame(set_cases)
    set_cases['nr_events']=list(log.groupby(['traceid'])['activity'].count())

    detection=test_idx
    detection['traceid']= test_idx[0]+1
    log=log[log['traceid'].isin(test_idx[0]+1)]
    log_original=log

    # Read y_pred
    y_pred=pd.read_csv(results_path /OUT_DIRNAME/ ("exp"+experiment_number+"_iter"+iteration_cv+"_sf"+scaling_factor+"_"+train_or_test+"_y_pred.csv"),header=None)
    detection['y_pred']=y_pred[0]

    test_traceid= list(detection['traceid'].values)
    detection['trace']=list(set_cases['activity'].loc[test_traceid])
    detection['nr_events']=list(set_cases['nr_events'].loc[test_traceid])
    detection=detection.drop([0],axis=1)

    detection['y_pred']=detection['y_pred'].replace(to_replace={'a':1,'n':0})

    # Number of predicted as anomalous and normal cases
    print('Predicted as anomalous and normal: ')
    print(detection.groupby('y_pred')['trace'].count())
    print('\n')

    # Add column 'trace_id_originais'. Stablish an relation with previous traceid
    detection['trace_id_original']=detection['traceid']
    if( table_id_encodings is not None):
        list1= list(table_id_encodings['trace_id_original'].values)
        list2= list(table_id_encodings['trace_id_nuevo'].values)
        dictionary_replace=dict(zip(list2,list1))
        detection['trace_id_original']=detection['trace_id_original'].map(dictionary_replace)

    # Create folder_output
    import os
    for folder in [folder_output,folder_output / DETECTION_DIR_NAME, folder_output / LABELED_DIR_NAME, folder_output /FILTERED_DIR_NAME]:
        if not (os.path.isdir(folder))  : #If folder doesnt exist
            os.mkdir(folder)    #Create folder

    # Save detection file
    detection= detection[['trace_id_original','trace','y_pred','nr_events','traceid']]
    filename_detection="logofcases_exp"+experiment_number+\
                       "_iter"+iteration_cv+\
                       "_sf"+scaling_factor+\
                       "_"+train_or_test+"_detection.csv"
    detection.to_csv(folder_output / DETECTION_DIR_NAME/ filename_detection,sep=',', encoding='utf-8',index=False) #detection_semantic

    # B) Create log_original_labeled

    if(log_type=='log_normal'):
        original_log_labeled= log_original.copy()
        original_log_labeled['label']=-1
        anomalous_tracesid_list=list(detection[detection['y_pred']==1]['trace_id_original'].values)
        original_log_labeled.loc[original_log_labeled['traceid'].isin(anomalous_tracesid_list),'label']='anomaly'
        original_log_labeled.loc[~original_log_labeled['traceid'].isin(anomalous_tracesid_list),'label']='normal'
        original_log_labeled= original_log_labeled.drop(columns='anomaly_type')
        #log_original_labeled.to_csv(results_path / ("exp"+experiment_number+"_iter"+iteration_cv+"_sf"+scaling_factor+"_"+train_or_test+"_log_detection.csv"),sep=',', encoding='utf-8',index=False)
        filename_log_rotulado="exp"+experiment_number+"_iter"+iteration_cv+"_sf"+scaling_factor+"_"+train_or_test+"_labeled_log.csv"
        original_log_labeled.to_csv(folder_output /LABELED_DIR_NAME/ filename_log_rotulado,sep=',', encoding='utf-8',index=False) #detection_labeled

    # C) Create filtered log
    # All anomalous labeled cases will be removed in dataframe
    log_filtrado=original_log_labeled[original_log_labeled['label']=='normal']

    # Save filtered event log
    filename_log_filtrado="exp"+experiment_number+\
                          "_iter"+iteration_cv+\
                          "_sf"+scaling_factor+\
                          "_"+train_or_test+"_filtered_log.csv"
    log_filtrado.to_csv(folder_output / FILTERED_DIR_NAME/ filename_log_filtrado,sep=',', encoding='utf-8',index=False)

    return [filename_detection,filename_log_filtrado,filename_log_rotulado]


def run_one_batch(execution_plan, results_path, folder_output, train_or_test, exp_group_id, batch_nro, logs_path, log_type='normal'):
    import time
    import pandas as pd
    import os
    if log_type== 'log_normal':

        if os.path.isfile(logs_path / "table_id_encodings.csv"):
            table_id_encodings=pd.read_csv(logs_path / "table_id_encodings.csv")
        else:
            table_id_encodings=None

    elif(log_type=='log_without_duplicates'):
        log= pd.read_csv(logs_path / "log_uci_refatored_without_duplicates.csv")
        table_id_encodings=pd.read_csv(logs_path / "table_id_encodings_without_duplicates.csv")
        log_original= pd.read_csv(logs_path / "log_uci_complete.csv", sep=',')

    actual_batch=execution_plan.copy()
    actual_batch=actual_batch[actual_batch['batch_nro']==batch_nro]
    print('batch_nro: '+str(batch_nro))

    for index, experiment in actual_batch.iterrows():

        start = time.time()

        # Capture information about executed experiment
        scaling_factor=str(experiment['scaling_factor'])
        experiment_number=str(experiment['p_modelit_id'])
        iteration_cv=str(experiment['iteracao_cv'])
        log_name = experiment['log_name'].replace('tstideplus_','').replace('ohe_','').replace('.pkl','.csv')

        if ("UCI" not in str(RQ_DIR)):
            log = pd.read_csv(logs_path / log_name)
            log_original = log

        # Generate filtered log and labeled_log
        [filename_detection,filename_log_filtrado,filename_log_rotulado]=generate_semantic_result_and_filtered_logs_and_labeled(scaling_factor, results_path, folder_output, experiment_number, train_or_test, iteration_cv, log, table_id_encodings, log_original,log_type)

        # Save names of files generated
        actual_batch.loc[index,'filename_detection']=filename_detection
        actual_batch.loc[index,'filename_log_filtrado']=filename_log_filtrado
        actual_batch.loc[index,'filename_log_rotulado']=filename_log_rotulado

        # Save results of batch
        actual_batch.to_csv(folder_output / (exp_group_id+"_all_results_details_batch"+str(batch_nro)+".csv"),
                            sep=',', encoding='utf-8',index=False)

        end = time.time()
        total_time_secs=end-start
        total_time_mins=total_time_secs/60
        print("Time:",total_time_secs, "(secs) ", total_time_mins,"(mins)")
    return execution_plan


def create_execution_plan(df_all_results, batch_definition_type, batch_definition_conditions):
    """
    Function that creates an execution plan. The plan is saved in an .csv specifying the order
    of execution of each filtering. A new column is included in df_all_results in order to specify a number of batch.
    That number will be used to prioritize execution.

    Parameters
    ----------
    df_all_results: DataFrame
        Each row of this dataframe contain information about one anomaly detection model and its classification metrics (F1-score, AP, etc)

    batch_definition_type. String
          It could be "by_priorities" or "one_unique_batch"
    batch_definition_conditions

    Returns
    -------
    df_all_results containing a new column that indicates the priority in execution
    """

    from anomaly_detection.general.utils_da import  filter_dataframes
    if batch_definition_type == "by_priorities":
        [cond,_]=filter_dataframes(df_all_results, batch_definition_conditions)
        df_all_results.loc[cond, 'batch_nro'] = 1
        df_all_results.loc[~cond, 'batch_nro'] = 2
    elif batch_definition_type == "one_unique_batch":
        df_all_results.loc[:, 'batch_nro']=1
    return df_all_results


def reformat_all_results_files_old(df):
    df=df.rename(columns={'nro_experimento':'p_modelit_id'})
    return df

def create_plan_and_execute(batch_definition_conditions, batch_definition_type, batch_nr_list, exp_group_id,results_path, all_results_filename, log_type="normal"):
    """
    Function that creates a plan of execution (*_all_results_details_execplan.csv)
    Rows in that .csv will be executed row by row. Each row is one prediction made for one approach, thus,
    given a specific row, anomalous cases will be filtered in full log (original log) according to predictions made
    by the anomaly detection model referred in that row.

    At the end, filtered logs will be saved in \logs_postproces_batch folder

    Parameters
    ----------
    batch_nr_list: list
        list containing the batchs that will be executed. Example [1,2,3]

    exp_group_id: String
        ID of experiment group related to all_results_filename

    results_path: String
        folder where execution plan will be saved

    all_results_filename: String
        filename of .csv containing results information

    """
    import pandas as pd
    import os

    # Paths
    path= ROOT_DIR
    logs_path=LOGS_DIR

    # Other parameters
    train_or_test='test'
    df_all_results= pd.read_csv(os.path.join(RESULTS_DIR,results_path, all_results_filename))

    # Create and save plan of execution
    execution_plan=create_execution_plan(df_all_results, batch_definition_type, batch_definition_conditions)
    execution_plan.to_csv(results_path /(all_results_filename.replace(".csv","_execplan.csv")),sep=',', encoding='utf-8',index=False)

    # Execute plan
    for batch_nro in batch_nr_list:
        folder_output = results_path / ("logs_postproces_batch" + str(batch_nro))
        run_one_batch(df_all_results, results_path, folder_output, train_or_test, exp_group_id, batch_nro, logs_path, log_type)

