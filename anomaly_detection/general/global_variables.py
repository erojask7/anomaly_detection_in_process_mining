from pathlib import Path

# Root
ROOT_DIR = Path(__file__).parent.parent.parent

# Outputs of anomaly detection_code
RQS_DIR = ROOT_DIR / 'research_questions'

# Folder where all logs and results are stored
RQ_DIR = RQS_DIR / 'best_anomaly_detection_and_descovery'  # Question1: best parameters

# Logs
# Folder name where the logs were saved (Subfolder in RQ_DIR)
LOGS_DIR = RQ_DIR / 'logs'

# Subfolders of LOGS_DIR:
# Subfolder name where bimp file were saved.
BIMP_LOGS_DIR = LOGS_DIR / 'bimp_and_reference_logs'
# Subfolder name where reference logs were saved
REFERENCE_LOGS_DIR = LOGS_DIR / 'bimp_and_reference_logs'
# Subfolder where will be saved the set of idx of every fold
LOG_ITERACOES_CV = LOGS_DIR / 'iteracoescv_idx'
# Subfolder where one hot encoding representations will be saved (used in autoencoder)
OHE_DIR = LOGS_DIR / 'one_hot_encoding'
# Subfolder where tstideplus representations will be saved (used in tstideplus)
TSTIDE_DIR = LOGS_DIR / 'stideplus_input'
# Subfolder where tstideplus representations will be saved (used in aalst_approach)
AALST_DIR = LOGS_DIR / 'aalst_input'

# Folder name where results will be saved (Subfolder in RQ_DIR)
RESULTS_DIR = RQ_DIR / 'resultados'

# Folder names where will be saved results about training and testing
OUT_DIRNAME = 'out_tr_tst'

# Folder name where results of phase 03 (Analyze approaches to process discovery improvement) will be saved
BATCHES_FOLDER_NAME = 'logs_postproces_batch'

# Sub folders of BATCHES_FOLDER_NAME:
# SubFolder where will be saved the predicted label for each case
DETECTION_DIR_NAME = 'detection_logofcases'
# SubFolder where will be saved the filtered logs (logs where all anomalous cases where deleted)
FILTERED_DIR_NAME = 'filtered_logs'
# SubFolder where will be saved the original log indicating label of predictions
LABELED_DIR_NAME = 'labeled_logs'

# folder where will be saved analysis and statistics results
STATS_ANALYSIS_DIR_NAME = 'stats'
QUALITY_ANALYSIS_DIR_NAME = 'quality_analysis'

# name of files where will be saved the results
ALL_RESULTS_DETAILS_END_STRING = 'results_details.csv'

# total number of approaches
NR_OF_APPROACHES = 6
