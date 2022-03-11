#FIX2022forpubrep: ready for publication
import pandas as pd
import sys

sys.path.insert(1, "E:\\EXPERIMENTOS2020\\anomaly_detection_artigo\\")
from anomaly_detection.general.global_variables import *


def create_input_representation_from_logs(logs_path, log_filenames, approach_name):
    import anomaly_detection.pre_processamento.convertir_a_ohe as convertir_a_ohe
    import os

    filenames = []
    for log_filename in log_filenames:
        if not (os.path.isfile(logs_path / log_filename.replace("log", "logofcases"))):  # IF does not exist
            from anomaly_detection.general.utils_da import create_informative_csvs_about_logs
            filenames = [log_filename]
            create_informative_csvs_about_logs(logs_path, filenames)  # Create file

        # Filenames of log of cases
        log_of_cases = pd.read_csv(logs_path / log_filename.replace("log", "logofcases"))

        if (approach_name == 'autoencoder_nolle'):
            # Convert log to one hot encoding and save
            convertir_a_ohe.convertir(log_filename, input_path=logs_path, output_path=OHE_DIR)
            ohe_filename = log_filename.replace("log", "ohe_log")
            filenames.append(ohe_filename)

            # Open one hot encoding created
            ohe_file = pd.read_csv(os.path.join(OHE_DIR, ohe_filename), header=None)

            # Add column traceid in ohe file and save
            ohe_file.iloc[:, -1] = log_of_cases["traceid"]
            ohe_file.to_csv(os.path.join(OHE_DIR, ohe_filename), header=False, index=False)

        if (approach_name == 'tstideplus'):
            from anomaly_detection.general.class_dataset import Dataset

            # Convert log to input representation and save
            dataset = Dataset()
            dataset.num_attributes = 2
            dataset.create_cases_x_activities_order(log_filename, logs_path,
                                                    save=True)

            input_representation_filename = log_filename.replace("log", "tstideplus_log")
            filenames.append(input_representation_filename.replace('.csv', ".pkl"))

            import pickle
            pickle.dump(dataset, open(os.path.join(TSTIDE_DIR, input_representation_filename.replace('.csv', ".pkl")),
                                      "wb"))  # save dataset object

    if (approach_name == 'aalst_approach'):
        import os
        if not (os.path.isdir(AALST_DIR)):  # If folder doesnt exist
            os.mkdir(AALST_DIR)  # Create folder

        import shutil
        import glob
        for filename in glob.glob(os.path.join(OHE_DIR, '*.csv')):
            shutil.copy(filename, AALST_DIR)

    return filenames