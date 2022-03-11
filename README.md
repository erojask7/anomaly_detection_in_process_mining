# Anomaly Detection In Process Mining
This is the repository for "Anomaly detection in business process event logs: a comparative study between neural networks-based and count-based approaches"
## Folder estructures
```
├── anomaly_detection 
├── research_questions
│   ├── best_parameters 
│   ├── ├── logs 
│   ├── ├── resultados
│   ├── best_anomaly_detection_and_descovery 
│   ├── ├── logs 
│   ├── ├── resultados
├── LICENSE 
└── README.md 
```

The main folders are:
- **anomaly_detection** folder containing all code
- **research_questions**. At the beginning, this folder only contains logs. After execution, this folder will contain results of each phase of research


## Guide to use
### Steps
1. Install Anaconda 
2. Install R (Some R statistics functions are necessary for this python code)
3. Create a conda environment using env_pyth3.8.yml ( made available in repository)

    `conda env create -f env_pyth3.8.yml`

    `conda activate env_pyth3.8.yml`
4. Execute jupyter 

    `jupyter notebook notebooks`
5. Execute notebook 

### Phases
We have three main phases in the research:
1. Look for the best parameterization
2. Analyze robustness of approaches
3. Analyze approaches to process discovery improvement
 
### Automatizar.py
The automatizar.py file contains the main code.
In order to execute phase 02 we can execute the following code in automatizar.py:

`main(phase_number=2)`

main() function was set using nome_grupo_experimentos = "test_case". In order to get the same 
results as in dissertation, nome_grupo_experimentos = "relatorio1-question2" should be used.

## Notebooks related to phase 02
### Notebook01: 
 - It executes phase02 and save results in folder "anomaly_detection\research_questions\best_anomaly_detection_and_descovery\resultados"
 - Results of experiment group [id_group] will be saved in "anomaly_detection\research_questions\best_anomaly_detection_and_descovery\resultados\resultados_gexp_binetn_[id_group]\gexp_binetn_[id_group]_all_results_details".  
 - A report in .csv will be saved in gexp_binetn_[id_group]_all_results_details.csv. That file include all information stored about experiments
 

