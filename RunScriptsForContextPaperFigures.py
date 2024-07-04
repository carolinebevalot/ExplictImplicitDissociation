#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:17:37 2022
This script calls every script necessary to analyse the differences between contexts.
PATHS['ScriptsPathRoot'] should correspond to the GitHub folder of the Bayesian Battery.
If not, it can be changed in the first lines of this script.

Data files should be in a folder called 'BayesianBattery_complements'
The ones required for the analyses are the following :
* DataFrame with answers from all subjects
    exp1_results : PATHS['exp1_data']
    exp3_results : PATHS['exp3_data']
 and dataframe with prior reports in T1
    report_df : PATHS['exp1_priorreport']

* Models parameters fitted to subjects' behavior with the Nelder-Mead algorithm
  generated with ModelBasedFitting.py:
        'exp3': PATHS['NM_FittedParameters']

* Logistic weights of priors, sensory likelihood and posteriors in subjects' answers
  computed with ComputeWeightsInLogisticRegression.py:
    Files in folder PATHS['LogregWeightsFolder'] (depending on the model, the type of priors and the evidence level)

* We can check whether fits are equivalent with the Nelder Mead or the Logistic Regression
  with CorrelateNMLogRegWeights.py

* List of subjects included in analysis
  generated with ComputeSubjectsLists_ForContextPaper.py:
    PATHS['SubjectsList_ErraticAnswers']

* Results of model testing :
    - Parameter recovery of BAYES-FIT-ALL model (computed with ParameterRecoveryAllSubjectsLists)
        PATHS['ParameterRecoveryResults']
    - Comparison of the Bayesian models (computed with MdoelCrossValidation.py)
        PATHS['ModelCrossValidation_folder']
    - Identifiability of the BAYES-FIT-ALL and BAYES-OPTIMAL models
        (computed with ModelIdentifiability_BayesModels.py)
        PATHS['ModelIdentifiability_BFA_BO_C']
    - Identifiability of the BAYES-FIT-DECISION and HEURISTIC models
        (computed with ModelCrossValidation.py )
        PATHS['ModelIdentifiability_BFD_H']
    cross validation of the different models
  computed with ModelCrossValidation.py
      'exp1': PATHS['ModelCrossValidation_exp1']
      'exp3': PATHS['ModelCrossValidation_exp3']

* Results of the model identifiability:
    Bayesian models: PATHS['ModelIdentifiability_BFA_BO_C']
    Bayesian vs. Heuristic models: PATHS['ModelIdentifiability_BFD_H']

* Results of parameter recovery of the Bayesian models

* Recovery
These first analysis steps can be processed again if run_logweights_computations
(to verify logistic weights) and run_model_comparison (to verify model cross validation) are set to True.
Be careful before changing these parameters, these steps are time consumming (hours and days respectively).

@author: caroline
"""

import os
import subprocess
import sys

computer = os.uname().nodename
if computer == 'beluga':
    working_dir = '/home_local/BayesianBattery/GitHub_BB/BayesianBattery'
    data_dir = '/neurospin/unicog/protocols/comportement/BayesianBattery_Bevalot'
    sys.path.append(working_dir)
    sys.path.append(data_dir)
elif computer == 'caroline-Latitude-5300':
    working_dir = '/home/caroline/Desktop/BayesianBattery/'
    data_dir = '/home/caroline/Desktop/BayesianBattery_complements/'
    sys.path.append(working_dir)
    sys.path.append(data_dir)
else:
    print('Please provide your paths in PathsForContextPaper.py')
os.chdir(working_dir)
print(os.getcwd())

from Analysis.AnalysisOfGorrillaData.VariablesForContextPaperAnalysis\
    import SIMULATION_OPTIONS_BOTH_EXP, SUBJECT_GROUP, EXPERIMENTS
from PathsForContextPaper import define_paths_for_context_paper
PATHS = define_paths_for_context_paper(computer)
print(PATHS['ScriptsPathRoot'])

save_figures = 'True'
parameters_date = '20230212'
run_preliminary_analysis = False  # /!\ SEVERAL HOURS TO BE COMPUTED
run_logistic_regressions_in_fig2 = False
run_model_comparison = False  # /!\ SEVERAL DAYS TO BE COMPUTED

#%% SCRIPTS TO RUN THE COMPUTATION OF LOGISTIC WEIGHTS OR OF THE MODELS COMPARISON
if run_preliminary_analysis:
    # Recovery analysis
    script = os.path.join(PATHS['ScriptsPathRoot'],
                          "ParameterRecoveryAllSubjectsLists.py")
    subprocess.call(["python3", script,
                     '-BayesianBattery_path', working_dir,
                     '-BayesianBattery_completements_path', data_dir,
                     '-nb_of_optimization_restart', 50,
                     '-total_nb_of_simulations', 10,  # 10*280 subjects
                     '-model_type', 'bayesian_model_of_decision_and_learning'])

    # Fit subjects' free parameters
    script = os.path.join(PATHS['ScriptsPathRoot'],
                          "ModelBasedFitting.py")
    subprocess.call(["python3", script,
                     '-BayesianBattery_path', working_dir,
                     '-BayesianBattery_completements_path', data_dir,
                     '-nb_of_optimization_restart', 50,
                     '-model_class', ['bayesian', 'linear']])

    # Infer subjects' priors and posteriors with the fitted parameters
    script = os.path.join(PATHS['ScriptsPathRoot'],
                          "Analysis",
                          "AnalysisOfGorillaData",
                          "AddIdealObserverToConcatResults_ForContextPaper.py")
    subprocess.call(["python3", script,
                     '-BayesianBattery_path', working_dir,
                     '-BayesianBattery_completements_path', data_dir])

    # Compute weights of priors, posteriors and likelihood with a logistic regression
    script = os.path.join(PATHS['ScriptsPathRoot'],
                          "Analysis",
                          "AnalysisOfGorillaData",
                          "ComputeWeightsInLogisticRegression_ForContetPaper.py")
    subprocess.call(["python3", script,
                     '-BayesianBattery_path', working_dir,
                     '-BayesianBattery_completements_path', data_dir,
                     '-experiments', EXPERIMENTS,
                     '-save_results', False,
                     '-simulation_options_both_exp', SIMULATION_OPTIONS_BOTH_EXP])

    # Check the correspondance of weights computed with the NM and the logistic regression
    script = os.path.join(PATHS['ScriptsPathRoot'],
                          "Analysis",
                          "AnalysisOfGorillaData",
                          "CorrelateNMLogRegWeights.py")
    subprocess.call(["python3", script,
                     '-BayesianBattery_path', working_dir,
                     '-BayesianBattery_completements_path', data_dir])

    # List of subjects included in analysis
    script = os.path.join(PATHS['ScriptsPathRoot'], "Analysis", "AnalysisOfGorillaData",
                          "ComputeSubjectsLists_ForContextPaper.py")
    subprocess.call(["python3", script,
                     "-computer", computer,
                     '-BayesianBattery_path', working_dir,
                     '-BayesianBattery_completements_path', data_dir])

    # Logistic weights of explicited and implicit priors on trials before of after reports
    script = os.path.join(PATHS['ScriptsPathRoot'],
                          "Analysis",
                          "AnalysisOfGorillaData",
                          "ComputeWeightsInLogisticRegressionAroundExplicitReports.py")
    subprocess.call(["python3", script,
                     '-BayesianBattery_path', working_dir,
                     '-BayesianBattery_completements_path', data_dir,
                     "-computer", computer,
                     '-experiments', EXPERIMENTS,
                     '-save_results', False,  # set to True to run computations of logistic weights
                     '-simulation_options_both_exp', SIMULATION_OPTIONS_BOTH_EXP])

if run_model_comparison:
    # Recovery and identifiability of BAYES-FIT-DECISION vs. HEURISTIC model
    script = os.path.join(PATHS['ScriptsPathRoot'],
                          "ModelCrossValidation.py")
    subprocess.call(["python3", script,
                     '-BayesianBattery_path', working_dir,
                     '-BayesianBattery_completements_path', data_dir,
                     "-njobs", -4,
                     '-nb_of_optimization_restart', 50,
                     '-type_of_models_to_test', ['bayesian', 'linear']])

    # Recovery and identifiability of BAYES-FIT-ALL vs. BAYES-OPTIMAL vs. CHANCE model
    script = os.path.join(PATHS['ScriptsPathRoot'],
                          "ModelIdentifiability_BayesModels.py")
    subprocess.call(["python3", script,
                     '-computer', computer,
                     '-BayesianBattery_path', working_dir,
                     '-BayesianBattery_completements_path', data_dir,
                     '-nb_of_simulations_by_model', 100,
                     '-nb_of_optimization_restart', 5,
                     '-run_simulations', 'True',
                     '-type_of_models_to_test', ['bayesian', 'linear']])

    # Recovery and identifiability of BAYES-FIT-DECISION vs. HEURISTIC vs. CHANCE model
    script = os.path.join(PATHS['ScriptsPathRoot'],
                          "ModelIdentifiability_BayesModels.py")
    subprocess.call(["python3", script,
                     '-computer', computer,
                     '-BayesianBattery_path', working_dir,
                     '-BayesianBattery_completements_path', data_dir,
                     '-nb_of_simulations_by_model', 1000,
                     '-run_simulations', 'True'])

#%% CREATE FIGURES AND RESULTS
# FIGURE 1 : EXPERIMENTAL DESIGN AND MODEL
# SUPPL. FIGURE 1 : EXCLUSION CHART BASED ON

## FIGURE 1.B : GENERATIVE PRIORS IN AN EXAMPLE SESSION
script = os.path.join(PATHS['ScriptsPathRoot'],
                      "Analysis",
                      "AnalysisOfGorillaData",
                      "PriorUpdateFigureGenerativePriorsInExampleSession.py")
subprocess.call(["python3", script,
                 '-BayesianBattery_path', working_dir,
                 '-BayesianBattery_completements_path', data_dir,
                 "-computer", computer,
                 '-save_figures', save_figures])

# FIGURE 1.C : LOG-ODDS FIGURE
script = os.path.join(PATHS['ScriptsPathRoot'],
                      "Analysis",
                      "AnalysisOfGorillaData",
                      "LogOddFigure.py")
subprocess.call(["python3", script,
                 '-BayesianBattery_path', working_dir,
                 '-BayesianBattery_completements_path', data_dir,
                 "-computer", computer,
                 '-experiments', EXPERIMENTS,
                 '-behaviors_to_plot', ['subject_choices'],
                 '-exclude_subjects_on_R2', True,
                 '-plot_sgimoid_logodd', False,
                 '-save_figures', save_figures])


# FIGURE 2.B (and S2) : UPDATE OF PRIORS FIGURE
script = os.path.join(PATHS['ScriptsPathRoot'],
                      "Analysis",
                      "AnalysisOfGorillaData",
                      "PriorUpdateFigure.py")
subprocess.call(["python3", script,
                 '-BayesianBattery_path', working_dir,
                 '-BayesianBattery_completements_path', data_dir,
                 "-computer", computer,
                 "-timestep", 1,
                 '-behaviors_to_plot', ['subject_choices',
                                        'ideal_observers_with_optimal_parameters_choices'],
                 '-run_logistic_regressions', run_logistic_regressions_in_fig2,
                 '-save_figures', save_figures])


# HISTOGRAMS AND CORRELATIONS OF PRIOR, LIKELIHOOD AND POSTERIORS WEIGHTS
### FIGURE 3 : With generative priors in T1 and fitted priors in T2
### FIGURE S6 : With different prior types and sensory evidence levels
script = os.path.join(PATHS['ScriptsPathRoot'],
                      "Analysis",
                      "AnalysisOfGorillaData",
                      "HistogramsAndCorrelationsOfLogisticRegressionWeights_SplitViolinPlots.py")
subprocess.call(["python3", script,
                 '-BayesianBattery_path', working_dir,
                     '-BayesianBattery_completements_path', data_dir,
                 '-computer', computer,
                 '-save_results', save_figures])

###  Results completing the FIGURE 3 : BOOTSTRAP OF CORRELATION OF PRIOR AND LIKELIHOOD WEIGHTS BETWEEN TASKS
# of the logistic regression of the optimal model on their choices
script = os.path.join(PATHS['ScriptsPathRoot'],
                      "Analysis",
                      "AnalysisOfGorillaData",
                      "BootstrapOfCorrelation.py")
subprocess.call(["python3", script,
                 '-BayesianBattery_path', working_dir,
                 '-BayesianBattery_completements_path', data_dir,
                 '-computer', computer,
                 '-simulation_options_both_exp', SIMULATION_OPTIONS_BOTH_EXP,
                 '-total_nb_of_samplings', 10000,
                 '-nb_of_sampling_by_CPU', 99,
                 '-correlation_method', 'pearson'])


# FIGURE 4 : ANALYSIS OF EXPLICIT REPORTS OF PRIOR VALUES
script = os.path.join(PATHS['ScriptsPathRoot'],
                      "Analysis",
                      "AnalysisOfGorillaData",
                      "HistogramsAndCorrelationsOfLogisticRegressionWeights_SplitViolinPlots_ExplicitedPriors.py")
subprocess.call(["python3", script,
                 '-BayesianBattery_path', working_dir,
                 '-BayesianBattery_completements_path', data_dir,
                 '-computer', computer,
                 '-save_results', save_figures])

# FIGURE 5 (AND FIGURE S5 AND S8): MODEL COMPARISON
script = os.path.join(PATHS['ScriptsPathRoot'],
                      "Analysis",
                      "AnalysisOfGorillaData",
                      "ModelCrossValidationAnalysis.py")
subprocess.call(["python3", script,
                 '-BayesianBattery_path', working_dir,
                 '-BayesianBattery_completements_path', data_dir,
                 "-computer", computer,
                 '-add_optimal_model', True,
                 '-method_to_compute_likelihood_of_subject_choices', 'median_of_fits',
                 '-method_to_compute_model_likelihood_at_group_level', 'median',
                 '-save_figures', save_figures])

# SUPPL FIGURE 1 : ACCURACY OF REPORT
script = os.path.join(PATHS['ScriptsPathRoot'],
                      "Analysis",
                      "AnalysisOfGorillaData",
                      "CorrelateGenerativePriorValuesAndReportedPriorValuesT1.py")
subprocess.call(["python3", script,
                 '-BayesianBattery_path', working_dir,
                 '-BayesianBattery_completements_path', data_dir,
                 '-computer', computer])


# SUPPL FIGURE 3 : PARAMETER RECOVERY
script = os.path.join(PATHS['ScriptsPathRoot'],
                      "Analysis",
                      "AnalysisOfGorillaData",
                      "ParameterRecoveryAllSubjectsListsAnalysis.py")
subprocess.call(["python3", script,
                 '-BayesianBattery_path', working_dir,
                 '-BayesianBattery_completements_path', data_dir,
                 '-computer', computer])

# SUPPL FIGURE 4 : MODEL IDENTIFIABILITY OF THE BAYESIAN MODELS
script = os.path.join(PATHS['ScriptsPathRoot'],
                      "ModelIdentifiability_BayesModels.py")
subprocess.call(["python3", script,
                 '-computer', computer,
                 '-BayesianBattery_path', working_dir,
                 '-BayesianBattery_completements_path', data_dir,
                 '-nb_of_simulations_by_model', 100,
                 '-nb_of_optimization_restart', 5,
                 '-run_simulations', 'False',
                 '-type_of_models_to_test', ['bayesian', 'linear']])


# SUPPL FIGURE 7 :  LOGISTC WEIGHTS IN BOTH TASKS
script = os.path.join(PATHS['ScriptsPathRoot'],
                      "Analysis",
                      "AnalysisOfGorillaData",
                      "ComputeWeightsInLogisticRegression_TaskInteraction.py")
subprocess.call(["python3", script,
                 '-BayesianBattery_path', working_dir,
                 '-BayesianBattery_completements_path', data_dir,
                 '-computer', computer])

# SUPPL FIGURE 9 : MODEL IDENTIFIABILITY OF THE BAYESIAN VS. HEURISTIC MODEL
script = os.path.join(PATHS['ScriptsPathRoot'],
                      "ModelIdentifiability_BayesModels.py")
subprocess.call(["python3", script,
                 '-computer', computer,
                 '-BayesianBattery_path', working_dir,
                 '-BayesianBattery_completements_path', data_dir,
                 '-nb_of_simulations_by_model', 1000,
                 '-run_simulations', 'False'])

# SUPPL FIGURE 10 :  LOGISTC WEIGHTS IN BOTH TASKS
script = os.path.join(PATHS['ScriptsPathRoot'],
                      "Analysis",
                      "AnalysisOfGorillaData",
                      "AnalyseLogisticWeightsBySession.py")
subprocess.call(["python3", script,
                 '-BayesianBattery_path', working_dir,
                 '-BayesianBattery_completements_path', data_dir,
                 '-computer', computer,
                 '-save_results', save_figures])

