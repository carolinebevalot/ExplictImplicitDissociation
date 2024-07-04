#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Fri Apr  8 11:29:57 2022

# @author: caroline

"""
Compute logistic regressions of the weights of priors and sensory likelihood for each subject.

# Fit weights of priors and likelihood in T1
    ## With bayesian combination (= bayesian decision and generative priors)
        ### For subject and ideal observer
            #### For priors and likelihood or posteriors
                ##### for several level of noise (ambiguous, obvious, all_trials)

    ## With linear combination (= linear decision and generative priors)
        ### For subject and ideal observer
            #### For priors and likelihood (posteriors have been inferred with bayesian model)

# Fit weights of priors and likelihood in T2
    ## With bayesian combination (= bayesian learning and bayesian decision)
        ### For subject or bayesian ideal observer
            #### For priors and likelihood or posteriors
                ##### for several type of priors (generative, optimal, fitted_with_complete_model,
                                                    fitted_with_best_model)
                ##### for several level of noise (ambiguous, obvious, all_trials)
    ## With linear combination (= bayesian learning and linear decision)
        ### For subjects and ideal observer (linear model should have a low R2 for IO)
            #### For priors and likelihood (posteriors have been inferred with bayesian model)
                ##### for several type of priors (generative, optimal, fitted_with_complete_model,
                                                    fitted_with_best_model)
"""

import argparse
import pandas as pd
import os
import sys

computer = os.uname().nodename
parser = argparse.ArgumentParser(
    description="Compute the weights of posteriors, priors, likelihood in answers")
parser.add_argument('-computer', type=str, default=os.uname().nodename,
                    help='Define on which computer the script is run to adapt paths')
parser.add_argument('-BayesianBattery_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery/',
                    help='Define the path for scripts directory')
parser.add_argument('-BayesianBattery_completements_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery_complements/',
                    help='Define the path for the data and results')

args = parser.parse_args()
sys.path.append(args.BayesianBattery_path)
sys.path.append(args.BayesianBattery_completements_path)
os.chdir(args.BayesianBattery_path)
print(os.getcwd())

from Analysis.AnalysisOfGorrillaData.VariablesForContextPaperAnalysis \
    import SIMULATION_OPTIONS_BOTH_EXP, PENALIZATION_SUFFIX, MODEL_OPTIONS_BOTH_EXP, COLORS_FOR_TASK_COMPARISON
from Analysis.AnalysisOfGorrillaData.AnalysisCommonFunctions import regression_analysis
from PathsForContextPaper import define_paths_for_context_paper

PATHS = define_paths_for_context_paper(computer)

concatenated_subjects_results\
    = {'exp1': pd.read_csv(PATHS['exp1_data'], index_col=0, engine='python'),
       'exp3': pd.read_csv(PATHS['exp3_data'], index_col=0, engine='python')}

pen_suffix = PENALIZATION_SUFFIX
subjects_to_include_both_exp = pd.read_csv(PATHS['SubjectsList_ErraticAnswers'],
                                           index_col=0)['participant_ID'].dropna().to_numpy()

#%% TASK 1 AND TASK 2 : DECISION (+/- LEARNING)
# Fit weights of priors and likelihood in T1
logreg_weights_in_both_exp = {'exp1': {}, 'exp3': {}}
for exp in ['exp1', 'exp3']:
    simulation_options = {key: value
                          for key, value in MODEL_OPTIONS_BOTH_EXP[exp].items()}

    ### With generative, optimal and fitted priors
    for reference_behavior in simulation_options.keys():
        ## With bayesian combination (= bayesian decision)
        combination_type = simulation_options[reference_behavior]['combination_type']

        print(reference_behavior, combination_type)
        behavior_to_analyse = 'subject_choices'
        logreg_weights_one_model = {level: [] for level in
                                    ['all_levels', 'ambiguous', 'obvious']}

        prior_type = simulation_options[reference_behavior]['prior_type']
        prior_column_name = simulation_options[reference_behavior]['prior_column_name']
        posterior_column_name \
            = simulation_options[reference_behavior]['posterior_column_name_opt_decision']\
            if 'posterior_column_name_opt_decision' in simulation_options[reference_behavior].keys()\
            else simulation_options[reference_behavior]['posterior_column_name']
        free_parameters = simulation_options[reference_behavior]['free_parameters']
        evidence_levels = ['all_levels']
        if ((prior_column_name == 'inferred_priors_with_fitted_decision_and_learning_parameters')
            or ((exp == 'exp1') and not('linear' in reference_behavior))):
            evidence_levels += ['ambiguous', 'obvious']

        #### For priors and likelihood
        explicative_variables = [prior_column_name, 'likelihood']
        print(f'{reference_behavior} : {explicative_variables}')
        for evidence_level in evidence_levels:
            logistic_regression \
                = regression_analysis(concatenated_subjects_results[exp],
                                      explicative_variables, behavior_to_analyse,
                                      prior_type, combination_type, free_parameters, evidence_level)
            logreg_weights_one_model[evidence_level].append(logistic_regression)

        #### For posteriors
        if reference_behavior not in ['linear_model_with_interaction_with_generative_priors',
                                      'linear_model_without_interaction_with_generative_priors',
                                      'linear_model_with_interaction_with_optimal_priors',
                                      'linear_model_without_interaction_with_optimal_priors']:
            explicative_variables = [posterior_column_name]
            print(f'{reference_behavior} : {explicative_variables}')
            for evidence_level in evidence_levels:
                logistic_regression \
                    = regression_analysis(concatenated_subjects_results[exp],
                                          explicative_variables, behavior_to_analyse,
                                          prior_type, combination_type,
                                          free_parameters, evidence_level)
                logreg_weights_one_model[evidence_level].append(logistic_regression)

        for evidence_level in evidence_levels:
            logreg_weights_in_both_exp[exp][f'{reference_behavior}_{evidence_level}']\
                = pd.concat(logreg_weights_one_model[evidence_level])

#%% SAVE RESULTS OF LOGISTIC REGRESSION
for exp in logreg_weights_in_both_exp.keys():
    for logreg_type in logreg_weights_in_both_exp[exp].keys():
        folder = PATHS['LogregWeightsFolder']
        logreg_weights_in_both_exp[exp][logreg_type].to_csv(
            f'{folder}/{exp}_{logreg_type}_{pen_suffix}.csv')

