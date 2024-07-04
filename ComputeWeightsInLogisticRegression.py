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
from Configurations import define_paths
import pandas as pd
import numpy as np
import statsmodels.api as sm

from SimulationOptimization.VariablesForFit import SUBJECT_GROUP
from Analysis.AnalysisOfGorrillaData.VariablesForContextPaperAnalysis import SIMULATION_OPTIONS_BOTH_EXP

from Analysis.AnalysisOfGorrillaData.AnalysisCommuneFunctions import lo, Z_score, regression_analysis

parser = argparse.ArgumentParser(description='Compute logistic regression of priors, likelihood and posteriors')
parser.add_argument('--computer', type=str, default='laptop',
                    help='Define on which computer the script is run to adapt paths')
parser.add_argument('--save_results', type=bool, default=True,
                    help="""Set to true to save results,
                    default set to False ensure that previous results are not overwritten unintentionally""")
parser.add_argument('--experiments', type=list, default=['exp1', 'exp3'],
                    help='Specify from which experiment you would like to gather results')
parser.add_argument('--simulation_options_both_exp', type=dict, default=SIMULATION_OPTIONS_BOTH_EXP,
                    help='Define which behavior to simlate (prior type, free parameters, model type)')
args = parser.parse_args()
print(args)

PATHS = define_paths(args.computer, SUBJECT_GROUP)

#%%
concatenated_subjects_results\
    = {'exp1': pd.read_csv(PATHS['exp1_results'], index_col=0, engine='python'),
       'exp3': pd.read_csv(PATHS['exp3_results'], index_col=0, engine='python')}

#%% TASK 1 AND TASK 2 : DECISION (+/- LEARNING)

# Fit weights of priors and likelihood in T1
logreg_weights_in_both_exp = {'exp1': [], 'exp3': []}
for exp in ['exp1', 'exp3']:
    simulation_options = args.simulation_options_both_exp[exp]
    ## With bayesian combination (= bayesian decision and generative priors)
    combination_type = 'bayesian'
    ### For each subject and their ideal observer
    ### With generative priors
    for reference_behavior in simulation_options.keys(): # use generative priors 
        for behavior_to_analyse in ['subject_choices', f'{reference_behavior}_choices']:
            #### For generative priors and likelihood   
            prior_type = simulation_options[reference_behavior]['prior_type']
            prior_column_name = simulation_options[reference_behavior]['prior_column_name']
            explicative_variables = [prior_column_name, 'likelihood']
            free_parameters = simulation_options[reference_behavior]['free_parameters']
            print(f'{reference_behavior} : {explicative_variables}')
            for evidence_level in ['all_levels', 'ambiguous', 'obvious']:
                logistic_regression \
                    = regression_analysis(concatenated_subjects_results[exp], explicative_variables, behavior_to_analyse,
                                          prior_type, combination_type, free_parameters, evidence_level)
                logreg_weights_in_both_exp[exp].append(logistic_regression)
            #### For posteriors
            prior_type = simulation_options[reference_behavior]['prior_type']
            posterior_column_name = simulation_options[reference_behavior]['posterior_column_name']
            explicative_variables = [posterior_column_name]
            free_parameters = simulation_options[reference_behavior]['free_parameters']
            print(f'{reference_behavior} : {explicative_variables}')
            evidence_levels = ['all_levels']
            if prior_column_name == 'inferred_priors_with_fitted_decision_and_learning_parameters':
                evidence_levels += ['ambiguous', 'obvious']
            for evidence_level in evidence_levels:
                logistic_regression \
                    = regression_analysis(concatenated_subjects_results[exp], explicative_variables, behavior_to_analyse,
                                          prior_type, combination_type, free_parameters, evidence_level)
                logreg_weights_in_both_exp[exp].append(logistic_regression)

    ## With linear combination (= linear decision and generative priors)
    combination_type = 'linear'
    ### For a subject
    for reference_behavior in simulation_options.keys():
        for behavior_to_analyse in ['subject_choices', f'{reference_behavior}_choices']:
            #### For priors and likelihood
            prior_type = simulation_options[reference_behavior]['prior_type']
            prior_column_name = simulation_options[reference_behavior]['prior_column_name']
            explicative_variables = [prior_column_name, 'likelihood']
            free_parameters = simulation_options[reference_behavior]['free_parameters']
            print(f'{reference_behavior} : {explicative_variables}')
            evidence_level = 'all_levels'
            logistic_regression \
                = regression_analysis(concatenated_subjects_results[exp], explicative_variables, behavior_to_analyse,
                                      prior_type, combination_type, free_parameters, evidence_level)
            logreg_weights_in_both_exp[exp].append(logistic_regression)



#%% SAVE RESULTS OF LOGISTIC REGRESSION
if args.save_results:
    for exp in logreg_weights_in_both_exp.keys():
        logreg_weights_in_both_exp[exp] = pd.concat(logreg_weights_in_both_exp[exp])
        logreg_weights_in_both_exp[exp].to_csv(PATHS[f'LogregWeightsFile_{exp}'].strip('.csv')+'_new.csv')
