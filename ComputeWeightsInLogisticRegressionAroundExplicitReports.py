#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Fri Apr  8 11:29:57 2022
# @author: caroline


"""
This script compute the weights given to reports of priors and implicit priors.

with a logistic regression (considering priors and sensory likelihood).
It select trials after the prior report.
Weights from these different selections are saved in separe files ('afterQ').
Implicit priors correspond to priors inferred with the complete bayesian model fitted to behavior

"""
import argparse
import pandas as pd
import os
import sys
import numpy as np

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
    import SIMULATION_OPTIONS_BOTH_EXP, PENALIZATION_SUFFIX
from Analysis.AnalysisOfGorrillaData.AnalysisCommonFunctions import regression_analysis
from PathsForContextPaper import define_paths_for_context_paper

PATHS = define_paths_for_context_paper(computer)

pen_suffix = PENALIZATION_SUFFIX

exp3_results = pd.read_csv(PATHS['exp3_data'], index_col=0, engine='python')
CONVERT_SCORES_INTO_PROBA = False

#%% FUNCTIONS
def convert_conf_scores_into_proba(subject_data):
    """
    Convert report of prior values into a probability.

    Parameters
    ----------
    subject_data : DataFrame
        DataFrame with subjects' data.

    Returns
    -------
    subject_data : DataFrame
        DataFrame with subjects' data containing the prior reports column.

    """
    min_conf = np.min(subject_data.loc[:, 'conf_scores'])
    max_conf = np.max(subject_data.loc[:, 'conf_scores'])
    print(f'Minimum and maximum confidence are : {min_conf}, {max_conf}')
    if min_conf < 0:
        subject_data.loc[:, 'prior_reports'] \
            = [(conf_score+100)/200 for conf_score in subject_data.loc[:, 'conf_scores']]
    else:
        subject_data.loc[:, 'prior_reports'] \
            = [conf_score/100 for conf_score in subject_data.loc[:, 'conf_scores']]

    return subject_data


#%% CONVERT SUBJECT REPORTS INTO A PROBABILITY IF NOT DONE PREVIOUSLY 
if CONVERT_SCORES_INTO_PROBA:
    for subject in exp3_results.participant_ID.unique():
        subject_results = exp3_results[exp3_results.loc[:, 'participant_ID'] == subject]
        subject_results.dropna(subset=['block_nb'], inplace=True)
        trial_nb_in_exp = [i for i in range(0, len(subject_results))]
        trial_nb_in_block \
            = np.concatenate([np.arange(len(subject_results[subject_results.block_nb == block_name]))
                              for block_name in subject_results.block_nb.unique()])
        exp3_results.loc[subject_results.index, 'trial_nb_in_block'] = trial_nb_in_block
        exp3_results.loc[subject_results.index, 'trial_nb_in_exp'] = trial_nb_in_exp

    exp3_results['prior_reports'] = pd.Series()
    for subject in exp3_results.participant_ID.unique():
        subject_data = exp3_results[exp3_results.loc[:, 'participant_ID'] == subject]
        subject_data = convert_conf_scores_into_proba(subject_data)
        exp3_results.loc[subject_data.index, :] = subject_data
    idx_prior_reports = exp3_results.prior_reports.dropna().index
    last_idx = len(exp3_results) - 1
    if (exp3_results.conf_scores.dropna().index != idx_prior_reports).all():
        print('WARNING UNMATCHED CONF SCORES AND CONF REPORTS')
    exp3_results['report_prior_value'] = 0
    exp3_results.loc[idx_prior_reports, 'report_prior_value'] = 1

    # Interpole results of explicit reports until next report
    # and select the adequate column for implicit priors
    exp3_results['explicited_priors'] = pd.Series()
    exp3_results['implicit_priors'] = pd.Series()
    for idx_prior_report, idx_next_prior_report in zip(idx_prior_reports[:],
                                                       np.append(idx_prior_reports[1:], last_idx)):
        idx_with_explicit_prior = exp3_results.loc[idx_prior_report:idx_next_prior_report,
                                                   'prior_reports'].index
        exp3_results.loc[idx_with_explicit_prior, 'explicited_priors'] \
            = exp3_results.loc[idx_prior_report, 'prior_reports']
    exp3_results.loc[:, 'implicit_priors'] \
        = exp3_results.loc[:, 'inferred_priors_with_fitted_decision_and_learning_parameters']
    exp3_results.to_csv(PATHS['exp3_data'])

#%% DEFINE INDEX OF PRIOR REPORTS
idx_prior_reports = exp3_results['prior_reports'].dropna().index
if (exp3_results.conf_scores.dropna().index != idx_prior_reports).all():
    print('WARNING UNMATCHED CONF SCORES AND CONF REPORTS')

#%% FIT WEIGHTS OF EXPLICITED, IMPLICIT OR GENERATIVE PRIORS WITH LIKELIHOOD IN T2 
# AFTER THE EXPLICIT REPORT
logistic_regression_weights_in_exp3 = {'after_Q': []}

## With bayesian combination (= bayesian learning and bayesian decision)
combination_type = 'bayesian'
for location_label, idx_to_select in [('after_Q', 1)]:
    restrained_exp3_results = exp3_results.loc[idx_prior_reports + idx_to_select, :]
    ### For a subject 
    behavior_to_analyse = 'subject_choices'
    for prior_column_name in ['implicit_priors', 'generative_priors', 'explicited_priors']:
        #### For priors and likelihood
        ##### for several type of priors
        explicative_variables = [prior_column_name, 'likelihood']
        free_parameters = 'fitted_decision_and_learning_parameters'
        print(f'{prior_column_name} : {explicative_variables}')
        evidence_level = 'all_levels'
        logistic_regression \
            = regression_analysis(restrained_exp3_results, explicative_variables,
                                  behavior_to_analyse, prior_column_name,
                                  combination_type, free_parameters, evidence_level,
                                  compute_correlation_explicit_report=True)
        logistic_regression_weights_in_exp3[location_label].append(logistic_regression)

#%% SAVE RESULTS
for location_label in logistic_regression_weights_in_exp3.keys():
    logistic_regression_weights_in_exp3[location_label] = pd.concat(logistic_regression_weights_in_exp3[location_label])
    logistic_regression_weights_in_exp3[location_label].to_csv(
        os.path.join(PATHS['LogregWeightsFolder'],
                     f"exp3_implicit_explicit_logreg_{location_label}_{pen_suffix}_mod.csv"))

#%% SAME FOR PRIORS BELOW OR ABOVE 0.5 :
# FIT WEIGHTS OF EXPLICITED OR IMPLICIT WITH LIKELIHOOD IN T2
# AFTER THE EXPLICIT REPORT
logistic_regression_weights_in_exp3 = {'pinf05_after_Q': [],
                                       'psup05_after_Q': []}

## With bayesian combination (= bayesian learning and bayesian decision)
combination_type = 'bayesian'
for prior_half in ['pinf05', 'psup05'] :
    for location_label, idx_to_select in [('after_Q', 1)]:
        restrained_exp3_results = exp3_results.loc[idx_prior_reports + idx_to_select, :]
        if prior_half == 'pinf05':
            restrained_exp3_results = \
                restrained_exp3_results[restrained_exp3_results['inferred_priors_with_fitted_decision_and_learning_parameters'] < 0.5]
        elif prior_half == 'psup05':
            restrained_exp3_results = \
                restrained_exp3_results[restrained_exp3_results['inferred_priors_with_fitted_decision_and_learning_parameters'] > 0.5]

        ### For a subject 
        behavior_to_analyse = 'subject_choices'
        for prior_column_name in ['implicit_priors', 'explicited_priors']:
            #### For priors and likelihood
            ##### for several type of priors
            explicative_variables = [prior_column_name, 'likelihood']
            free_parameters = 'fitted_decision_and_learning_parameters'
            print(f'{prior_column_name} : {explicative_variables}')
            evidence_level = 'all_levels'
            logistic_regression \
                = regression_analysis(restrained_exp3_results, explicative_variables,
                                      behavior_to_analyse, prior_column_name,
                                      combination_type, free_parameters, evidence_level,
                                      compute_correlation_explicit_report=True)
            logistic_regression_weights_in_exp3[f'{prior_half}_{location_label}'].append(logistic_regression)

#%% SAVE RESULTS
for location_label in logistic_regression_weights_in_exp3.keys():
    logistic_regression_weights_in_exp3[location_label] = pd.concat(logistic_regression_weights_in_exp3[location_label])
    logistic_regression_weights_in_exp3[location_label].to_csv(
        os.path.join(PATHS['LogregWeightsFolder'],
                     f"exp3_implicit_explicit_logreg_{location_label}_{pen_suffix}_mod.csv"))