#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:20:51 2022
This script compares the fitted parameters with the NM and with the logistic regression.

@author: caroline
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy
import argparse
import sys

computer = os.uname().nodename
parser = argparse.ArgumentParser(description='Parameter recovery')
parser.add_argument('-computer', type=str, default=computer,
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

from PathsForContextPaper import define_paths_for_context_paper
PATHS = define_paths_for_context_paper(computer)


#%% DATA
# Subjects results
concatenated_subjects_results = {'exp1': pd.read_csv(PATHS['exp1_data'],
                                                     index_col=0, engine='python'),
                                 'exp3': pd.read_csv(PATHS['exp3_data'],
                                                     index_col=0, engine='python')}
for exp in concatenated_subjects_results.keys():
    concatenated_subjects_results[exp].rename({'key': 'subject_choices'}, axis=1, inplace=True)
    concatenated_subjects_results[exp].index = concatenated_subjects_results[exp]['participant_ID']
# Weights in logistic regression
logreg_weights_folder = PATHS['LogregWeightsFolder']
logreg_weights = {
    'fitted_parameters_from_complete_bayesian_model':
        {'exp1': pd.read_csv(logreg_weights_folder
                             + '/exp1_ideal_observer_with_generative_priors_all_levels.csv',
                             index_col=0, engine='python'),
         'exp3': pd.read_csv(logreg_weights_folder
                             + '/exp3_ideal_observer_with_fitted_decision_and_learning_parameters_all_levels.csv',
                             index_col=0, engine='python')},
    'fitted_parameters_from_complete_linear_model':
        {'exp1': pd.read_csv(logreg_weights_folder
                             + '/exp1_linear_ideal_observer_with_generative_priors_all_levels.csv',
                             index_col=0, engine='python'),
         'exp3': pd.read_csv(logreg_weights_folder
                             + '/exp3_linear_ideal_observer_with_fitted_decision_and_learning_parameters_all_levels.csv',
                             index_col=0, engine='python')},
    'fitted_parameters_from_best_bayesian_model':
        {'exp3': pd.read_csv(logreg_weights_folder
                             + '/exp3_ideal_observer_with_fitted_decision_parameters_and_volatility_all_levels.csv',
                             index_col=0, engine='python')},
    'fitted_parameters_from_bayesian_model_with_opt_priors':
        {'exp1': pd.read_csv(logreg_weights_folder
                             + '/exp1_ideal_observer_with_generative_priors_all_levels.csv',
                             index_col=0, engine='python'),
         'exp3': pd.read_csv(logreg_weights_folder
                             + '/exp3_ideal_observer_with_optimal_parameters_all_levels.csv',
                             index_col=0, engine='python')},
    'fitted_parameters_from_linear_model_with_opt_priors':
        {'exp1': pd.read_csv(logreg_weights_folder
                             + '/exp1_linear_ideal_observer_with_generative_priors_all_levels.csv',
                             index_col=0, engine='python'),
         'exp3': pd.read_csv(logreg_weights_folder
                             + '/exp3_linear_ideal_observer_with_fitted_linear_decision_and_optimal_learning_parameters_all_levels.csv',
                             index_col=0, engine='python')}}

# Parameters in the NM

NM_weights = {
    'fitted_parameters_from_complete_bayesian_model':
        {'exp1': pd.read_csv(PATHS['SubjectsFitResults']
                             + '/Gorilla_V4/exp1_parameters_fits_fit_bayesian_param_nov22_best_fit.csv',
                             index_col=0, engine='python'),
         'exp3': pd.read_csv(PATHS['SubjectsFitResults']
                             + '/Gorilla_V4/exp3_parameters_fits_fit_bayesian_param_dec22_best_fit.csv',
                             index_col=0, engine='python')},
    'fitted_parameters_from_complete_linear_model':
        {'exp1': pd.read_csv(PATHS['SubjectsFitResults']
                             + '/Gorilla_V4/exp1_parameters_fits_fit_linear_param_nov22_best_fit.csv',
                             index_col=0, engine='python'),
         'exp3': pd.read_csv(PATHS['SubjectsFitResults']
                             + '/Gorilla_V4/exp3_parameters_fits_fit_linear_param_dec22_best_fit.csv',
                             index_col=0, engine='python')},
    'fitted_parameters_from_best_bayesian_model':
        {'exp3': pd.read_csv(PATHS['SubjectsFitResults']
                             + '/Gorilla_V4/exp3_parameters_fits_fit_bayesian_param_bestM_dec22_best_fit.csv',
                             index_col=0, engine='python')},
        }


eq_param = {'priors_weight': 'fit_prior_weight',
            'likelihood_weight': 'fit_lik_weight',
            'response_bias': 'fit_resp_bias'}

colors = {'priors_weight': 'skyblue',
          'fit_prior_weight': 'skyblue',
          'likelihood_weight': 'salmon',
          'fit_lik_weight': 'salmon',
          'response_bias': 'grey',
          'fit_resp_bias': 'grey'}

#%% Correlate weights computed in the logistic regression with weights computed in the NM
for model_type in ['fitted_parameters_from_complete_linear_model',
                   'fitted_parameters_from_best_bayesian_model',
                   'fitted_parameters_from_complete_bayesian_model']:
    parameters = {exp: pd.DataFrame(
        index=concatenated_subjects_results['exp3']['participant_ID'].unique())
                  for exp in ['exp1', 'exp3']}
    exps = ['exp1', 'exp3'] if model_type == 'fitted_parameters_from_complete_bayesian_model'\
        else ['exp3']
    for exp in exps:
        exp_logreg_weights = logreg_weights[model_type][exp].loc[:, eq_param.keys()].dropna()
        parameters[exp].loc[exp_logreg_weights.index, eq_param.keys()] = exp_logreg_weights
        exp_NM_weights = NM_weights[model_type][exp].loc[:, eq_param.values()].dropna()
        parameters[exp].loc[exp_NM_weights.index, eq_param.values()] = exp_NM_weights
        for param in eq_param.keys():
            plt.figure()
            plt.plot(parameters[exp][param], parameters[exp][eq_param[param]], '.',
                     color=colors[param], label=model_type)
            plt.xlabel('Logistic regression')
            plt.ylabel('Nelder Mead')
            plt.title(f'{exp} {param}')
            plt.legend()
#%% Correlate between experiments weights computed with the NM 
for param in eq_param.keys():
    plt.figure()
    weights = pd.concat([parameters['exp1'][eq_param[param]],
                         parameters['exp3'].loc[parameters['exp1'][eq_param[param]].index,
                                                eq_param[param]]], axis=1)
    weights.dropna(inplace=True)
    weights.columns = ['exp1', 'exp3']
    rho, pvalue = scipy.stats.spearmanr(weights['exp1'].values, weights['exp3'].values)
    weights = weights[weights < 20]
    weights = weights[weights > -10]
    plt.plot(weights['exp1'], weights['exp3'], '.',
             color=colors[param], label=f'rho={rho:.2f}   pval={pvalue:.2f}')
    plt.xlabel('Explicit context')
    plt.ylabel('Implicit context')
    plt.title(param)
    plt.legend()

#%% Correlate weights computed with a linear or a bayesian model
linweights = logreg_weights['fitted_parameters_from_linear_model_with_opt_priors']
bayweights = logreg_weights['fitted_parameters_from_bayesian_model_with_opt_priors']

for exp in ['exp1', 'exp3']:
    for param in ['priors_weight', 'likelihood_weight', 'response_bias']:
        plt.figure()
        weights = pd.concat([bayweights[exp].dropna(subset=['priors_weight'])[param],
                             linweights[exp].dropna(subset=['priors_weight'])[param]], axis=1)
        weights.dropna(inplace=True)
        weights.columns = ['bayesian', 'linear']
        rho, pvalue = scipy.stats.spearmanr(weights['bayesian'].values, weights['linear'].values)
        weights = weights[weights < 20]
        weights = weights[weights > -10]
        plt.plot(weights['bayesian'], weights['linear'], '.',
                 color=colors[param], label=f'rho={rho:.2f}   pval={pvalue:.2f}')
        plt.xlabel('Bayesian model')
        plt.ylabel('Linear model')
        plt.title(f'{param} in {exp}')
        plt.legend()
