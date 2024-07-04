
        
#%%%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:05:43 2022

@author: caroline

This script draw x lists of n subjects with replacement and compute the difference 
between the correlation of prior weights in T1 and T2 
and the correlation of likelihood weights in T1 and T2.
The prior and likelihood weights correspond to the weights in the logistic regression of logodds.
x corresponds to the total number of samplings
n corresponds to the number of subjects included in the analysis.

We can write the computed difference as follow
bootstrapped difference = rho(wPriorT1, wPriorT2) - rho(wLikelihoodT1, wLikelihoodT2)

The script plot by default the pearson coefficient of correlation but can also plot
the spearman coefficient of correlation.
"""
import argparse
import numpy as np
import pandas as pd
import os.path as op
import scipy.stats
import sys
import os
import datetime

computer = os.uname().nodename
parser = argparse.ArgumentParser(description="Compute and plot bootstrap of the difference of correlation between tasks")
parser.add_argument('--computer', type=str, default=computer,
                    help='Define on which computer the script is run to adapt paths')
parser.add_argument('-BayesianBattery_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery/',
                    help='Define the path for scripts directory')
parser.add_argument('-BayesianBattery_completements_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery_complements/',
                    help='Define the path for the data and results')
parser.add_argument('--total_nb_of_samplings', type=int, default=10000,
                    help='Set the total number of samplings in the bootstrap')
parser.add_argument('--correlation_method', type=str, default='spearman',
                    help='Method to compute the correlation (pearson or spearman')
parser.add_argument('--save_results', type=bool, default=True,
                    help='Define whether to save results')

args = parser.parse_args()
sys.path.append(args.BayesianBattery_path)
sys.path.append(args.BayesianBattery_completements_path)
os.chdir(args.BayesianBattery_path)
print(os.getcwd())

from Analysis.AnalysisOfGorrillaData.VariablesForContextPaperAnalysis\
    import SIMULATION_OPTIONS_BOTH_EXP
from PathsForContextPaper import define_paths_for_context_paper

PATHS = define_paths_for_context_paper(computer)

subjects_to_include_both_exp = pd.read_csv(PATHS['SubjectsList_ErraticAnswers'],
                                           index_col=0)['participant_ID'].dropna().to_numpy()
simulation_options_exp3 = SIMULATION_OPTIONS_BOTH_EXP['exp3']

nb_of_subjects = len(subjects_to_include_both_exp)

date = datetime.date.today().isoformat().replace('-', '')
#%% FUNCTIONS
def sample_correlation_difference(values, correlation_type):
    """
    Sample with replacement and compute the intertask correlation.

    Parameters
    ----------
    values : DataFrame
        DataFrame of the logistic weights of priors and sensory likelihood in T1 and T2.
    correlation_type : str
        Define which coefficient to compute (pearson or spearman).

    Returns
    -------
    Bootstrapped difference between the intertask correlation of priors
    and sensory likelihood.
    """

    resample_values = values.sample(n=values.shape[0], axis=0, replace=True)
    if correlation_type == "pearson":
        prior_correlation = scipy.stats.pearsonr(resample_values['prior_T1'],
                                                 resample_values['prior_T2'])[0]
        lik_correlation = scipy.stats.pearsonr(resample_values['lik_T1'],
                                               resample_values['lik_T2'])[0]
    elif correlation_type == "spearman":
        prior_correlation = scipy.stats.spearmanr(resample_values['prior_T1'],
                                                  resample_values['prior_T2'])[0]
        lik_correlation = scipy.stats.spearmanr(resample_values['lik_T1'],
                                                resample_values['lik_T2'])[0]
    else:
        raise ValueError(f"correlation_type is {correlation_type}, it should be pearson or spearman")

    return prior_correlation - lik_correlation

#%% COMPUTE CORRELATIONS


corr_diff_by_prior_type = {}

behavior_to_plot = 'subject_choices'
evidence_level = 'all_levels'
combination_type = 'bayesian'
penalization_suffix = 'pen'

# Refine the selection for T2 with generative priors or priors with optimal or fitted parameters
# fitted with the full or the best model
logreg_weights_folder = PATHS['LogregWeightsFolder']
logreg_weights_exp1 = pd.read_csv(f"{logreg_weights_folder}/exp1_ideal_observer_" +
                                  f"with_generative_priors_all_levels_{penalization_suffix}.csv",
                                  index_col=0)

for prior_parameters_options in [option for option in simulation_options_exp3.keys()
                                 if 'linear' not in option]:
    free_parameters = simulation_options_exp3[prior_parameters_options]['free_parameters']
    prior_column_name = simulation_options_exp3[prior_parameters_options]['prior_column_name']
    file_ID = prior_column_name if 'generative' in prior_parameters_options else free_parameters
    print(file_ID)
    logreg_weights_exp3 \
        = pd.read_csv(f"{logreg_weights_folder}/exp3_ideal_observer_with_" +
                      f"{file_ID}_all_levels_{penalization_suffix}.csv", index_col=0)

    # Concatenate weights together to sample the four subject's weights together
    subjects_weights = pd.concat([logreg_weights_exp1['priors_weight'].dropna(),
                                  logreg_weights_exp3['priors_weight'].dropna(),
                                  logreg_weights_exp1['likelihood_weight'].dropna(),
                                  logreg_weights_exp3['likelihood_weight'].dropna()],
                                 axis=1).dropna()
    subjects_weights.columns = ['prior_T1', 'prior_T2', 'lik_T1', 'lik_T2']

    corr_diff_by_prior_type[prior_column_name] = {
        'true_difference':
            scipy.stats.spearmanr(subjects_weights['prior_T1'], subjects_weights['prior_T2'])[0]
            - scipy.stats.spearmanr(subjects_weights['lik_T1'], subjects_weights['lik_T2'])[0],
        'true_prior_correlation':
            scipy.stats.spearmanr(subjects_weights['prior_T1'], subjects_weights['prior_T2'])[0],
        'true_likelihood_correlation':
            scipy.stats.spearmanr(subjects_weights['lik_T1'], subjects_weights['lik_T2'])[0]
            }

    # Run boostrap
    bootstrap_diff = [sample_correlation_difference(subjects_weights, args.correlation_method)
                      for k in range(args.total_nb_of_samplings)]
    corr_diff_by_prior_type[prior_column_name]['median_bootstrap_difference'] \
        = np.median(bootstrap_diff)
    corr_diff_by_prior_type[prior_column_name]['bootstrap_conf_interval_95'] \
        = np.percentile(bootstrap_diff, [2.5, 97.5])
    corr_diff_by_prior_type[prior_column_name]['bootstrap_conf_interval_99'] \
        = np.percentile(bootstrap_diff, [0.5, 99.5])

    print(f'Done for {prior_column_name}')

if args.save_results:
    pd.DataFrame(corr_diff_by_prior_type).to_csv(op.join(PATHS['ContextPaperFiguresPathRoot'],
                                                 f'bootstrap_of_correlation_difference_{date}.csv'))
