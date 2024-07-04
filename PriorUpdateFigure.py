#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Compute the logistic weights of priors and sensory likelihood at each trial around the change points.

This script gather, for each subject, trials with the same distance to a change point
and compute for each distance, a logistic regression of the sensory likelihood and
the generative priors before or after the change point.
The weights given to priors and sensory likelihood are saved in csv files.
If this step has already been done by a previous call to this script, it can be
skipped with the argument --run_logistic_regressions set to False.
The script then plots, for each distance, the mean weight given to prior values
and sensory likelihood with its standard deviation as a line.
Ideal behavior is also plot as a scatter plot. It can be removed if --behavior_to_plot
only contains subjects_choices.
The error plotted is the s.e.m.
Figures can be saved under svg and png format with the argument --save_figures
set to True.

@author: caroline
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import statsmodels.api as sm
import os
import sys
import seaborn as sns

computer = os.uname().nodename
parser = argparse.ArgumentParser(description="Create a frame with summaries of subjects' results")
parser.add_argument('--computer', type=str, default='laptop',
                    help='Define on which computer the script is run to adapt paths')
parser.add_argument('-BayesianBattery_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery/',
                    help='Define the path for scripts directory')
parser.add_argument('-BayesianBattery_completements_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery_complements/',
                    help='Define the path for the data and results')

parser.add_argument('--timestep', type=int, default=1,
                    help='Number of trials to bin together when plotting variables around change point (do not affect regression)')
parser.add_argument('--behaviors_to_plot', type=list,
                    default=['ideal_observers_with_optimal_parameters_choices', 'subject_choices'],
                    help='Define which behavior to plot (subjects or ideal observers')
parser.add_argument('--run_logistic_regressions', type=bool, default=False,
                    help='Define whether to run logistic regressions or whether to use computations saved in computer')

parser.add_argument('--save_figures', type=bool, default='True',
                    help='Define whether to save figures')
args = parser.parse_args()
sys.path.append(args.BayesianBattery_path)
sys.path.append(args.BayesianBattery_completements_path)
os.chdir(args.BayesianBattery_path)
print(os.getcwd())

from Analysis.AnalysisOfGorrillaData.VariablesForContextPaperAnalysis\
    import THETA, PRIOR_UPDATE_COLORS
from BatteryScript.BayesianBatteryVariables_stimgeneration \
    import NB_OF_TRIALS, MAX_JITTER_ON_NB_OF_TRIALS
from Analysis.AnalysisOfGorrillaData.AnalysisCommonFunctions import lo
from PathsForContextPaper import define_paths_for_context_paper

PATHS = define_paths_for_context_paper(computer)

MIN_LEN_OF_BLOCK = NB_OF_TRIALS - MAX_JITTER_ON_NB_OF_TRIALS
EXP = 'exp3'
plot_figS2 = False
subjects_to_include_both_exp = pd.read_csv(PATHS['SubjectsList_ErraticAnswers'],
                                           index_col=0)['participant_ID'].dropna().to_numpy()
concatenated_subjects_results = pd.read_csv(PATHS['exp3_data'], index_col=0)

#%% FUNCTIONS

def incremental_trial_count_within_block(block_number_array):
    padded_list = np.hstack(['pad', block_number_array])
    count = []
    for id_before, id_now in zip(padded_list[0:-1], padded_list[1:]):
        if id_before != id_now:
            count.append(1)
        else:
            count.append(count[-1]+1)
    return np.array(count)

#%% COMPUTE POSITION OF TRIALS RELATIVELY TO THE CHANGE POINT AND THE PRIOR VALUES
# Get relative prosition with respect to change point
# NB: here we use the fact that the last prior of one subject is never the first prior of the next subject
concatenated_subjects_results['trial_nb_after_change'] \
    = incremental_trial_count_within_block(concatenated_subjects_results['block_nb'].to_numpy())

concatenated_subjects_results['trial_nb_before_change'] \
     = -np.flip(incremental_trial_count_within_block(np.flip(concatenated_subjects_results['block_nb'].to_numpy())))


# Get the previous-block prior value
# Here we use the fact that all subjects got the same sequence of priors
previous_block_prior_value_given_block_nb \
    = {block_name: np.hstack([np.nan, THETA[EXP]])[i]
       for i, block_name in enumerate(concatenated_subjects_results['block_nb'].unique())}
concatenated_subjects_results['previous_block_prior_value'] = np.nan
for block_id in concatenated_subjects_results['block_nb'].unique():
    concatenated_subjects_results['previous_block_prior_value'][concatenated_subjects_results['block_nb'] == block_id]\
         = previous_block_prior_value_given_block_nb[block_id]

# Get the next-block prior value
# Here we use the fact that all subjects got the same sequence of priors
next_block_prior_value_given_block_nb \
    = {block_name: np.hstack([THETA[EXP][1:], np.nan])[i]
       for i, block_name in enumerate(concatenated_subjects_results['block_nb'].unique())}
concatenated_subjects_results['next_block_prior_value'] = np.nan
for block_id in concatenated_subjects_results['block_nb'].unique():
    concatenated_subjects_results['next_block_prior_value'][concatenated_subjects_results['block_nb'] == block_id]\
        = next_block_prior_value_given_block_nb[block_id]

# Select the part of the dataframe that we need
data = concatenated_subjects_results[[
    'subject_choices',
    'ideal_observers_with_optimal_parameters_choices',
    'ideal_observers_with_fitted_decision_and_learning_parameters_choices',
    'trial_nb_after_change',
    'trial_nb_before_change',
    'previous_block_prior_value',
    'next_block_prior_value',
    'prior_values',
    'likelihood']]
data.dropna(inplace=True)
regressors_names = ['prior_before_change', 'prior_after_change', 'likelihood']
regressors_colors = PRIOR_UPDATE_COLORS

#%% COMPUTE THE LOGISTIC REGRESSION
# Compute logistic fit
fit_results = {'subject_choices': {},
               'ideal_observers_with_optimal_parameters_choices': {},
               'ideal_observers_with_fitted_decision_and_learning_parameters_choices': {}}
pos_rel_change_list = list(range(-MIN_LEN_OF_BLOCK, 0)) + list(range(1, MIN_LEN_OF_BLOCK+1))
for pos_rel_change in pos_rel_change_list:
    if pos_rel_change > 0:
        selected_trials = data['trial_nb_after_change'] == pos_rel_change
        regressors = pd.DataFrame({'prior_before_change': lo(data['previous_block_prior_value'][selected_trials]),
                                   'prior_after_change': lo(data['prior_values'][selected_trials]),
                                   'likelihood': lo(data['likelihood'][selected_trials])})
    else:
        selected_trials = data['trial_nb_before_change'] == pos_rel_change
        regressors = pd.DataFrame({'prior_before_change': lo(data['prior_values'][selected_trials]),
                                   'prior_after_change': lo(data['next_block_prior_value'][selected_trials]),
                                   'likelihood': lo(data['likelihood'][selected_trials])})
    max_before_inf = regressors.loc[regressors['likelihood'] != np.inf, 'likelihood'].max()
    regressors['likelihood'].replace(np.inf, max_before_inf, inplace=True)
    min_before_inf = regressors.loc[regressors['likelihood'] != -np.inf, 'likelihood'].min()
    regressors['likelihood'].replace(-np.inf, min_before_inf, inplace=True)
    regressors = sm.add_constant(regressors)

    for dep_var in ['subject_choices', 'ideal_observers_with_optimal_parameters_choices',
                    'ideal_observers_with_fitted_decision_and_learning_parameters_choices']:
        model = sm.Logit(data[dep_var][selected_trials],
                         regressors)
        fit_results[dep_var][pos_rel_change] = model.fit(method='bfgs')


#%% PLOT THE FIGURE

plt.rcParams.update({'font.size': 16})
figsize = (8, 5)
fig = plt.figure(figsize=figsize, dpi=800)
plt.axvspan(-36, 0, facecolor='lightgrey', alpha=0.3)

if plot_figS2:
    behaviors_to_plot = ['subject_choices',
                         'ideal_observers_with_fitted_decision_and_learning_parameters_choices']
else:
    behaviors_to_plot = ['subject_choices', 'ideal_observers_with_optimal_parameters_choices']

for dep_var in behaviors_to_plot:
    for reg_name in regressors_names:
        marker, markersize = ('+', 6) if 'ideal' in dep_var else ('.', 2)
        if dep_var == 'subject_choices':
            plt.plot(pos_rel_change_list,
                 [fit_results[dep_var][pos_rel_change].params[reg_name]
                  for pos_rel_change in pos_rel_change_list],
                 '-', color=regressors_colors[reg_name], label=reg_name)

            plt.plot(pos_rel_change_list,
                 [fit_results[dep_var][pos_rel_change].params[reg_name]
                  for pos_rel_change in pos_rel_change_list],
                 '.', color='white', markersize=5, label=reg_name)

            plt.fill_between(pos_rel_change_list,
                             [fit_results[dep_var][pos_rel_change].conf_int().loc[reg_name, 0]
                              for pos_rel_change in pos_rel_change_list],
                             [fit_results[dep_var][pos_rel_change].conf_int().loc[reg_name, 1]
                              for pos_rel_change in pos_rel_change_list],
                             color=regressors_colors[reg_name], alpha=0.3)
        plt.plot(pos_rel_change_list,
                 [fit_results[dep_var][pos_rel_change].params[reg_name]
                  for pos_rel_change in pos_rel_change_list],
                 marker, markersize=markersize, color=regressors_colors[reg_name], label=reg_name)

    sns.despine()
    plt.ylabel('Weights in logistic regressions')
    plt.xlabel('Trial number relative to change point')
    plt.ylim([-0.3, 1.1])
    plt.xlim([-38, 38])
    plt.title('Logistic regression of choices around change points')

nb_subjects = len(subjects_to_include_both_exp)
if args.save_figures:
    if plot_figS2:
        fig.savefig(op.join(PATHS['ContextPaperFiguresPathRoot'], 'svg', 'SupplementaryFigures',
                    f"PriorUpdateFigure_n{nb_subjects}_BAYESFITALL.svg"), format='svg',
                    bbox_inches='tight', pad_inches=1, dpi=10000)
        fig.savefig(op.join(PATHS['ContextPaperFiguresPathRoot'], 'png', 'SupplementaryFigures',
                    f"PriorUpdateFigure_n{nb_subjects}_BAYESFITALL.png"), format='png',
                    bbox_inches='tight', pad_inches=1, dpi=300)
    else:
        fig.savefig(op.join(PATHS['ContextPaperFiguresPathRoot'], 'svg', 'MainFigures',
                    f"PriorUpdateFigure_n{nb_subjects}.svg"), format='svg',
                    bbox_inches='tight', pad_inches=1, dpi=10000)
        fig.savefig(op.join(PATHS['ContextPaperFiguresPathRoot'], 'png', 'MainFigures',
                    f"PriorUpdateFigure_n{nb_subjects}.png"), format='png',
                    bbox_inches='tight', pad_inches=1, dpi=300)
