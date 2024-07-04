# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""
Compute log-odds of answers house for the different generative prior values and the different levels of sensory likelihood.

It plots a barplot for the selected experiments (args --experiments).
Whether subjects' behavior or ideal observer behavior or both are plotted can be changed
with the argument --behavior_to_plot.
The script can also plot logodds as a sigmoid with the argument
--plot_sigmoid_logodd set to True
Figures can be saved under svg and png format with the argument --save_figures
set to True.
By default, subjects considered for the fogures are subjects included after
all of the exclusion steps. It can be changed with the argument --exclude_subjects_on_R2
set to False, thus considering the 280 subjects.
The error plotted in the figure is s.e.m.

Created on Tue May  4 08:55:56 2021.
@author: caroline

"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import os
import pandas as pd
from scipy.stats import sem
import seaborn as sns
import sys

computer = os.uname().nodename
parser = argparse.ArgumentParser(description="Create a frame with summaries of subjects' results")
parser.add_argument('--computer', type=str, default=computer,
                    help='Define on which computer the script is run to adapt paths')
parser.add_argument('-BayesianBattery_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery/',
                    help='Define the path for scripts directory')
parser.add_argument('-BayesianBattery_completements_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery_complements/',
                    help='Define the path for the data and results')

parser.add_argument('--behaviors_to_plot', type=list, default=['subject_choices'],
                    help='Define which behavior to plot (subject_choices or ideal_observers_choices or both in the list')
parser.add_argument('--experiments', type=list, default=['exp1', 'exp3'],
                    help='Define which experiment to plot')
parser.add_argument('--moment_to_plot', type=str, default='mean',
                    help='Define the moment to plot in figures (mean or median with sem or ci)')
parser.add_argument('--exclude_subjects_on_R2', type=bool, default=True,
                    help='Define whether to exclude subjects based on their performance')
parser.add_argument('--save_figures', type=bool, default=True,
                    help='Define whether to save figures')
args = parser.parse_args()
sys.path.append(args.BayesianBattery_path)
sys.path.append(args.BayesianBattery_completements_path)
os.chdir(args.BayesianBattery_path)
print(os.getcwd())

from Analysis.AnalysisOfGorrillaData.AnalysisCommonFunctions import lo
from Analysis.AnalysisOfGorrillaData.VariablesForContextPaperAnalysis\
    import LOGODDS_COLORS
from PathsForContextPaper import define_paths_for_context_paper

PATHS = define_paths_for_context_paper(computer)

#%% FUNCTIONS
def bootstrap_lo_ci(data, ci=[0.025, 0.975], n_permut=1000, moment='mean'):
    """
    Compute the confidence interval around the median by bootstrapping.

    Parameters
    ----------
    data : dataframe
           data from which the median was computed.
    ci : list, optional
        Confidence interval to compute. The default is [0.025, 0.975].
    n_permut : integer, optional
        Number of permutations in bootstrap. The default is 1000.

    Returns
    -------
    arrays, computed confidence interval.

    """

    sample_size = len(data)
    if moment == 'mean':
        resampled_lo_moment = [
            lo(np.mean(np.random.choice(data, sample_size, replace=True)))
            for k in range(n_permut)]
    elif moment == 'median':
        resampled_lo_moment = [
            lo(np.median(np.random.choice(data, sample_size, replace=True)))
            for k in range(n_permut)]

    return np.quantile(resampled_lo_moment, ci)

#%% DEFINE VARIABLES
subjects_to_include_both_exp = pd.read_csv(PATHS['SubjectsList_ErraticAnswers'], 
                                           index_col=0)['participant_ID'].dropna().to_numpy()

concatenated_subjects_results = {
    'exp1': pd.read_csv(PATHS['exp1_data'], index_col=0),
    'exp3': pd.read_csv(PATHS['exp3_data'], index_col=0)}

ideal_observer_column_names_both_exp = {
    'exp1': {'posteriors': 'posteriors_with_generative_priors',
             'choices': 'ideal_observer_with_generative_priors_choices'},
    'exp3': {'posteriors': 'posteriors_with_generative_priors',
             'choices': 'ideal_observers_with_optimal_parameters_choices'}}

plt.rcParams.update({'font.size': 28})
p_color = LOGODDS_COLORS

pretty_names = {'subject_choices': 'Subjects',
                'ideal_observers_choices': 'Ideal observers',
                'exp1': 'Explicit context',
                'exp3':  'Implicit context',
                'ob_faces': 'OF',
                'amb_faces': 'AF',
                'amb_houses': 'AH',
                'ob_houses': 'OH'}
index_order = ['ob_faces', 'amb_faces', 'amb_houses', 'ob_houses']

subjects_summary = pd.read_csv(PATHS['SubjectsSummaryFile'], index_col=0)

#%% COMPUTATIONS AND FIGURES
plot_range = 1

fig, axes = plt.subplots(len(args.behaviors_to_plot), len(args.experiments), sharey=True,
                         figsize=(20, 10), dpi=200)
plt.subplots_adjust(wspace=0.05)

for behavior_type in args.behaviors_to_plot:
    for exp in args.experiments:
        exp_results = concatenated_subjects_results[exp]
        # exp_results['ev_group'] = [pretty_names[ev] for ev in exp_results['ev_group']]
        if 'ideal_observer' in behavior_type:
            choices_column = ideal_observer_column_names_both_exp[exp]['choices']
        else:
            choices_column = behavior_type
        if args.exclude_subjects_on_R2:
            exp_results = exp_results[exp_results['participant_ID'].isin(subjects_to_include_both_exp)]

        # Compute proportion of choices per subject and condition
        avg_per_subject \
            = exp_results.groupby(['participant_ID', 'generative_priors', 'ev_group'])\
                .mean()[choices_column].reindex(level=2, index=index_order)
        # Compute median across subjects, per condition
        moment_group = \
            avg_per_subject.groupby(level=['ev_group', 'generative_priors']).median()\
            if args.moment_to_plot == 'median'\
            else avg_per_subject.groupby(level=['ev_group', 'generative_priors']).mean()
        # Convert proportions of choices into log odds
        moment_group_lo = lo(moment_group)
        # Compute a bootstrap estimate for the CI of the lo of the median or the mean
        confidence_interval_lo \
            = avg_per_subject.dropna().groupby(level=['ev_group', 'generative_priors'])\
            .apply(bootstrap_lo_ci, moment=args.moment_to_plot)
        low_ci = [moment_group_lo.values[i] - ci[0] for i, ci
                  in enumerate(confidence_interval_lo.to_numpy())]
        high_ci = [ci[1] - moment_group_lo.values[i] for i, ci
                   in enumerate(confidence_interval_lo.to_numpy())]

        # Create bar plot
        ax = axes[plot_range-1]
        x = np.concatenate([range(i, i+5) for i in range(0, 22, 6)])
        y = moment_group_lo.values
        yerr = np.array([low_ci, high_ci])
        ax.bar(x, y, color=p_color*4, yerr=yerr, width=1)
        ax.set_title(f"{pretty_names[exp]}", x=0.3, y=0.9)
        ax.set_xlabel('Sensory likelihood')
        ax.set_xticks(range(2, 25, 6))
        xlabels = [pretty_names[lik] for lik in moment_group_lo.index.get_level_values(0).unique()]
        ax.set_xticklabels(xlabels, rotation=0, ha='center')
        plt.legend([], [], frameon=False)
        ax.set_ylabel("Proportion of reports 'houses'\n (log-odds)")
        if plot_range % 2 == 0:
            plt.ylabel("")
            sides_to_despine = ['top', 'right', 'left']
            ax.tick_params(labelleft=False, length=0)
        else:
            sides_to_despine = ['top', 'right']
        for side in sides_to_despine:
            ax.spines[side].set_visible(False)
        plot_range += 1

plt.suptitle("Influence of priors on answers by sensory likelihood and prior level")
nb_of_subjects = len(exp_results['participant_ID'].unique())

if args.save_figures:
    fig.savefig(op.join(PATHS['ContextPaperFiguresPathRoot'], 'svg',
                        'MainFigures', f'LogOddFigure_n{nb_of_subjects}.svg'),
                format='svg', bbox_inches='tight', pad_inches=1, dpi=10000)
    fig.savefig(op.join(PATHS['ContextPaperFiguresPathRoot'], 'png', 
                        'MainFigures', f'LogOddFigure_n{nb_of_subjects}.png'),
                format='png', bbox_inches='tight', pad_inches=1, dpi=300)
