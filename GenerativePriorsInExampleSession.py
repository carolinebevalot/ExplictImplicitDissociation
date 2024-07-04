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
parser.add_argument('--save_figures', type=bool, default='True',
                    help='Define whether to save figures')
args = parser.parse_args()
sys.path.append(args.BayesianBattery_path)
sys.path.append(args.BayesianBattery_completements_path)
os.chdir(args.BayesianBattery_path)
print(os.getcwd())

from Analysis.AnalysisOfGorrillaData.AnalysisCommonFunctions import lo
from PathsForContextPaper import define_paths_for_context_paper

PATHS = define_paths_for_context_paper(computer)

subjects_to_include_both_exp = pd.read_csv(PATHS['SubjectsList_ErraticAnswers'],
                                           index_col=0)['participant_ID'].dropna().to_numpy()
concatenated_subjects_results = pd.read_csv(PATHS['exp3_data'], index_col=0)

figsize = (8, 5)

#%%% GENERATIVE STRUCTURE OF THE IMPLICIT TASK
fig = plt.figure(figsize=figsize, dpi=800)
s_data = concatenated_subjects_results[concatenated_subjects_results['participant_ID']
                                       == '6142f8d8ecccc3c44ccd8fa3']
plt.plot(s_data['generative_priors'].index, s_data['generative_priors'].values,
         color='darkgrey')
plt.plot(np.repeat(194, 20), np.linspace(0, 1, 20), '--',
         color='black')
plt.plot(np.repeat(436, 20), np.linspace(0, 1, 20), '--',
         color='black')
sns.despine()
fig.savefig(op.join(PATHS['ContextPaperFiguresPathRoot'], 'svg', 'MainFigures',
                    "generative_structure_of_implicit_task.svg"), format='svg',
            bbox_inches='tight', pad_inches=1, dpi=10000)
fig.savefig(op.join(PATHS['ContextPaperFiguresPathRoot'], 'png', 'MainFigures',
                    "generative_structure_of_implicit_task.png"), format='png',
            bbox_inches='tight', pad_inches=1, dpi=300)

#%% GENERATIVE STRUCTURE OF THE EXPLICIT TASK
concatenated_subjects_results_exp1 = pd.read_csv(PATHS['exp1_data'], index_col=0)
s_data = concatenated_subjects_results_exp1[concatenated_subjects_results_exp1['participant_ID']
                                            == '6142f8d8ecccc3c44ccd8fa3']
fig = plt.figure(figsize=figsize, dpi=800)

plt.plot(s_data['generative_priors'].index, s_data['generative_priors'].values,
         color='darkgrey')
plt.plot(np.repeat(116, 20), np.linspace(0, 1, 20), '--',
         color='black')
plt.plot(np.repeat(232, 20), np.linspace(0, 1, 20), '--',
         color='black')
sns.despine()

fig.savefig(op.join(PATHS['ContextPaperFiguresPathRoot'], 'svg', 'MainFigures',
            "generative_structure_of_explicit_task.svg"), format='svg',
            bbox_inches='tight', pad_inches=1, dpi=10000)
fig.savefig(op.join(PATHS['ContextPaperFiguresPathRoot'], 'png', 'MainFigures',
            "generative_structure_of_explicit_task.png"), format='png',
            bbox_inches='tight', pad_inches=1, dpi=300)
