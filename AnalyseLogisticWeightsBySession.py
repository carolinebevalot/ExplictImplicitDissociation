#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:56:56 2024

@author: caroline
"""


import argparse
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy.stats
from scipy.stats import spearmanr

computer = os.uname().nodename

from Analysis.AnalysisOfGorrillaData.VariablesForContextPaperAnalysis \
    import COLORS_FOR_TASK_COMPARISON, MODEL_CROSS_VALIDATION_COLORS_CORRELATIONS
from PathsForContextPaper import define_paths_for_context_paper

PATHS = define_paths_for_context_paper(computer)
    
parser = argparse.ArgumentParser(description="Plot model comparison")
parser.add_argument('-computer', type=str, default=computer,
                    help='Define on which computer the script is run to adapt paths')
parser.add_argument('-BayesianBattery_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery/',
                    help='Define the path for scripts directory')
parser.add_argument('-BayesianBattery_completements_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery_complements/',
                    help='Define the path for the data and results')
parser.add_argument('--save_figures', type=bool, default=True,
                    help='Define whether to save figures')

args = parser.parse_args()
print(args)

subjects_to_include_both_exp = pd.read_csv(PATHS['SubjectsList_ErraticAnswers'],
                                            index_col=0)['participant_ID'].dropna().to_numpy()

#%% COMPUTE THE CORRELATION BETWEEN AND WITHIN TASKS OF THE LOGISTIC WEIGHTS OF PRIORS (FIGS8)

logreg_weights_folder = PATHS['LogregWeightsFolder']

subjects_to_include_both_exp = [s for s in subjects_to_include_both_exp if s != '6143cf2fdb4da562b917fa91']

for variable in ['priors_weight', 'likelihood_weight']:
    all_exp1 = pd.read_csv(PATHS['LogregWeightsFolder'] + '/exp1_ideal_observer_with_generative_priors_3folds_pen.csv',
                           index_col=0).loc[subjects_to_include_both_exp, variable].dropna()
    all_exp3 = pd.read_csv(PATHS['LogregWeightsFolder'] + '/exp3_ideal_observer_with_fitted_decision_and_learning_parameters_3folds_pen.csv',
                           index_col=0).loc[subjects_to_include_both_exp, variable].dropna()

    exp1_3folds = pd.concat([pd.read_csv(PATHS['LogregWeightsFolder'] + f'/exp1_ideal_observer_with_generative_priors_fold{i}_pen.csv',
                                  index_col=0).loc[subjects_to_include_both_exp, variable].dropna()
                                  for i in [0,1,2]], axis=1)
    exp1_3folds.columns = [f'T1_{variable}_in_fold{i}' for i in [0,1,2]]

    exp3_3folds = pd.concat([pd.read_csv(PATHS['LogregWeightsFolder'] + f'/exp3_ideal_observer_with_fitted_decision_and_learning_parameters_fold{i}_pen.csv',
                                  index_col=0).loc[subjects_to_include_both_exp, variable].dropna()
                                  for i in [0,1,2]], axis=1)
    exp3_3folds.columns = [f'T2_{variable}_in_fold{i}' for i in [0,1,2]]

    exp13_3folds = pd.concat([exp1_3folds, exp3_3folds], axis=1)
    exp13_3folds[f'T1_{variable}'] = all_exp1
    exp13_3folds[f'T2_{variable}'] = all_exp3

    if variable == 'priors_weight':
        true_btw_task_rho, pvalue = spearmanr(all_exp1, all_exp3)

        true_wtn_task_rho = []
        wtn_task_pval = []
        for exp_cols in [exp1_3folds.columns, exp3_3folds.columns]:
            for col_a, col_b in combinations(exp_cols, 2):
                rho, pvalue = spearmanr(exp13_3folds[col_a], exp13_3folds[col_b])
                true_wtn_task_rho.append(rho)
                wtn_task_pval.append(pvalue)
        true_wtn_task_rho = np.mean(true_wtn_task_rho)

        possible_subjects = exp13_3folds.index.to_numpy()

        boostrapped_correlations = pd.DataFrame()

        for i in range(100):
            resample_subjects = exp13_3folds.sample(n=possible_subjects.shape[0],
                                                   axis=0, replace=True)
            resample_subjects.dropna(inplace=True)

            # BETWEEN-TASK CORRELATIONS
            btw_task_rho, pvalue = spearmanr(resample_subjects['T1_priors_weight'],
                                             resample_subjects['T2_priors_weight'])
            boostrapped_correlations.loc[i, 'btw_task_rho'] = btw_task_rho

            wtn_task_rho = []
            for exp_cols in [exp1_3folds.columns, exp3_3folds.columns]:
                for col_a, col_b in combinations(exp_cols, 2):
                    rho, pvalue = spearmanr(resample_subjects[col_a], resample_subjects[col_b])
                    wtn_task_rho.append(rho)

            boostrapped_correlations.loc[i, 'wtn_task_rho'] = np.mean(wtn_task_rho)

        confidence_intervals = np.array([np.quantile(boostrapped_correlations[col], [0.025, 0.975])
                                         for col in boostrapped_correlations.columns])
        
        ci = np.array([(boostrapped_correlations.mean() - confidence_intervals[:, 0]).values,
                       (confidence_intervals[:, 1] - boostrapped_correlations.mean()).values])
        

        tvalue, pvalue = scipy.stats.ttest_1samp(boostrapped_correlations.diff(axis=1).dropna(axis=1), 0)

        # FIGURE S8 : BTW VS. WTN TASK CORRELATION
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        ax.bar([0,1],
               boostrapped_correlations.mean(axis=0).values,
               color=MODEL_CROSS_VALIDATION_COLORS_CORRELATIONS,
               width=0.9,
               yerr=ci)
        sns.despine()

        if args.save_figures:
            fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                                     'svg', 'SupplementaryFigures',
                                     f'Bootstrap_of_correlation_of_logweights.svg'), format='svg',
                        bbox_inches='tight', pad_inches=1, dpi=1000)
            fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                                     'png', 'SupplementaryFigures',
                                     f'Bootstrap_of_correlation_of_logweights.png'), format='png',
                        bbox_inches='tight', pad_inches=1, dpi=300)

    # FIGURE S10: LOGISTIC WEIGHT BY SESSION
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    to_plot = exp13_3folds.drop([f'T1_{variable}', f'T2_{variable}'], axis=1)
    x = [0, 1, 2, 4, 5, 6]
    ax.bar(x, to_plot.mean().values,
           color=COLORS_FOR_TASK_COMPARISON[variable][1],
           width=0.9,
           yerr=to_plot.sem(axis=0).values)
    plt.ylim([0, 1.2])
    plt.xticks(x, ['T1_S1', 'T1_S2', 'T1_S3', 'T2_S1', 'T2_S2', 'T2_S3'],
               rotation=45)
    plt.title(variable)
    sns.despine()

    if args.save_figures:
        fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                                 'svg', 'SupplementaryFigures',
                                 f'Logistic_weight_of_{variable}_by_session.svg'), format='svg',
                    bbox_inches='tight', pad_inches=1, dpi=1000)
        fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                                 'png', 'SupplementaryFigures',
                                 f'Logistic_weight_of_{variable}_by_session.png'), format='png',
                    bbox_inches='tight', pad_inches=1, dpi=300)
