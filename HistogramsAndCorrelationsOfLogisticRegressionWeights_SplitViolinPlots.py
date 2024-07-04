#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 22:07:18 2022
This script creates the figure 4 and the supplementary figures 3 and 4.
These figures compare the weights of explicit and implicit priors between tasks
 and their correlation.
The script first load previously computed logistic weights of priors, posteriors
and sensory likelihood for the different prior types and the different levels of
sensory likelihood.
The first part of the script is about selecting rows that concern bayesian combination
of priors and likelihood, that include subjects with a sufficient R2.

The main figure is then created with generative priors in T1 and fitted priors
(with the complete model) in T2. The violinplot and the correlation are plotted
in 5 panels.

Results are then plotted for the different prior types and the different levels
of sensory evidence (obvious or ambiguous) and saved in a csv file (for suppl. figure 3).
Correlations are used to build suppl.figure 4

#%% PLOT OF THE WEIGHTS OF PRIORS, LIKELIHOOD AND POSTERIORS IN T1 COMPARED TO T2
    # WITH GENERATIVE AND FITTED PRIORS FOR ALL LEVELS OF EVIDENCE (MAIN FIGURE)
    # WITH GENERATIVE AND FITTED PRIORS FOR VARYING EVIDENCE LEVELS (SUPPL FIGURES)
    # WITH ALL LEVELS OF EVIDENCE FOR VARYING TYPE PRIORS (SUPPL FIGURES)

@author: caroline
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import scipy
import os
import sys

computer = os.uname().nodename
parser = argparse.ArgumentParser(description="Plot violinplots of weights in the logistic regression")
parser.add_argument('--computer', type=str, default=computer,
                    help='Define on which computer the script is run to adapt paths')
parser.add_argument('-BayesianBattery_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery/',
                    help='Define the path for scripts directory')
parser.add_argument('-BayesianBattery_completements_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery_complements/',
                    help='Define the path for the data and results')
parser.add_argument('--plot_only_main_figure', type=bool, default=True,
                    help='Define whether to plot only the main figure and not supplementary figures')
parser.add_argument('--save_figures', type=bool, default=True,
                    help='Define whether to save figures')
args = parser.parse_args()
sys.path.append(args.BayesianBattery_path)
sys.path.append(args.BayesianBattery_completements_path)
os.chdir(args.BayesianBattery_path)
print(os.getcwd())

from Analysis.AnalysisOfGorrillaData.VariablesForContextPaperAnalysis\
    import MODEL_OPTIONS_BOTH_EXP, PRETTY_NAMES

from Analysis.AnalysisOfGorrillaData.AnalysisCommonFunctions\
    import run_statistic_comparisons, plot_correlation, plot_violinplot
from PathsForContextPaper import define_paths_for_context_paper

PATHS = define_paths_for_context_paper(computer)

subjects_to_include_both_exp = pd.read_csv(PATHS['SubjectsList_ErraticAnswers'],
                                           index_col=0)['participant_ID'].dropna().to_numpy()

#%% DEFINE VARIABLES SPECIFIC TO THE TASK
behavior_to_plot = 'subject_choices'  # 'ideal_observer'
weights_in_violinplot = ['posteriors_weight', 'priors_weight', 'likelihood_weight']
weights_to_correlate = ['priors_weight', 'likelihood_weight']
combination_type = 'bayesian'
evidence_level = 'all_levels'
penalization_suffix = 'pen'
#%% LOAD LOGISTIC WEIGHTS 

logreg_weights_folder = PATHS['LogregWeightsFolder']

#%% DEFINE FIGURES PARAMETERS

figsave_options = dict(format='svg', bbox_inches='tight', pad_inches=0, dpi=10000)
figsave_options_png = dict(format='png', bbox_inches='tight', pad_inches=0, dpi=300)

#%% Plot whether weights are significantly different from 0

pvalues = pd.DataFrame(index=['priors_weight', 'likelihood_weight'])
logreg_weights_subsets_exp3 \
    = pd.read_csv(f"{logreg_weights_folder}/exp3_ideal_observer_with_generative_priors_all_levels_"
                  + f"{penalization_suffix}.csv",
                  index_col=0)
logreg_weights_subsets_exp1 \
    = pd.read_csv(f"{logreg_weights_folder}/exp1_ideal_observer_with_generative_priors_all_levels_"
                  + f"{penalization_suffix}.csv",
                  index_col=0)
for var in pvalues.index:
    pvalues.loc[var, 'mean_exp1'] = logreg_weights_subsets_exp1[var].dropna().mean()
    pvalues.loc[var, 'mean_exp3'] = logreg_weights_subsets_exp3[var].dropna().mean()
    pvalues.loc[var, 'cohend_exp1'] = \
        logreg_weights_subsets_exp1[var].dropna().mean() / logreg_weights_subsets_exp1[var].dropna().std()
    pvalues.loc[var, 'cohend_exp3'] = \
        logreg_weights_subsets_exp3[var].dropna().mean() / logreg_weights_subsets_exp3[var].dropna().std()
    pvalues.loc[var, ['ci_inf_exp1', 'ci_sup_exp1']] = \
        np.percentile(logreg_weights_subsets_exp1[var].dropna(), [2.5, 97.5])
    pvalues.loc[var, ['ci_inf_exp3', 'ci_sup_exp3']] = \
        np.percentile(logreg_weights_subsets_exp3[var].dropna(), [2.5, 97.5])

    pvalues.loc[var, 'sem_exp1'] = logreg_weights_subsets_exp1[var].dropna().sem()
    pvalues.loc[var, 'sem_exp3'] = logreg_weights_subsets_exp3[var].dropna().sem()
    pvalues.loc[var, 'tval_exp1'] = scipy.stats.ttest_1samp(logreg_weights_subsets_exp1[var].dropna().values, 0)[0]
    pvalues.loc[var, 'tval_exp3'] = scipy.stats.ttest_1samp(logreg_weights_subsets_exp3[var].dropna().values, 0)[0]
    pvalues.loc[var, 'wpval_exp1'] = scipy.stats.wilcoxon(logreg_weights_subsets_exp1[var].dropna().values)[1]
    pvalues.loc[var, 'wpval_exp3'] = scipy.stats.wilcoxon(logreg_weights_subsets_exp3[var].dropna().values)[1]
    
#%% SELECT COLUMNS TO KEEP IN THE FRAME
statistic_comparisons = []
correlations = []
#%% WITH ALL LEVELS OF EVIDENCE FOR VARYING TYPE PRIORS 
# for prior_parameters_options in MODEL_OPTIONS_BOTH_EXP['exp3'].keys():
for prior_parameters_options in ['ideal_observer_with_optimal_parameters', 'ideal_observer_with_fitted_decision_and_learning_parameters']:
    combination = 'linear' if 'lin' in prior_parameters_options else 'bayesian'
    weights_in_violinplot \
        = ['priors_weight', 'likelihood_weight'] if 'linear' in prior_parameters_options \
        else ['posteriors_weight', 'priors_weight', 'likelihood_weight']
    interctxt_diffs_to_test = ['priors_weight', 'likelihood_weight']
    if (args.plot_only_main_figure) and (prior_parameters_options != 'ideal_observer_with_fitted_decision_and_learning_parameters'):
        pass
    else:
        if (not args.plot_only_main_figure) and (prior_parameters_options == 'ideal_observer_with_fitted_decision_and_learning_parameters'):
            evidence_levels_to_plot = ['all_levels', 'obvious', 'ambiguous']
        else:
            evidence_levels_to_plot = ['all_levels']
        for evidence_level in evidence_levels_to_plot:
            free_parameters = MODEL_OPTIONS_BOTH_EXP['exp3'][prior_parameters_options]['free_parameters']
            prior_column_name = PRETTY_NAMES[MODEL_OPTIONS_BOTH_EXP['exp3'][prior_parameters_options]['prior_column_name']]
            logreg_options_label = f"{MODEL_OPTIONS_BOTH_EXP['exp3'][prior_parameters_options]['logreg_options_label']}_{evidence_level.replace('_','')}"

            logreg_weights_subsets_exp3 \
                = pd.read_csv(f"{logreg_weights_folder}/exp3_{prior_parameters_options}_"
                              + f"{evidence_level}_{penalization_suffix}.csv", index_col=0)\
                    .loc[subjects_to_include_both_exp, :]

            exp1_file \
                = f"exp1_linear_ideal_observer_with_generative_priors_{evidence_level}_"\
                  + f"{penalization_suffix}.csv" if 'linear' in prior_parameters_options\
                else f"exp1_ideal_observer_with_generative_priors_{evidence_level}_"\
                    + f"{penalization_suffix}.csv"
            logreg_weights_subsets_exp1 \
                = pd.read_csv(f"{logreg_weights_folder}/{exp1_file}",
                              index_col=0).loc[subjects_to_include_both_exp, :]


            stats_for_violinplot = \
                run_statistic_comparisons(logreg_weights_subsets_exp1, logreg_weights_subsets_exp3,
                                          logreg_options_label, weights_in_violinplot, 
                                          interctxt_diffs_to_test)

            violinplots = plot_violinplot(logreg_weights_subsets_exp1, logreg_weights_subsets_exp3,
                                          weights_in_violinplot, prior_column_name, free_parameters, stats_for_violinplot,
                                          logreg_options_label=logreg_options_label)

            filename = f"T1T2_gen_priors_vs_{prior_column_name.replace(' ','_')}_" + \
                f"n{len(subjects_to_include_both_exp)}_{evidence_level}"
            if args.save_figures:
                if prior_parameters_options == 'ideal_observer_with_fitted_decision_and_learning_parameters'\
                 and evidence_level == 'all_levels':
                    png_subfolder = op.join(PATHS['ContextPaperFiguresPathRoot'],
                                            'png',
                                            'MainFigures')
                    svg_subfolder = op.join(PATHS['ContextPaperFiguresPathRoot'],
                                            'svg',
                                            'MainFigures')
                    csv_subfolder = op.join(PATHS['ContextPaperFiguresPathRoot'],
                                            'csv')

                else:
                    png_subfolder = op.join(PATHS['ContextPaperFiguresPathRoot'],
                                            'png',
                                            'SupplementaryFigures')
                    svg_subfolder = op.join(PATHS['ContextPaperFiguresPathRoot'],
                                            'svg',
                                            'SupplementaryFigures')
                    csv_subfolder = op.join(PATHS['ContextPaperFiguresPathRoot'],
                                            'csv')

                try:
                    os.makedirs(png_subfolder)
                    os.makedirs(svg_subfolder)
                    os.makedirs(csv_subfolder)
                except FileExistsError:
                    pass

                violinplots.savefig(op.join(svg_subfolder,
                                            f"Violinplot_{filename}_{combination}.svg"),
                                    **figsave_options)
                violinplots.savefig(op.join(png_subfolder,
                                            f"Violinplot_{filename}_{combination}.png"),
                                    **figsave_options_png)

            # Plot the correlation between weights in T1 and T2
            scatterplots, correlation_strength\
                = plot_correlation(logreg_weights_subsets_exp1, logreg_weights_subsets_exp3,
                                   weights_to_correlate, prior_column_name,
                                   logreg_options_label=logreg_options_label)
            correlation_strength.loc[:, 'logreg_options_label'] = logreg_options_label
            if args.save_figures:
                scatterplots.savefig(op.join(svg_subfolder,
                                             f"Correlation_{filename}_{combination}_logscale.svg"),
                                     **figsave_options)
                scatterplots.savefig(op.join(png_subfolder,
                                             f"Correlation_{filename}_{combination}_logscale.png"),
                                     **figsave_options_png)
            statistic_comparisons.append(stats_for_violinplot
                                         .drop('lik_prior', axis=0)
                                         .drop('logreg_options', axis=1).round(3))
            correlations.append(correlation_strength.round(3))
#%%
if args.save_figures and not args.plot_only_main_figure:
    statistic_comparisons = pd.concat(statistic_comparisons)
    correlations = pd.concat(correlations)
    pd.set_option('display.precision', 3)
    statistic_comparisons.T.to_csv(op.join(csv_subfolder,
                                         'Violinplot_and_correlation_T1T2_statistic_comparisons'
                                         + f'_n{len(subjects_to_include_both_exp)}_2103_mod.csv'),
                                 float_format='%.3f')
    correlations.T.to_csv(op.join(csv_subfolder,
                                         'Violinplot_and_correlation_T1T2_statistic_comparisons_correlations_suppl'
                                         + f'_n{len(subjects_to_include_both_exp)}_2103_mod.csv'),
                                 float_format='%.3f')
