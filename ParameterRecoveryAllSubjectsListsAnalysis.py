#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:29:31 2021

@author: caroline
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:15:02 2020

@author: carolinebevalot
"""
import numpy as np
import pickle
import os.path as op
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from SimulationOptimization.VariablesForFit import \
    LEARNING_PARAMETERS, DECISION_PARAMETERS

from Analysis.AnalysisOfGorrillaData.VariablesForContextPaperAnalysis \
    import SIMULATION_OPTIONS_BOTH_EXP, PENALIZATION_SUFFIX, EXPERIMENTS,\
    MODEL_CROSS_VALIDATION_COLORS_CMAP, MODEL_CROSS_VALIDATION_COLORS_CMAP_INV

from PathsForContextPaper import define_paths_for_context_paper
computer = os.uname().nodename
PATHS = define_paths_for_context_paper(computer)

experiment_to_plot = 'exp3'  # 'exp1' or 'exp3'

file = PATHS[f'ParameterRecoveryResults_{experiment_to_plot}']
save_figure = True

if 'learning' in file:
    MODEL_TYPE = 'bayesian_model_of_decision_and_learning'
else:
    MODEL_TYPE = 'bayesian_model_of_decision'


#%% Functions
def define_plotting_window(restricted_window=False):
    ylimits = {}
    xlimits = {}
    for p in PARAMETER_NAMES:
        ylimits[p]=[-40,40]
        xlimits[p]=[-40,40]
    ylimits['volatility'] = [-0.05,1]
    xlimits['volatility'] = [-0.05,1]

    if restricted_window:
        xlimits = ylimits ={'volatility' : [0, 0.3],
                           'strength_evidence' : [-0.1, 5.1],
                           'bias_evidence': [-0.22, 0.30],
                           'soft_max_beta': [0, 12],
                           'resp_bias': [-1.1, 1.1],
                           'prior_lik_weight': [0,5],
                           'prior_weight': [0,5],
                           'lik_weight': [0,5]}

    return xlimits, ylimits



def plot_parameters_and_regression(x, y, parameter,  xlim, ylim):

    plt.scatter(x, y, s=2, color='black')
    xlab = 'generative {}'.format(parameter)
    ylab = 'fitted {}'.format(parameter)
    title = '{}'.format(parameter) if parameter == 'volatility' \
        else '{}'.format(parameter)
    plt.title(f'{title}')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.ylim(ylim)
    plt.xlim(xlim)


def define_filename(param_to_plot):
    plot_type = 'all_scatters'
    path = 'BatteryData/Figures_parameter_recovery/{}'.format(plot_type)
    pathroot = 'BatteryData/Figures_parameter_recovery'

    file_names = {'var_beta_dist': 'beta_var',
                  'mean_beta_tone_dist': 'beta_tone_mean',
                  'p_c_novol': 'p_c',
                  'p_c_vol': 'p_c',
                  'bias_evidence': 'evidenceB',
                  'strength_evidence': 'evidence'}
    filename = '{}_median_{}'.format(file_names[param_to_plot[0]], plot_type)

    return path, pathroot, filename


def create_plot_gen_vs_fitted_parameters_with_regression(all_generative_param, all_fitted_param,
                                                         PARAMETER_NAMES, title):
    """Function which plot as scatter plot or bar plot (with inter quartile range),
    the median of the 12 fitted values for each genertive value of a parameter.
    yerr should be fit_median_for_gen_value,parameter"""

    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(6.5*len(PARAMETER_NAMES)/2, 13))
    plt.suptitle(title)
    xlimits, ylimits = define_plotting_window(restricted_window=True)

    for i, parameter in enumerate(PARAMETER_NAMES):
        plt.subplot(2, 3, i+1)
        # plt.subplots_adjust(wspace=2, hspace=1)
        print(parameter)
        frame_to_plot = pd.concat([all_generative_param[f'gen_{parameter}'],
                                   all_fitted_param[f'fit_{parameter}']], axis=1)
        frame_to_plot.columns = ['gen', 'fit']
        frame_to_plot.dropna(axis=0, inplace=True)
        x = frame_to_plot['gen'].values
        y = frame_to_plot['fit'].values

        xlim, ylim = xlimits[parameter], ylimits[parameter]

        plot_parameters_and_regression(x, y, parameter, xlim, ylim)

def plot_correlation_matrix(all_generative_param, all_fitted_param, title, method='spearman'):

    to_corr = pd.concat([all_generative_param, all_fitted_param], axis=1)
    to_corr.replace([np.inf, -np.inf], np.nan, inplace=True)
    to_corr.dropna(how='any', axis=0, inplace=True)
    spearcor = to_corr.corr(method=method)
    figure = plt.figure()
    sns.heatmap(spearcor.loc[all_fitted_param.columns, all_generative_param.columns],
                vmin=-1, vmax=1,
                cmap='coolwarm',
                annot=True, fmt='.2f',
                xticklabels=all_generative_param.columns, yticklabels=all_fitted_param.columns)
    plt.xticks(rotation=90)
    plt.title(title)
    return figure

def plot_identifiability_matrix(all_fitted_param, title, method='spearman'):

    to_corr = all_fitted_param
    to_corr.replace([np.inf, -np.inf], np.nan, inplace=True)
    to_corr.dropna(how='any', axis=0, inplace=True)
    spearcor = to_corr.corr(method=method)
    figure = plt.figure()
    sns.heatmap(spearcor.loc[all_fitted_param.columns, all_fitted_param.columns],
                vmin=0, vmax=1, cmap='coolwarm', annot=True, fmt='.2f',
                xticklabels=all_fitted_param.columns, yticklabels=all_fitted_param.columns)
    plt.title(title)
    plt.xticks(rotation=90)

    return figure

#%% Define model parameters
PARAMETER_NAMES = []
if 'learning' in MODEL_TYPE:
    PARAMETER_NAMES += LEARNING_PARAMETERS
if 'bayesian' in MODEL_TYPE:
    PARAMETER_NAMES += DECISION_PARAMETERS
else:
    PARAMETER_NAMES += LINEAR_DECISION_PARAMETERS

#%% Retrieve generative and fitted parameter from the parameter recovery
analysis_mode = 'minimum_negative_loglikelihood'  # 'median_parameters' #'
gen_param_names = [f'gen_{p}' for p in PARAMETER_NAMES]
fit_param_names = [f'fit_{p}' for p in PARAMETER_NAMES]
parameter_recovery_results = pd.read_csv(file, index_col=0)

if analysis_mode == 'minimum_negative_loglikelihood':
    best_fits = parameter_recovery_results[parameter_recovery_results['fit_status'] == 'best_fit']
    all_generative_param = best_fits[gen_param_names]
    all_fitted_param = best_fits[fit_param_names]

else:
    all_generative_param = pd.DataFrame()
    all_fitted_param = pd.DataFrame()
    for i, simulation in enumerate(parameter_recovery_results['sim_ID'].dropna().unique()):
        recovery_iterations = parameter_recovery_results[parameter_recovery_results['sim_ID'] == simulation]
        all_generative_param.loc[i, gen_param_names] = recovery_iterations[gen_param_names].median()
        all_fitted_param.loc[i, fit_param_names] = recovery_iterations[fit_param_names].median()
        
#%%
create_plot_gen_vs_fitted_parameters_with_regression(all_generative_param, all_fitted_param,
                                                     PARAMETER_NAMES, 'Parameter Recovery')
figure = plot_correlation_matrix(all_generative_param, all_fitted_param,
                                 'Recovery Matrix ', method='spearman')
figure = plot_identifiability_matrix(all_fitted_param,
                                      'Identifiability Matrix ', method='spearman')
if save_figure:
    figure.savefig(op.join(PATHS['ContextPaperFiguresPathRoot'],
                           'svg',
                           'SupplementaryFigures',
                           "ParameterRecoveryCompleteModelFigure2_T1.svg"),
                   format='svg', bbox_inches='tight', pad_inches=0,
                   dpi=1200)
    figure.savefig(op.join(PATHS['ContextPaperFiguresPathRoot'],
                           'png',
                           'SupplementaryFigures',
                           "ParameterRecoveryCompleteModelFigure2_T1.png"),
                   format='png', bbox_inches='tight', pad_inches=0,
                   dpi=1200)




