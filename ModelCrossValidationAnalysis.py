#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:18:32 2020

This script plot the results of the cross validation of the possible models of behavior
simulated with ModelCressValidation.py
The 279 subjects have been considered in this analysis.
We have two types of models : bayesian models and linear models.
For each model type, an increasing number of parameters (from 0 to 4) is fixed.

For each model, we adopted a cross validated procedure as we fitted the model on two folds
and computed the loglikelihood on the third one.
The fitting procedure is restarted 50 times as when we fit subjects' parameters.
As the median parameters were selected, we consider the median likelihood.
The best fit can also be considered by setting --method_to_compute_likelihood_of_subject_choices
to 'best_fit'.
In this script, we sum the three log likelihood corresponding to the three testing sets.
We compute the likelihood of a choice by taking the exponential of the log likelihood
and dividing it by the number of trials used in the fit (excluding trials with no answer)
We thus have the likelihood of a choice for each subject.
In order to plot results as the group level, the median likelihood is considered
(as distributions are skewed). It can be changed to use the mean by setting
--method_to_compute_model_likelihood_at_group_level to 'mean'.

We then plot in a bar graph the likelihoods at the group level under the different models.
The error plotted is the s.e.m.
The optimal models (with no free parameters) can be removed from the plot with
the argument --add_optimal_model set to False.
The set of adjusted and fixed parameters is represented as a table.
Models are compared one to another and the p value of each t test is reported in a matrix .

In case of a conflict with model names use :
cross_validation['model'] = [RENAME_MODELS[model] if model in RENAME_MODELS.keys()
                              else model for model in cross_validation['model']]
cross_validation.index = cross_validation['participant']

@author: carolinebevalot
"""

import argparse
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy.stats
import six
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from statsmodels.stats.api import DescrStatsW

computer = os.uname().nodename

from SimulationOptimization.VariablesForFit import LEARNING_PARAMETERS, DECISION_PARAMETERS, LINEAR_DECISION_PARAMETERS
from Analysis.AnalysisOfGorrillaData.VariablesForContextPaperAnalysis \
    import SUBJECT_GROUP, MODEL_CROSS_VALIDATION_COLORS_COMPARISONS, \
        MODEL_CROSS_VALIDATION_COLORS_CORRELATIONS, BLUEGREY#, MODEL_CROSS_VALIDATION_COLORS_CORRELATIONS2
from SimulationOptimization.OptimizationFunctions import \
    define_models_to_test, define_models_to_test_opt_priors
from PathsForContextPaper import define_paths_for_context_paper

PATHS = define_paths_for_context_paper(computer)
    
parser = argparse.ArgumentParser(description="Plot model comparison")
parser.add_argument('--computer', type=str, default='laptop', help='Define on which computer the script is run to adapt paths')
parser.add_argument('--add_optimal_model', type=bool, default=True,
                    help='Whether optimal bayesian model should be plotted on the figure')

parser.add_argument('--save_figures', type=bool, default=True,
                    help='Define whether to save figures')

args = parser.parse_args()
print(args)

filenames = {
    'exp1': PATHS['ModelCrossValidation_exp1'],
    'exp3': PATHS['ModelCrossValidation_exp3'],
    'exp1_opt': PATHS['ModelCrossValidation_OptModel_exp1'],
    'exp3_opt': PATHS['ModelCrossValidation_OptModel_exp3'],
    'exp3_bayes_var': PATHS['ModelCrossValidation_BayesVar_exp3']}


model_options = {
    ** {f'exp1_{model_name}': options for model_name, options in define_models_to_test_opt_priors('exp1', ['linear', 'bayesian']).items()},
    ** {f'exp3_{model_name}': options for model_name, options in define_models_to_test_opt_priors('exp3', ['linear', 'bayesian']).items()},
    ** {f'{exp}_gen_priors_only': {'model_type': 'PM', 'free_parameters_list': [],
                                   'fixed_parameters': {'prior_weight'}} for exp in ['exp1', 'exp3']},
    ** {f'{exp}_priors_only': {'model_type': 'PM', 'free_parameters_list': [],
                               'fixed_parameters': {'prior_weight'}} for exp in ['exp1', 'exp3']},
    ** {f'{exp}_likelihood_only': {'model_type': 'PM', 'free_parameters_list': [],
                                   'fixed_parameters': {'lik_weight'}} for exp in ['exp1', 'exp3']},
    'exp1_optimal_bayesian': {'model_type': 'bayesian_model_of_decision', 'free_parameters_list': [],
                              'fixed_parameters': set(DECISION_PARAMETERS)},
    'exp3_optimal_bayesian': {'model_type': 'bayesian_model_of_decision_and_learning', 'free_parameters_list': [],
                              'fixed_parameters': set(LEARNING_PARAMETERS).union(set(DECISION_PARAMETERS))},
    'exp3_linear_model_with_7_free_parameters__fix_': {'model_type': 'linear_model_of_decision_and_learning',
                                                       'free_parameters_list': set(LEARNING_PARAMETERS).union(set(LINEAR_DECISION_PARAMETERS)),
                                                       'fixed_parameters': []}}

#%% FUNCTIONS

def set_row_edge_color(the_table, color):
    """
    Function which set the color of each table cell
    """
    for k, cell in six.iteritems(the_table._cells):
        cell.set_edgecolor(color)

def get_fitted_and_fixed_parameters(model_name):
    exp = model_name[:4]
    fixed_parameters = model_options[model_name]['fixed_parameters']
    free_parameters = model_options[model_name]['free_parameters_list']
    param_ID_if_fix = 'opt' if 'bayesian' in model_name else 'Z'

    if 'only' in model_name:
        mapping = {'priors': 'PM opt priors',
                   'gen_priors': 'PM gen priors',
                   'likelihood': 'PM likelihood'}
        model_name_for_fig = mapping[model_name[5:-5]]
    else:
        model_name_for_fig = 'Bayesian' if 'bayesian' in model_name \
                              else 'Linear'

    return fixed_parameters, free_parameters, param_ID_if_fix, model_name_for_fig

#%% DEFINE SUBJECTS TO INCLUDE AND FILENAME   
subjects_to_include_both_exp = pd.read_csv(PATHS['SubjectsList_ErraticAnswers'],
                                            index_col=0)['participant_ID'].dropna().to_numpy()
print(f'subjects to include in the analysis = {len(subjects_to_include_both_exp)}')

n_subjects = len(subjects_to_include_both_exp) 
ANALYSIS_SPECIFIC_NAME \
    = f'_{n_subjects}_subjects'


#%% COMPUTE AND PLOT THE cvLL OF THE BAYES-FIT-DECISION AND THE HEURISTIC MODELS (FIGURE 5)

both_exp_choice_likelihood_on_testing_sets = []
exp_to_plot = ['exp1', 'exp3']

for exp in exp_to_plot:
    cross_validation = pd.read_csv(filenames[exp], index_col=0)
    cv_opt_models = pd.read_csv(filenames[f'{exp}_opt'], index_col=0).drop('participant.1', axis=1)
    cross_validation = pd.concat([cross_validation, cv_opt_models])
    cross_validation = cross_validation.loc[subjects_to_include_both_exp, :]

    print(f" {exp} nb of participants = {len(cross_validation.index.unique())}")

    # Compute the likelihood of each choice of the testing set under each adjusted model
    choice_likelihood_on_testing_sets = pd.DataFrame()

    columns_for_lik_on_testing_set_by_fold \
        = [col for col in cross_validation.columns
            if (col.startswith('LL_adjusted_param_best_fit_testing_set')
                and 'mean_parameters_loglik_for_choices' in col)]
    columns_for_optimal_lik_on_testing_set_by_fold \
        = [col for col in cross_validation.columns
           if (col.startswith('LL_optimal_param_testing_set')
               and 'mean_parameters_loglik_for_choices' in col)]

    models_to_plot = cross_validation.model.unique()

    # Sum the likelihood of the three testing sets and compute the mean likelihood of a choice
    for model in models_to_plot:
        if 'optimal' in model:
            optimal_lik_by_subjects_complete_exp =\
                   cross_validation[cross_validation['model'] == model][columns_for_optimal_lik_on_testing_set_by_fold].mean(axis=1)
            choice_likelihood_on_testing_sets[f'{exp}_optimal_bayesian'] = optimal_lik_by_subjects_complete_exp
        else:
            model_lik_by_subjects_complete_exp =\
            cross_validation[cross_validation['model'] == model][columns_for_lik_on_testing_set_by_fold].mean(axis=1)
            choice_likelihood_on_testing_sets[f'{exp}_{model}'] = model_lik_by_subjects_complete_exp

    # Sort results according to the selected estimator (mean or median)
    columns_sorted = choice_likelihood_on_testing_sets.median().sort_values().index

    both_exp_choice_likelihood_on_testing_sets.append(choice_likelihood_on_testing_sets[columns_sorted])

both_exp_choice_likelihood_on_testing_sets = pd.concat(both_exp_choice_likelihood_on_testing_sets, axis=1)

#%% FIGURE 4 SIMPLIFIED WITH DIFFERENCES IN ACCURACY (LEFT PART)
txtsize = 18
plt.rcParams.update({'font.size': txtsize})
bar_colors = BLUEGREY
figsize = (8,8)  # if exp_to_plot == 'exp3' else (1.2, 10)
fig, ax = plt.subplots(1, 1, figsize=figsize)

# Add a white space between the two contexts
cols_exp1 = ['exp1_linear_model_with_fitted__lik_weight__prior_weight__resp_bias',
             'exp1_bayesian_model_with_fitted__lik_weight__prior_weight__resp_bias']
cols_exp3 = ['exp3_linear_model_with_fitted__lik_weight__prior_weight__resp_bias',
             'exp3_bayesian_model_with_fitted__lik_weight__prior_weight__resp_bias']

bay_exp1 = both_exp_choice_likelihood_on_testing_sets[
    'exp1_bayesian_model_with_fitted__lik_weight__prior_weight__resp_bias']
bay_exp3 = both_exp_choice_likelihood_on_testing_sets[
    'exp3_bayesian_model_with_fitted__lik_weight__prior_weight__resp_bias']
lin_exp1 = both_exp_choice_likelihood_on_testing_sets[
    'exp1_linear_model_with_fitted__lik_weight__prior_weight__resp_bias']
lin_exp3 = both_exp_choice_likelihood_on_testing_sets[
    'exp3_linear_model_with_fitted__lik_weight__prior_weight__resp_bias']

# Compute the difference between T1 and T2 of the difference of performances
# between the bayesian and linear models
print('DIFFERENCE OF CVLL BETWEEN BAYES-FIT-DECISION AND HEURISTIC MODEL')
diff_bay_lin_exp3 = bay_exp3 - lin_exp3
diff_bay_lin_exp1 = bay_exp1 - lin_exp1

plot_both_exp_choice_likelihood_on_testing_sets = pd.DataFrame()
plot_both_exp_choice_likelihood_on_testing_sets['exp1_BayesianLinearDiff'] \
    = diff_bay_lin_exp1
plot_both_exp_choice_likelihood_on_testing_sets['exp3_BayesianLinearDiff']\
    = diff_bay_lin_exp3

# Create the bar plot
ax.bar(range(len(plot_both_exp_choice_likelihood_on_testing_sets.columns)),
       plot_both_exp_choice_likelihood_on_testing_sets.median().values,
       color=bar_colors,
       yerr=plot_both_exp_choice_likelihood_on_testing_sets.sem().values,
       width=0.9)

# Improve legend
lims = [-0.003, 0.003]
plt.ylim(lims)
plt.yticks(labels=np.round(ax.get_yticks(), 4), ticks=ax.get_yticks(), rotation=90,
           verticalalignment='center')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', right=False, top=False, labelbottom=True)
ax.set(xticklabels=[])
ax.spines['top'].set_position('zero')
# ax.yaxis.tick_right()
ax.yaxis.set_label_position("left")
ax.set_ylabel("Relative predictive accuracy of the models",
              fontsize=txtsize+3, y=0.5)
sns.despine(top=False, right=False, bottom=True, left=True)

if args.save_figures:
    fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                             'svg',  'MainFigures',
                             f'ModelCrossValidation_T1T2_{ANALYSIS_SPECIFIC_NAME}.svg'),
                format='svg', bbox_inches='tight', pad_inches=1, dpi=10000)

    fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                             'png',  'MainFigures',
                             f'ModelCrossValidation_T1T2_{ANALYSIS_SPECIFIC_NAME}.png'),
                format='png', bbox_inches='tight', pad_inches=1, dpi=300)

#%% COMPUTE THE DIFFERENCE BETWEEN BAYES-OPTIMAL AND CHANCE OR
# BETWEEN BAYES-FIT-DECISION AND BAYES-OPTIMAL

opt_vs_chance_exp1 = both_exp_choice_likelihood_on_testing_sets.exp1_optimal_bayesian.values - np.log(0.5)
print(f"Optimal vs Chance in exp1")
print(f"Median opt : {both_exp_choice_likelihood_on_testing_sets.exp1_optimal_bayesian.median()}")
print(f"Confidence intervale opt : {np.percentile(both_exp_choice_likelihood_on_testing_sets['exp1_optimal_bayesian'], [2.5, 97.5])}")
print(f"SEM opt : {both_exp_choice_likelihood_on_testing_sets.exp1_optimal_bayesian.sem()}")
print(f"Cohen's d : {opt_vs_chance_exp1.mean() / opt_vs_chance_exp1.std()}")
print(f"T test : {scipy.stats.ttest_1samp(opt_vs_chance_exp1,0)}")
print(f"Wilcoxon : {scipy.stats.wilcoxon(opt_vs_chance_exp1)}")
print('')
opt_vs_chance_exp3 = both_exp_choice_likelihood_on_testing_sets.exp3_optimal_bayesian.values - np.log(0.5)
print(f"Optimal vs Chance in exp3")
print(f"Median opt : {both_exp_choice_likelihood_on_testing_sets.exp3_optimal_bayesian.median()}")
print(f"Confidence intervale opt : {np.percentile(both_exp_choice_likelihood_on_testing_sets['exp3_optimal_bayesian'], [2.5, 97.5])}")
print(f"SEM opt : {both_exp_choice_likelihood_on_testing_sets.exp3_optimal_bayesian.sem()}")
print(f"Cohen's d : {opt_vs_chance_exp3.mean() / opt_vs_chance_exp3.std()}")
print(f"T test : {scipy.stats.ttest_1samp(opt_vs_chance_exp3,0)}")
print(f"Wilcoxon : {scipy.stats.wilcoxon(opt_vs_chance_exp3)}")
print('')

bay_vs_opt_diff_exp1 = both_exp_choice_likelihood_on_testing_sets.exp1_bayesian_model_with_fitted__lik_weight__prior_weight__resp_bias\
     - both_exp_choice_likelihood_on_testing_sets.exp1_optimal_bayesian
print(f"Optimal vs BAYES-FIT-DECISION in exp1")
print(f"Median bayes fit : {both_exp_choice_likelihood_on_testing_sets['exp1_bayesian_model_with_fitted__lik_weight__prior_weight__resp_bias'].median()}")
print(f"Confidence intervale bayes fit : \
      {np.percentile(both_exp_choice_likelihood_on_testing_sets['exp1_bayesian_model_with_fitted__lik_weight__prior_weight__resp_bias'], [2.5, 97.5])}")
print(f"SEM bayes fit : {both_exp_choice_likelihood_on_testing_sets.exp1_bayesian_model_with_fitted__lik_weight__prior_weight__resp_bias.sem()}")
print(f"Mean diff : {bay_vs_opt_diff_exp1.mean()}")
print(f"SEM diff : {bay_vs_opt_diff_exp1.sem()}")
print(f"Cohen's d : {bay_vs_opt_diff_exp1.mean() / bay_vs_opt_diff_exp1.std()}")
print(f"T test : {scipy.stats.ttest_1samp(bay_vs_opt_diff_exp1,0)}")
print(f"Wilcoxon : {scipy.stats.wilcoxon(bay_vs_opt_diff_exp1)}")
print('')

bay_vs_opt_diff_exp3 = both_exp_choice_likelihood_on_testing_sets['exp3_bayesian_model_with_fitted__lik_weight__prior_weight__resp_bias']\
     - both_exp_choice_likelihood_on_testing_sets['exp3_optimal_bayesian']
print(f"Optimal vs BAYES-FIT_DECISION in exp3")
print(f"Median bayes fit : {both_exp_choice_likelihood_on_testing_sets['exp3_bayesian_model_with_fitted__lik_weight__prior_weight__resp_bias'].median()}")
print(f"Confidence intervale bayes fit : \
      {np.percentile(both_exp_choice_likelihood_on_testing_sets['exp3_bayesian_model_with_fitted__lik_weight__prior_weight__resp_bias'], [2.5, 97.5])}")
print(f"SEM bayes fit : {both_exp_choice_likelihood_on_testing_sets['exp3_bayesian_model_with_fitted__lik_weight__prior_weight__resp_bias'].sem()}")
print(f"Mean diff : {bay_vs_opt_diff_exp3.mean()}")
print(f"SEM diff : {bay_vs_opt_diff_exp3.sem()}")
print(f"Cohen's d : {bay_vs_opt_diff_exp3.mean() / bay_vs_opt_diff_exp3.std()}")
print(f"T test : {scipy.stats.ttest_1samp(bay_vs_opt_diff_exp3,0)}")
print(f"Wilcoxon : {scipy.stats.wilcoxon(bay_vs_opt_diff_exp3)}")
print('')


#%% COMPUTE THE DIFFERENCE OF CVLL BETWEEN MODELS (BAYES-FIT AND THE HEURISTIC)
# AND BETWEEN TASKS

# Define models that we will use
bay_exp1 = both_exp_choice_likelihood_on_testing_sets[
    'exp1_bayesian_model_with_fitted__lik_weight__prior_weight__resp_bias']
bay_exp3 = both_exp_choice_likelihood_on_testing_sets[
    'exp3_bayesian_model_with_fitted__lik_weight__prior_weight__resp_bias']
lin_exp1 = both_exp_choice_likelihood_on_testing_sets[
    'exp1_linear_model_with_fitted__lik_weight__prior_weight__resp_bias']
lin_exp3 = both_exp_choice_likelihood_on_testing_sets[
    'exp3_linear_model_with_fitted__lik_weight__prior_weight__resp_bias']
lin_int_exp1 = both_exp_choice_likelihood_on_testing_sets[
    'exp1_linear_model_with_fitted__interaction_weight__lik_weight__prior_weight__resp_bias']
lin_int_exp3 = both_exp_choice_likelihood_on_testing_sets[
    'exp3_linear_model_with_fitted__interaction_weight__lik_weight__prior_weight__resp_bias']


# Compute the difference between T1 and T2 of the difference of performances
# between the bayesian and linear models
print('DIFFERENCE OF CVLL BETWEEN BAYES-FIT-DECISION AND HEURISTIC MODEL')
diff_bay_lin_exp3 = bay_exp3 - lin_exp3
diff_bay_lin_exp1 = bay_exp1 - lin_exp1
tval1, pval1 = scipy.stats.ttest_1samp(diff_bay_lin_exp1, 0)
wval1, wpval1 = scipy.stats.wilcoxon(diff_bay_lin_exp1)
d_cohen1 = diff_bay_lin_exp1.mean()/diff_bay_lin_exp1.std()
ci1 = np.percentile(diff_bay_lin_exp1, [2.5, 97.5])
tval2, pval2 = scipy.stats.ttest_1samp(diff_bay_lin_exp3, 0)
wval2, wpval2 = scipy.stats.wilcoxon(diff_bay_lin_exp3)
d_cohen2 = diff_bay_lin_exp3.mean()/diff_bay_lin_exp3.std()
ci2 = np.percentile(diff_bay_lin_exp3, [2.5, 97.5])

print(f'mean diff BAYES-FIT-DECISION - HEURISTIC in T1 = {diff_bay_lin_exp1.mean()}')
print(f'sem of the diff BAYES-FIT-DECISION - HEURISTIC in T1 = {diff_bay_lin_exp1.sem()}')
print(f'sd of the diff BAYES-FIT-DECISION - HEURISTIC in T1 = {diff_bay_lin_exp1.std()}')
print(f'tvalue of the diff = {tval1}, pvalue= {pval1}')
print(f'wvalue of the diff = {wval1}, wpvalue= {wpval1}')
print(f'd cohen of the diff = {d_cohen1}')
print(f'95% confidence intervale of the diff = {ci1}')
print('')
print(f'mean diff BAYES-FIT-DECISION - HEURISTIC in T2 = {diff_bay_lin_exp3.mean()}')
print(f'sem of the diff BAYES-FIT-DECISION - HEURISTIC in T2 = {diff_bay_lin_exp3.sem()}')
print(f'sd of the diff BAYES-FIT-DECISION - HEURISTIC in T2 = {diff_bay_lin_exp3.std()}')
print(f'tvalue of the diff = {tval2}, pvalue= {pval2}')
print(f' wvalue of the diff = {wval2}, wpvalue= {wpval2}')
print(f' d cohen of the diff = {d_cohen2}')
print(f'95% confidence intervale of the diff = {ci2}')

# Compute the difference between-tasks
# of the difference of the BAYES-FIT-DECISION and the HEURISTIC models
print('')
print('')
print('DIFFERENCE BETWEEN TASKS OF THE DIFFERENCE OF CVLL BETWEEN BAYES-FIT-DECISION AND HEURISTIC MODEL')
tvalue, pvalue = scipy.stats.ttest_1samp(diff_bay_lin_exp3 - diff_bay_lin_exp1, 0)
wvalue, wpvalue = scipy.stats.wilcoxon(diff_bay_lin_exp3 - diff_bay_lin_exp1)
d_cohen = (diff_bay_lin_exp3 - diff_bay_lin_exp1).mean()/(diff_bay_lin_exp3 - diff_bay_lin_exp1).std()
ci = np.percentile(diff_bay_lin_exp3 - diff_bay_lin_exp1, [2.5, 97.5])
print(f'mean diff of diff T2 - diff T1 = {(diff_bay_lin_exp3 - diff_bay_lin_exp1).mean()}')
print(f'sd diff T2 - diff T1 = {(diff_bay_lin_exp3 - diff_bay_lin_exp1).std()}')
print(f'sem diff T2 - diff T1 = {(diff_bay_lin_exp3 - diff_bay_lin_exp1).sem()}')
print(f'wpvalue= {wpvalue}, d cohen= {d_cohen}')
print(f'tvalue= {tvalue}, pvalue= {pvalue}, wvalue= {wvalue}')
print(f'95% Confidence intervale = {ci}')

# Compute the difference of performance of bayesian models between T1 and T2
print('DIFFERENCE IN CVLL BETWEEN-TASKS FOR THE BAYES-FIT-DECISION MODEL')
diff_bay_exp13 = bay_exp3 - bay_exp1
tvalue, pvalue = scipy.stats.ttest_1samp(diff_bay_exp13, 0)
wvalue, wpvalue = scipy.stats.wilcoxon(diff_bay_exp13)
d_cohen = diff_bay_exp13.mean()/diff_bay_exp13.std()
print(f' mean bay_exp3 - bay_exp1 = {diff_bay_exp13.mean()}')
print(f'tvalue= {tvalue}, pvalue= {pvalue}, wvalue= {wvalue}, wpvalue= {wpvalue} d cohen= {d_cohen}')

# Compute the difference of performance of linear models between T1 and T2
print('DIFFERENCE IN CVLL BETWEEN-TASKS FOR THE HEURISTIC MODEL')
diff_lin_exp13 = lin_exp3 - lin_exp1
tvalue, pvalue = scipy.stats.ttest_1samp(diff_lin_exp13, 0)
wvalue, wpvalue = scipy.stats.wilcoxon(diff_lin_exp13)
d_cohen = diff_lin_exp13.mean()/diff_lin_exp13.std()
print(f'mean lin_exp3 - lin_exp1 = {diff_lin_exp13.mean()}')
print(f'tvalue= {tvalue}, pvalue= {pvalue}, wvalue= {wvalue}, wpvalue= {wpvalue}, d cohen= {d_cohen}')

#%% COMPUTE AND PLOT THE CORRELATION OF CVLL BETWEEN-TASK (FIG 5C)
# FOR THE BAYES-FIT-DECISION AND HEURISTIC MODEL
correlations = {}

#  Between-task correlations of CVLL for the BAYES-FIT-DECISION model
if len(bay_exp1) != len(bay_exp3):
    print('WARNING DIFFERENCE OF LENGTHS')
else:
    n_subjects = len(bay_exp1)

# BAYES-FIT-DECISION MODEL
rho_bay, pval_bay = spearmanr(bay_exp1, bay_exp3)
ols_bay = LinearRegression().fit(bay_exp1.values.reshape(-1, 1),
                                 bay_exp3.values.reshape(-1, 1))
tvalue_bay = (rho_bay * np.sqrt(n_subjects-2))/np.sqrt(1 - rho_bay**2)
correlations['BayesianModel_betweentask_rho'] = rho_bay
correlations['BayesianModel_betweentask_p'] = pval_bay
correlations['BayesianModel_betweentask_tvalue'] = tvalue_bay
correlations['BayesianModel_betweentask_slope'] = ols_bay.coef_[0][0]

# HEURISTIC MODEL
rho_lin, pval_lin = spearmanr(lin_exp1, lin_exp3)
ols_lin = LinearRegression().fit(lin_exp1.values.reshape(-1, 1),
                                 lin_exp3.values.reshape(-1, 1))
tvalue_lin = (rho_lin * np.sqrt(n_subjects - 2))/np.sqrt(1 - rho_lin**2)
correlations['LinearModel_betweentask_rho'] = rho_lin
correlations['LinearModel_betweentask_p'] = pval_lin
correlations['LinearModel_betweentask_tvalue'] = tvalue_lin
correlations['LinearModel_betweentask_slope'] = ols_lin.coef_[0][0]

# BEST MODEL
rho_best, pval_best = spearmanr(lin_exp1, bay_exp3)
correlations['BestModel_betweentask'] = rho_best
correlations['BestModel_betweentask_p'] = pval_best

# COMPUTE THE CORRELATION (HERE THE DISSOCIATION) BETWEEN-TASK OF THE DIFFERENCE IN CVLL
# BETWEEN THE BAYES-FIT-DECISION AND HEURISTIC MODEL
rho_diff, pval_diff = spearmanr(diff_bay_lin_exp3, diff_bay_lin_exp1)
correlations['BayesianLinearModelsDifference_betweentask'] = rho_diff
correlations['BayesianLinearModelsDifference_betweentask_p'] = pval_diff


# PLOT CORRELATIONS
fig = plt.figure()
plt.plot(diff_bay_lin_exp3, diff_bay_lin_exp1, '.', color='black',
         alpha=0.2, markersize=15, label=f'rho={rho_diff:.3f}, pval={pval_diff:.3f}')
plt.ylabel('Explicit context')
plt.xlabel('Implicit context')
plt.ylim([-0.06, 0.05])
plt.xlim([-0.06, 0.05])
plt.title('Relative predictive accuracy', y=1.05)
sns.despine()
plt.legend()

if args.save_figures:
    fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                             'svg', 'MainFigures',
                             'Correlation_of_out_performance_of_the_Bayesian_model.svg'),
                format='svg',
                bbox_inches='tight', pad_inches=1, dpi=1000)
    fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                             'png', 'MainFigures',
                             'Correlation_of_out_performance_of_the_Bayesian_model.png'),
                format='png',
                bbox_inches='tight', pad_inches=1, dpi=300)


#%% COMPUTE AND PLOT WITHIN AND BETWEEN TASKS CORRELATIONS OF CVLL OF THE BAYES-FIT-DECISION AND HEURISTIC MODELS (FIG 5B)
# Test whether the correlation of the difference of performances between the bayesian model
# and the linear model is stronger between folds than between experiments
from scipy.stats import spearmanr
from itertools import combinations

# BUILD A FRAM WITH THE CVLL BY FOLD FOR EACH MODEL
model_performances = pd.DataFrame(index=subjects_to_include_both_exp)

for exp in ['exp1', 'exp3']:
    cross_validation = pd.read_csv(filenames[exp], index_col=0).loc[subjects_to_include_both_exp, :]

    columns_for_lik_on_testing_set_by_fold \
        = [col for col in cross_validation.columns
           if (col.startswith(f'LL_adjusted_param_best_fit_testing_set')
               and 'mean_parameters_loglik_for_choices' in col)]
    for fold in ['fold0', 'fold1', 'fold2']:
        bay_model = 'bayesian_model_with_fitted__lik_weight__prior_weight__resp_bias'
        lin_model = 'linear_model_with_fitted__lik_weight__prior_weight__resp_bias'
        col_for_lik_on_testing_set = [col for col in columns_for_lik_on_testing_set_by_fold
                                      if fold in col][0]
        bay = cross_validation[cross_validation['model'] == bay_model][col_for_lik_on_testing_set]
        lin = cross_validation[cross_validation['model'] == lin_model][col_for_lik_on_testing_set]

        model_performances[f'diff_baylin_in_{exp}_{fold}'] = bay-lin
        model_performances[f'bayesian_{exp}_{fold}'] = bay
        model_performances[f'linear_{exp}_{fold}'] = lin

model_performances.dropna(inplace=True)
exp1_cols = [col for col in model_performances.columns if col.startswith('diff_baylin_in_exp1')]
exp3_cols = [col for col in model_performances.columns if col.startswith('diff_baylin_in_exp3')]
exp1_bay_cols = [col for col in model_performances.columns if col.startswith('bayesian_exp1')]
exp3_bay_cols = [col for col in model_performances.columns if col.startswith('bayesian_exp3')]
exp1_lin_cols = [col for col in model_performances.columns if col.startswith('linear_exp1')]
exp3_lin_cols = [col for col in model_performances.columns if col.startswith('linear_exp3')]

# COMPUTE THE TRUE CORRELATION
# Compute the mean CVLL on the 3 folds to have the mean CVLL in each task
true_btw_task_rho, pvalue = spearmanr(model_performances[exp1_bay_cols].mean(axis=1),
                                      model_performances[exp3_bay_cols].mean(axis=1))
true_wtn_task_rho = []
true_wtn_task_slope = []
for exp_cols in [exp1_bay_cols, exp3_bay_cols]:
    for col_a, col_b in combinations(exp_cols, 2):
        rho, pvalue = spearmanr(model_performances[col_a], model_performances[col_b])
        true_wtn_task_rho.append(rho)
        ols = LinearRegression().fit(model_performances[col_a].values.reshape(-1, 1),
                                     model_performances[col_b].values.reshape(-1, 1))
        slope = ols.coef_[0][0]
        true_wtn_task_slope.append(slope)
true_wtn_task_rho = np.mean(true_wtn_task_rho)
true_wtn_task_slope = np.mean(true_wtn_task_slope)

# COMPUTE THE BOOTSTRAPED CORRELATIONS
possible_subjects = model_performances.index.to_numpy()
boostrapped_correlations = pd.DataFrame()

for i in range(10): #000):
    resample_subjects = model_performances.sample(n=possible_subjects.shape[0],
                                                  axis=0, replace=True)
    resample_subjects.dropna(inplace=True)

    # BETWEEN-TASK CORRELATIONS
    # Of the performances of Bayesian models
    btw_task_rho, pvalue = spearmanr(resample_subjects[exp1_bay_cols].mean(axis=1),
                                     resample_subjects[exp3_bay_cols].mean(axis=1))
    boostrapped_correlations.loc[i, 'btw_task_rho_of_bay'] = btw_task_rho

    # Of the performances of Linear models
    btw_task_rho, pvalue = spearmanr(resample_subjects[exp1_lin_cols].mean(axis=1),
                                     resample_subjects[exp3_lin_cols].mean(axis=1))
    boostrapped_correlations.loc[i, 'btw_task_rho_of_lin'] = btw_task_rho

    # WITHIN-TASK CORRELATIONS
    # Intracontext correlation of the difference
    wtn_task_rho = {'bay': [],
                    'lin': []}
    # Of the performances of Bayesian models
    for exp_cols in [exp1_bay_cols, exp3_bay_cols]:
        for col_a, col_b in combinations(exp_cols, 2):
            rho, pvalue = spearmanr(resample_subjects[col_a], resample_subjects[col_b])
            wtn_task_rho['bay'].append(rho)
    # Of the performances of Linear models
    for exp_cols in [exp1_lin_cols, exp3_lin_cols]:
        for col_a, col_b in combinations(exp_cols, 2):
            rho, pvalue = spearmanr(resample_subjects[col_a], resample_subjects[col_b])
            wtn_task_rho['lin'].append(rho)
    for key in wtn_task_rho.keys():
        wtn_task_rho[key] = np.mean(wtn_task_rho[key])
        boostrapped_correlations.loc[i, f'wtn_task_rho_of_{key}'] = wtn_task_rho[key]

selected_cols = [
    'btw_task_rho_of_lin', 'btw_task_rho_of_bay',
    'wtn_task_rho_of_lin', 'wtn_task_rho_of_bay']

correlations_to_plot = boostrapped_correlations[selected_cols]
confidence_intervals = np.array([np.quantile(correlations_to_plot[col], [0.025, 0.975])
                                 for col in correlations_to_plot.columns])

ci = np.array([(correlations_to_plot.mean() - confidence_intervals[:, 0]).values,
               (confidence_intervals[:, 1] - correlations_to_plot.mean()).values])

plt.rcParams.update({'font.size': 30})
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.bar(range(len(correlations_to_plot.columns)),
       correlations_to_plot.mean(axis=0).values,
       color=MODEL_CROSS_VALIDATION_COLORS_CORRELATIONS,
       width=0.9, yerr=ci)
ax.set_xticks(range(len(correlations_to_plot.columns)))
ax.set_xticklabels(labels=['lin_btw', 'bay_btw',
                           'lin_wth', 'bay_wth'],
                   rotation=0, horizontalalignment='center')
sns.despine()
if args.save_figures:
    fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                             'svg', 'MainFigures',
                             f'Bootstrap_of_correlation_models.svg'), format='svg',
                bbox_inches='tight', pad_inches=1, dpi=1000)
    fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                             'png', 'MainFigures',
                             f'Bootstrap_of_correlation_models.png'), format='png',
                bbox_inches='tight', pad_inches=1, dpi=300)

#%% COMPUTE THE CORRELATION BETWEEN RELATIVE PREDICTIVE ACCURACY AND THE ACCURACY OF PRIOR REPORT
rho_and_predictive_accuracy_diff = pd.concat([
    both_exp_choice_likelihood_on_testing_sets.loc[:, cols_exp1].diff(axis=1).dropna(axis=1),
    both_exp_choice_likelihood_on_testing_sets.loc[:, cols_exp3].diff(axis=1).dropna(axis=1)], axis=1)

rho_and_predictive_accuracy_diff.columns = ['diff_lin_bay_expl_ctxt', 'diff_lin_bay_impl_ctxt']

rho_and_predictive_accuracy_diff['rho_expl_ctxt'] = \
    pd.read_csv(os.path.join(PATHS['PriorReportCorrelationsFolder'],
                             'prior_report_correlation_expl_context.csv'),
                             index_col=0)['rhos']
rho_and_predictive_accuracy_diff['rho_impl_ctxt'] = \
    pd.read_csv(os.path.join(PATHS['PriorReportCorrelationsFolder'],
                             'prior_report_correlation_impl_context.csv'),
                             index_col=0)['rhos']

rho_and_predictive_accuracy_diff.dropna(inplace=True)


rho, pval = scipy.stats.spearmanr(rho_and_predictive_accuracy_diff['rho_expl_ctxt'].values,
                                  rho_and_predictive_accuracy_diff['diff_lin_bay_expl_ctxt'].values)
ols = LinearRegression().fit(rho_and_predictive_accuracy_diff['rho_expl_ctxt'].values.reshape(-1, 1),
                             rho_and_predictive_accuracy_diff['diff_lin_bay_expl_ctxt'].values.reshape(-1, 1))
bias, slope = ols.intercept_, ols.coef_

print('EXPLICIT CONTEXT')
print(f'The correlation between the difference of predictive accuracy of the linear and the bayesian model')
print(f'with the precision of report is : rho={rho}   pval={pval}')

rho, pval = scipy.stats.spearmanr(rho_and_predictive_accuracy_diff['rho_impl_ctxt'].values,
                                  rho_and_predictive_accuracy_diff['diff_lin_bay_impl_ctxt'].values)
ols = LinearRegression().fit(rho_and_predictive_accuracy_diff['rho_impl_ctxt'].values.reshape(-1, 1),
                             rho_and_predictive_accuracy_diff['diff_lin_bay_impl_ctxt'].values.reshape(-1, 1))
bias, slope = ols.intercept_, ols.coef_
print('IMPLICIT CONTEXT')
print(f'The correlation between the difference of predictive accuracy of the linear and the bayesian model')
print(f'with the precision of report is : rho={rho}   pval={pval}')



#%% COMPUTE THE SUM AND MEAN cvLL OF THE DIFFERENT BAYESIAN MODELS (FIGURE S5)

exp = 'exp3'

cross_validation = pd.read_csv(filenames['exp3_bayes_var'], index_col=0)
cross_validation = cross_validation.loc[subjects_to_include_both_exp, :]

print(f" {exp} nb of participants = {len(cross_validation.index.unique())}")

# Compute the likelihood of each choice of the testing set under each adjusted model
choice_likelihood_on_testing_sets = pd.DataFrame()

columns_for_lik_on_testing_set_by_fold \
    = [col for col in cross_validation.columns
        if (col.startswith(f'LL_adjusted_param_best_fit_testing_set')
            and 'mean_parameters_loglik_for_choices' in col)]

columns_for_optimal_lik_on_testing_set_by_fold \
    = [col for col in cross_validation.columns
       if (col.startswith('LL_optimal_param_testing_set')
           and 'mean_parameters_loglik_for_choices' in col)]


models_to_plot = [model for model in cross_validation.model.unique()]

for model in models_to_plot:
    # Sum the likelihood of the three testing sets and compute the mean likelihood of a choice
    if 'optimal' in model:
        optimal_lik_by_subjects_complete_exp =\
               cross_validation[cross_validation['model'] == model][columns_for_optimal_lik_on_testing_set_by_fold].mean(axis=1)
        choice_likelihood_on_testing_sets[f'{exp}_optimal_bayesian'] = optimal_lik_by_subjects_complete_exp
    else:
        model_lik_by_subjects_complete_exp =\
           cross_validation[cross_validation['model'] == model][columns_for_lik_on_testing_set_by_fold].mean(axis=1)
        choice_likelihood_on_testing_sets[f'{exp}_{model}'] = model_lik_by_subjects_complete_exp

# Sort results according to the selected estimator (mean or median)
columns_sorted = choice_likelihood_on_testing_sets.median().sort_values().index

plot_both_exp_choice_likelihood_on_testing_sets = choice_likelihood_on_testing_sets[columns_sorted].drop('exp3_optimal_bayesian', axis=1)


#%% COMPARE BAYES-FIT-ALL TO BAYES-OPTIMAL
bayfitall_vs_opt_exp3 = choice_likelihood_on_testing_sets['exp3_bayesian_model_with_6_free_parameters__fix_']\
      - choice_likelihood_on_testing_sets['exp3_optimal_bayesian']
print(f"Optimal vs BAYES-FIT-ALL in exp3")
print(f"Median bayes fit all: {choice_likelihood_on_testing_sets['exp3_bayesian_model_with_6_free_parameters__fix_'].median()}")
print(f"Confidence intervale opt : {np.percentile(choice_likelihood_on_testing_sets['exp3_bayesian_model_with_6_free_parameters__fix_'], [2.5, 97.5])}")
print(f"SEM bayes fit all : {choice_likelihood_on_testing_sets['exp3_bayesian_model_with_6_free_parameters__fix_'].sem()}")
print(f"Mean diff : {bayfitall_vs_opt_exp3.mean()}")
print(f"SEM diff : {bayfitall_vs_opt_exp3.sem()}")
print(f"Cohen's d : {bayfitall_vs_opt_exp3.mean() / bayfitall_vs_opt_exp3.std()}")
print(f"T test : {scipy.stats.ttest_1samp(bayfitall_vs_opt_exp3,0)}")
print(f"Wilcoxon : {scipy.stats.wilcoxon(bayfitall_vs_opt_exp3)}")

#%% PLOT THE COMPARISON OF THE BAYESIAN MODELS (FIG S5)
figsize = (10, 10)
txtsize = 14
plt.rcParams.update({'font.size': txtsize})

bar_colors = [(98/255, 126/255, 172/255)]*8

model_pretty_names = {
 'exp3_bayesian_model_with_6_free_parameters__fix_': 'V, WLL, DL, WDP, WDL, DD',
 'exp3_bayesian_model_with_5_free_parameters__fix_volatility': 'WLL, DL, WDP, WDL, DD',
 'exp3_bayesian_model_with_5_free_parameters__fix_strength_evidence': 'V, DL, WDP, WDL, DD',
 'exp3_bayesian_model_with_5_free_parameters__fix_bias_evidence': 'V, WLL, WDP, WDL, DD',
 'exp3_bayesian_model_with_4_free_parameters__fix_bias_evidence__volatility': 'WLL, WDP, WDL, DD',
 'exp3_bayesian_model_with_4_free_parameters__fix_strength_evidence__volatility': 'DL, WDP, WDL, DD',
 'exp3_bayesian_model_with_3_free_parameters__fix_bias_evidence__strength_evidence__volatility': 'WDP, WDL, DD',
 'exp3_bayesian_model_with_4_free_parameters__fix_bias_evidence__strength_evidence': 'V, WDP, WDL, DD'}

medians = plot_both_exp_choice_likelihood_on_testing_sets.median().rename(model_pretty_names)
sems = plot_both_exp_choice_likelihood_on_testing_sets.sem().rename(model_pretty_names)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[5,1]})
fig.subplots_adjust(hspace=0.2)

ax1.bar(range(len(medians)), medians.values,
        color=bar_colors, yerr=sems.values, width=0.9)
ax2.bar(range(len(medians)), medians.values,
        color=bar_colors, yerr=sems.values, width=0.9)

# zoom-in / limit the view to different portions of the data
ax1.set_ylim([-0.4, -0.45])  # outliers only
ax2.set_ylim([0, -0.2])  # most of the data
ax1.set_xticklabels(np.round(ax1.get_yticks(),3), rotation=270, va='center')
ax2.set_xticklabels(np.round(ax2.get_yticks(),3), rotation=270, va='center')


ax1.spines.bottom.set_visible(False)
ax1.spines.top.set_visible(False)
ax2.spines.top.set_visible(False)
ax2.spines.left.set_visible(False)
ax1.spines.left.set_visible(False)
ax1.set_xticks([])
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
ax1.yaxis.tick_right()
ax2.yaxis.tick_right()

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=14,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([1], [0], transform=ax1.transAxes, **kwargs)
ax2.plot([1], [1], transform=ax2.transAxes, **kwargs)
plt.title("Model predictive accuracy",
              fontsize=txtsize+3, y=6)


# #%%
# fig, ax = plt.subplots(1, 1, figsize=figsize)
# ax.bar(range(len(medians)),
#        medians.values,
#        color=bar_colors,
#        yerr=sems.values,
#        width=0.9)

# # Improve legend
# plt.ylim([-0.38, -0.45])
# plt.yticks(rotation=90, va='center')

# plt.yticks(ticks=ax.get_yticks()[1:], labels=np.round(ax.get_yticks()[1:], 3),
#            va='center', rotation=90)
# plt.xticks(labels=medians.index, ticks=range(len(medians)), rotation=90, ha='center') #which='both',
# plt.tick_params(axis='x',  bottom=False, top=False, labelbottom=False)
# plt.tick_params(axis='y',  right=False, top=False, labelbottom=False) #which='both',
# ax.yaxis.tick_right()
# ax.yaxis.set_label_position("left")
# ax.set_ylabel("Model predictive accuracy",
#               fontsize=txtsize+3, y=0.63)
# ax.yaxis.tick_right()
# ax.yaxis.set_label_position("left")
# plt.gca().invert_yaxis()
# # ax.tick_params(axis='y', reset=True, labelrotation=180, length=10, width=3, colors='black', direction='inout')

# sns.despine(top=False, right=False, bottom=True, left=True)


if args.save_figures:
    compl = 'var_of_bay_models'
    fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                              'svg',  'MainFigures',
                              f'ModelCrossValidation_T2_{ANALYSIS_SPECIFIC_NAME}_{compl}_bars.svg'),
                format='svg', bbox_inches='tight', pad_inches=1, dpi=10000)

    fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                              'png',  'MainFigures',
                              f'ModelCrossValidation_T1T2_{ANALYSIS_SPECIFIC_NAME}_{compl}.png'), format='png',
                bbox_inches='tight', pad_inches=1, dpi=300)

#%% COMPARISON OF PAIRS OF MODELS (FIG S5 right part)
# Define parameters of each models
both_exp_legend_df_index \
    = LEARNING_PARAMETERS + ['lik_weight', 'prior_weight', 'resp_bias']
pretty_param_names = {'volatility': 'V',
                      'strength_evidence': 'WLL',
                      'bias_evidence': 'DL',
                      'lik_weight': 'WDL',
                      'prior_weight': 'WDP',
                      'resp_bias': 'DD'}

# Precise parameter status (fixed or adjusted)
both_exp_legend_df = pd.DataFrame(columns=plot_both_exp_choice_likelihood_on_testing_sets.columns,
                                  index=both_exp_legend_df_index)

for model_name in plot_both_exp_choice_likelihood_on_testing_sets.columns:
    fixed_parameters, adjusted_parameters, param_ID_if_fix, model_name_for_fig \
        = get_fitted_and_fixed_parameters(model_name)
    both_exp_legend_df.loc[adjusted_parameters, model_name] = 'fit'
    both_exp_legend_df.loc[fixed_parameters, model_name] = 'opt'

# Define palette
nb_of_cols_both_exp = len(both_exp_legend_df.columns)

# Define legend cells colors
cell_colors = []
for i in range(len(both_exp_legend_df.index)):
    cell_colors.append(bar_colors)

both_exp_legend_df.fillna('', inplace=True)
cell_colors = np.array(cell_colors)
for i_col, col in enumerate(both_exp_legend_df.columns):
    for i_row, row in enumerate(both_exp_legend_df.index):
        if both_exp_legend_df.loc[row, col] == 'fit':
            cell_colors[i_row, i_col] = np.array([0.38431373, 0.49411765, 0.6745098])
        if both_exp_legend_df.loc[row, col] == 'opt':
            cell_colors[i_row, i_col] = np.array([0, 0, 0])

# Compute t-test of the distribution of p(choice) for pairs of models
cross_val_ttest_pvalues = pd.DataFrame(index=plot_both_exp_choice_likelihood_on_testing_sets.columns,
                                       columns=plot_both_exp_choice_likelihood_on_testing_sets.columns)
cross_val_ttest_tvalues = pd.DataFrame(index=plot_both_exp_choice_likelihood_on_testing_sets.columns,
                                       columns=plot_both_exp_choice_likelihood_on_testing_sets.columns)

list_of_combinations = list(combinations(plot_both_exp_choice_likelihood_on_testing_sets.columns, 2))
for model1, model2 in list_of_combinations:
    tvalue, pvalue = scipy.stats.wilcoxon(plot_both_exp_choice_likelihood_on_testing_sets[model1].values -
                                          plot_both_exp_choice_likelihood_on_testing_sets[model2].values)
    cross_val_ttest_pvalues.loc[model1, model2] = np.log(pvalue)
    cross_val_ttest_pvalues.loc[model2, model1] = np.log(pvalue)
    cross_val_ttest_pvalues.loc[model1, model1] = 'i'
    cross_val_ttest_pvalues.loc[model2, model2] = 'i'
    cross_val_ttest_tvalues.loc[model1, model2] = tvalue
    cross_val_ttest_tvalues.loc[model2, model1] = tvalue

min_log_value = np.nanmin(cross_val_ttest_pvalues.replace('i', np.nan).values) - 20
max_log_value = np.nanmax(cross_val_ttest_pvalues.replace('i', np.nan).values) + 0.1
cross_val_ttest_pvalues.fillna(max_log_value)

# Change the annotations from float to stars
annotations = cross_val_ttest_pvalues.copy()
annotations = annotations.apply(lambda x: ['*' if (y not in ['i','s'] and np.exp(y) < 0.05) else '' for y in x])

cross_val_ttest_pvalues.replace('i', max_log_value, inplace=True)
cross_val_ttest_pvalues.replace('s', min_log_value, inplace=True)

cmap = sns.color_palette('Greys', as_cmap=True)
plt.rcParams.update({'font.size': txtsize})

fig = plt.figure(figsize=(10, 8), dpi=200)
heatmap_ax = sns.heatmap(cross_val_ttest_pvalues.fillna(1), cmap=cmap, vmin=np.log(0.3),
                         vmax=np.log(0.00000001), annot=annotations.fillna(1),  fmt='',
                         xticklabels=False, yticklabels=False, linecolor='white',
                         annot_kws={'color': 'black'})
colorbar = heatmap_ax.collections[0].colorbar
colorbar.set_ticks(colorbar.get_ticks())
colorbar.set_ticklabels([f"{round(nb,2)}"
                         if nb > 0.009 else f"{nb:.2E}" for nb in np.exp(colorbar.get_ticks())])

plt.suptitle('    Wilcoxon-test comparing model likelihoods', fontsize=txtsize+3, x=0.5, y=0.97)

# Create legend table
the_table = plt.table(cellText=both_exp_legend_df.values, cellColours=cell_colors, cellLoc='center',
                      rowLabels=[pretty_param_names[p] for p in both_exp_legend_df.index],
                      rowLoc='right', colLoc='center', alpha=0.6)
the_table.set_fontsize(txtsize)
plt.title('    Variations \n         of free parameters', fontsize=txtsize+3, x=-0.3, y=1.05)
the_table.auto_set_font_size(False)

vert_both_exp_legend_df = both_exp_legend_df.T.iloc[:, ::-1]
vert_cell_colors = cell_colors.transpose(1, 0, 2)[:, ::-1]
# Create legend table
the_table2 = plt.table(cellText=vert_both_exp_legend_df.values, cellColours=vert_cell_colors,
                       cellLoc='center',
                       # rowLabels=[pretty_param_names[p] for p in both_exp_legend_df.index],
                       rowLoc='right', colLoc='center', loc='left', alpha=0.6)
the_table2.set_fontsize(txtsize)
the_table2.auto_set_font_size(False)

cell_size = 0.1255
large_cell_size = 0.14  # 0.18

for k, cell in six.iteritems(the_table._cells):
    cell.set_edgecolor('white')
    cell.set_width(cell_size)
    cell.set_height(cell_size)
    if both_exp_legend_df.values[k] in ['opt']:
        cell.get_text().set_color('grey')

for k, cell in six.iteritems(the_table2._cells):
    cell.set_edgecolor('white')
    cell.set_width(cell_size-0.02)
    cell.set_height(cell_size)
    if vert_both_exp_legend_df.values[k] in ['opt']:
        cell.get_text().set_color('grey')

if args.save_figures:
    compl = 'var_of_bay_models'
    fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                             'svg',  'MainFigures',
                             f'ModelCrossValidation_T1T2_{ANALYSIS_SPECIFIC_NAME}_pvalues_{compl}.svg'),
                format='svg',
                bbox_inches='tight', pad_inches=1, dpi=1000)
    fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                             'png',  'MainFigures',
                             f'ModelCrossValidation_T1T2_{ANALYSIS_SPECIFIC_NAME}_pvalues_{compl}.png'),
                format='png',
                bbox_inches='tight', pad_inches=1, dpi=300)