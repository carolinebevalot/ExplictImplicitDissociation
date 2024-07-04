#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:32:23 2022

@author: caroline
"""

import argparse
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from Analysis.AnalysisOfGorrillaData.VariablesForContextPaperAnalysis\
    import COLORS_FOR_TASK_COMPARISON, PRETTY_NAMES
title_size = 18
text_size = 14
lims = {'priors_weight': np.array([-2, 6]),
        'likelihood_weight': np.array([-0.5, 2.5]),
        'posteriors_weight': np.array([-0.5, 2.5])}

h_figsize = 4
v_figsize = 4
precision = 3
wspace=0.4
violinplots_paireddiff_lines_alpha = 0.5
violinplots_paireddiff_lines_lw = 0.4
violinplots_paireddiff_markersize = 0.5
violinplots_edge_lw = 0.5
violinplots_alpha = 1
scatterplot_size = 15
scatterplot_alpha = 0.5
scatterplot_lw = 0.6
dpi_inlineplots=500

#%% FUNCITONS
def plot_correlation(logreg_weights_subsets_exp1, logreg_weights_subsets_exp3,
                     weights_to_correlate, prior_column_name, evidence_level='all_levels',
                     squeeze_before_plot=True, logreg_options_label=''):

    panels_IDs = ['C.'] if prior_column_name == 'explicit explicited' else ['D.', 'E.']
    alpha = 0.3 if prior_column_name == 'explicit explicited' else scatterplot_alpha 

    figure, axes = plt.subplots(1, len(weights_to_correlate), sharey=False,
                                figsize=(h_figsize*len(weights_to_correlate), v_figsize),
                                dpi=dpi_inlineplots)

    plt.subplots_adjust(wspace=wspace)
    axis_labels = ['Weights of explicit priors', 'Weights of reported priors'] if prior_column_name == 'explicit explicited'\
        else ['Explicit context', 'Implicit context']

    correlation_strength = pd.DataFrame(index=weights_to_correlate, columns=['R2', 'pvalue'])
    for i, weight_type in enumerate(weights_to_correlate):
        weights = pd.concat([logreg_weights_subsets_exp1.loc[:, weight_type].dropna(),
                             logreg_weights_subsets_exp3.loc[:, weight_type].dropna()], axis=1).dropna()
        weights.columns = axis_labels
        spearman_corr, pvalue = scipy.stats.spearmanr(weights[axis_labels[0]], weights[axis_labels[1]])
        len_weights = len(weights)
        stderr = 1.0 / math.sqrt(len_weights - 3)
        delta = 1.96 * stderr
        lower = math.tanh(math.atanh(spearman_corr) - delta)
        upper = math.tanh(math.atanh(spearman_corr) + delta)

        tvalue = (spearman_corr * np.sqrt(len(weights)-2))/np.sqrt(1- spearman_corr**2) 
        ols = LinearRegression().fit(weights[axis_labels[0]].values.reshape(-1, 1),
                                     weights[axis_labels[1]].values.reshape(-1, 1))
        bias, slope = ols.intercept_, ols.coef_[0][0]


        ax = axes[i] if type(axes) == np.ndarray else axes

        ax.scatter(weights[axis_labels[0]], weights[axis_labels[1]],
                   color=COLORS_FOR_TASK_COMPARISON[weight_type][1],
                   s=scatterplot_size, alpha=alpha)
        ax.set_xlabel(axis_labels[0], fontsize=text_size)
        ax.set_ylabel(axis_labels[1], fontsize=text_size)
        title = f'{panels_IDs[i]}' if prior_column_name == 'explicit explicited'\
            else f'{panels_IDs[i]} {PRETTY_NAMES[weight_type]}'
        ax.set_title(title, fontsize=title_size, x=-0.09, y=1.05, loc='left')

        ax.plot(0, 0, '.', color='white', label=f'rho = {spearman_corr:.3f}')
        ax.plot(0, 0, '.', color='white', label=f'pval = {pvalue:.3f}')
        ax.legend(loc=('lower right'), ncol=1, frameon=False, framealpha=0.1)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        ax.axis('equal')
        ax.plot(np.linspace(weights.min().min(), weights.max().max(),10), 
                np.linspace(weights.min().min(), weights.max().max(),10),
                '-', color='black', lw=scatterplot_lw)
        correlation_strength.loc[weight_type, ['spearmanR', 'pvalue', 'tvalue']] \
            = spearman_corr, pvalue, tvalue
        correlation_strength.loc[weight_type, ['spearman_ci_inf', 'spearman_ci_sup']] \
            = lower, upper
        correlation_strength.loc[weight_type, ['slope', 'bias']] \
            = slope, bias

    plt.suptitle(logreg_options_label, y=1.1, size=text_size)

    return figure, correlation_strength


def plot_violinplot(logreg_weights_subsets_exp1, logreg_weights_subsets_exp3,
                    weights_to_plot, prior_column_name, free_parameters,
                    statistic_comparisons, logreg_options_label='', evidence_level='all_levels'):

    panels_IDs = ['B.'] if prior_column_name == 'explicited implicit' else ['A.', 'B.', 'C.']
    alpha = 0.5 if prior_column_name == 'explicited implicit' else violinplots_alpha 
    figure, axes = plt.subplots(1, len(weights_to_plot), sharey=False,
                                figsize=(h_figsize*len(weights_to_plot), v_figsize),
                                dpi=dpi_inlineplots)
    plt.subplots_adjust(wspace=wspace)
    for i, weight_type in enumerate(weights_to_plot):
        ax = axes[i] if type(axes) == np.ndarray else axes
        to_plot = pd.concat([logreg_weights_subsets_exp1[f'{weight_type}'].dropna(),
                            logreg_weights_subsets_exp3[f'{weight_type}'].dropna()], axis=1).dropna()
        to_plot.columns = ['Reported', 'Implicit'] if prior_column_name == 'explicited implicit' else ['Explicit', 'Implicit']

        for expl_weight, impl_weight in to_plot.values:
            color = COLORS_FOR_TASK_COMPARISON[weight_type][1] if expl_weight < impl_weight else 'lightgrey'
            ax.plot(np.array([0, 0.2]), np.array([expl_weight, impl_weight]),
                    linewidth=violinplots_paireddiff_lines_lw,
                    alpha=violinplots_paireddiff_lines_alpha,
                    color=color)

        v1 = ax.violinplot(to_plot.iloc[:, 0],  positions=[0], points=len(to_plot.iloc[:, 0]),
                           showmeans=False, showextrema=False, showmedians=False)

        for b in v1['bodies']:
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
            b.set_color(COLORS_FOR_TASK_COMPARISON[weight_type][0])
            b.set_edgecolor('black')
            b.set_linewidth(violinplots_edge_lw)
            b.set_alpha(alpha)

        v2 = ax.violinplot(to_plot.iloc[:, 1], positions=[0.2], points=len(to_plot.iloc[:, 1]),
                           showmeans=False, showextrema=False, showmedians=False)

        for b in v2['bodies']:
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
            b.set_color(COLORS_FOR_TASK_COMPARISON[weight_type][1])
            b.set_edgecolor('black')
            b.set_linewidth(violinplots_edge_lw)
            b.set_alpha(violinplots_alpha)

        ax.plot(np.zeros((len(to_plot.iloc[:, 0]), 1)), to_plot.iloc[:, 0], '.',
                markersize=violinplots_paireddiff_markersize, color='black')
        ax.plot(np.array([0.2]*len(to_plot.iloc[:, 1])), to_plot.iloc[:, 1], '.',
                markersize=violinplots_paireddiff_markersize, color='black')
        ax.plot([-0.05, 0], np.repeat(to_plot.iloc[:, 0].median(),2), '-', linewidth=0.5, color='black')
        ax.plot([0.2, 0.25], np.repeat(to_plot.iloc[:, 1].median(),2), '-', linewidth=0.5, color='black')

        ax.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
        for side in ['top', 'right', 'bottom']:
            ax.spines[side].set_visible(False)

        x_label = 'Reported     Implicit' if prior_column_name == 'explicited implicit' else 'Explicit     Implicit'
        ax.set_xlabel(x_label, fontsize=text_size)
        if prior_column_name == 'explicited implicit':
            ax.set_ylabel(PRETTY_NAMES[weight_type])
        title = f'{panels_IDs[i]}' if prior_column_name == 'explicited implicit'\
            else f'{panels_IDs[i]} {PRETTY_NAMES[weight_type]}'

        ax.set_title(title, fontsize=title_size, x=-0.1, y=1.05, loc='left')
        ax.set_ylim(lims[weight_type])
        ax.set_yticklabels(ax.get_yticks())

        d_cohen = statistic_comparisons.loc[weight_type, 'D_cohen']
        ax.plot(0, 0, color='white', linewidth=0.01, label=f'd cohen = {d_cohen:.2f}')
        ax.legend(loc='upper right', edgecolor='white', framealpha=0.1)

    plt.suptitle(logreg_options_label, y=1.1, size=text_size)

    return figure

def lo(probability):
    """
    Compute the log-odd of a proprotion.

    Parameters
    ----------
    p : the proportion between 0 and 1 to turn into log-odds.

    Returns
    -------
    float, log-odd ratio.

    """
    probability = np.clip(probability, 0.01, 0.99)
    return np.log(probability/(1-probability))


def Z_score(data):
    """
    Compute the Z score of a series.

    Parameters
    ----------
    data : pd.Series
        data to Z score.

    Returns
    -------
    Z-scored data series

    """
    return (data - data.mean())/data.std(ddof=0)


def regression_analysis(exp_results, explicative_variables, behavior_to_analyse, prior_type,
                        combination_type, free_parameters, evidence_level='all_levels',
                        pen_suffix='pen', compute_correlation_explicit_report=False):
    """Function which models with a logistic regression, weights of each factor
    in subjects'answers : the likelihood and the prior.
    Combination between priors and likelihood can be bayesian or linear (adding their interaction)
    A constant is add to capture a response bias"""

    penalty = 'l2' if pen_suffix == 'pen' else None
    if evidence_level != 'all_levels':
        exp_results = exp_results[exp_results['evidence'] == evidence_level]

    results = []
    errors_nb = 0

    for subject in exp_results.participant_ID.unique():
        subject_data = exp_results[exp_results.participant_ID == subject]
        all_variables = subject_data[[behavior_to_analyse] + explicative_variables].dropna()

        if combination_type == 'bayesian':
            explicative_variables_df = lo(all_variables.loc[:, explicative_variables])
        elif combination_type == 'linear_with_interaction':
            explicative_variables_df = all_variables.loc[:, explicative_variables]
            explicative_variables_df['interaction'] \
                    = explicative_variables_df[explicative_variables[0]] \
                    * explicative_variables_df[explicative_variables[1]]
        elif combination_type == 'linear_without_interaction':
            explicative_variables_df = all_variables.loc[:, explicative_variables]
        else:
            print(f'The combination type {combination_type} is not implemented')

        variable_to_explain = all_variables.loc[:, behavior_to_analyse]
        explicative_variables_df.columns = ['posteriors_weight' if 'post' in var else 'priors_weight' if 'prior' in var else f'{var}_weight'
                                            for var in explicative_variables_df.columns]
        explicative_variables_df['response_bias'] = 1

        if ((explicative_variables_df == np.inf).any()).any():
            print('Infinite values in explicative variables will be replaced by min and max')
            maximums = explicative_variables_df[explicative_variables_df != np.inf].max()
            minimums = explicative_variables_df[explicative_variables_df != -np.inf].min()
            explicative_variables_df.replace(np.inf, maximums, inplace=True)
            explicative_variables_df.replace(-np.inf, minimums, inplace=True)

        try:
            result = LogisticRegression(penalty=penalty).fit(explicative_variables_df.values, variable_to_explain.values)
            df = pd.DataFrame(result.coef_, index=[subject], columns=explicative_variables_df.columns)
            df['participant_ID'] = subject
            if compute_correlation_explicit_report:
                df.loc[:,'explicited_implicit_priors_pearson_rho'] \
                    = pearsonr(subject_data['explicited_priors'].values,
                               subject_data['implicit_priors'].values)[0]
                df.loc[:,'explicited_implicit_priors_spearman_rho'] \
                    = spearmanr(subject_data['explicited_priors'].values,
                                subject_data['implicit_priors'].values)[0]

            results.append(df)
        except ValueError:
            errors_nb += 1

    print(f'Logistic regression failed for {errors_nb} subjects')

    logistic_regression = pd.concat(results)
    logistic_regression['prior_type'] = prior_type
    logistic_regression['combination_type'] = combination_type
    logistic_regression['evidence_level'] = evidence_level
    logistic_regression['behavior_to_analyse'] = behavior_to_analyse
    logistic_regression['free_parameters'] = free_parameters

    return logistic_regression


def fit_parameters_with_logistic_regression(subject_data, explicative_variables, behavior_to_analyse, prior_type,
                                            combination_type, pen_suffix='pen'):
    """Function which models with a logistic regression, weights of each factor
    in subjects'answers : the likelihood and the prior.
    Combination between priors and likelihood can be bayesian or linear (adding their interaction)
    A constant is add to capture a response bias"""

    penalty = 'l2' if pen_suffix == 'pen' else None

    results = []

    all_variables = subject_data[[behavior_to_analyse] + explicative_variables].dropna()

    if combination_type == 'bayesian':
        explicative_variables_df = lo(all_variables.loc[:, explicative_variables])
    elif combination_type == 'linear_with_interaction':
        explicative_variables_df = all_variables.loc[:, explicative_variables]
        explicative_variables_df['interaction'] \
                = explicative_variables_df[explicative_variables[0]] \
                * explicative_variables_df[explicative_variables[1]]
    elif combination_type == 'linear_without_interaction':
        explicative_variables_df = all_variables.loc[:, explicative_variables]  
    else:
        print(f'The combination type {combination_type} is not implemented')

    variable_to_explain = all_variables.loc[:, behavior_to_analyse]
    explicative_variables_df.columns = ['posteriors_weight' if 'post' in var else 'priors_weight' if 'prior' in var else f'{var}_weight'
                                        for var in explicative_variables_df.columns]
    explicative_variables_df['response_bias'] = 1

    try:
        result = LogisticRegression(penalty=penalty).fit(explicative_variables_df.values, variable_to_explain.values)
        logistic_regression = pd.DataFrame(result.coef_, columns=explicative_variables_df.columns)

    except ValueError:
        print('Logistic regression failed')
        logistic_regression = pd.DataFrame()


    return logistic_regression.squeeze().to_dict()


def run_statistic_comparisons(distributionT1, distributionT2, logreg_options,
                              weights_to_compare_between_ctx, diffs_to_test=None):
    """This function computes t-test, Cohen's D and Wilcoxon test to test whether
     the mean of the two distributions (T1 and T2) is different and what is the effect size.
     Is then uses Levene test to test whether the two populations have equal variances.
     Levene p value will be <0.05 in case of non equal variance.
     It finally computes the mean and std of each distribution and its difference to the IO.
     {logreg_options} precise which type of priors was used in the logistic regression
     for each distribution. It will be one column of the statistics dataframe.
    """
    priors_pretty_names = {'genT1': 'Generative priors',
                           'lingenT1': 'Generative priors linear combination',
                           'repT2': 'Reported priors',
                           'genT2': 'Generative priors',
                           'lingenT2': 'Generative priors linear combination',
                           'linoptT2': 'Optimal priors linear combination',
                           'optT2': 'Optimal priors',
                           'fitallT2': ' Priors fitted to subject behavior with the complete model',
                           'linfitallT2': ' Priors fitted to subject behavior with the linear complete model',
                           'fitbestT2': 'Priors fitted to subject behavior with the best model'}

    evidence_pretty_names = {'alllevels': 'All levels',
                             'obvious': 'Obvious trials',
                             'ambiguous': 'Ambiguous trials'}

    columns_for_stats_frame = ['logreg_options', 'priorsT1', 'priorsT2', 'Evidence',
                               'tvalue', 'tpvalue', 'D_cohen',
                               'wvalue', 'wpvalue',
                               'mean_diff', 'sd_diff', 'sem_diff',
                               'µ_T1', 'µ_T2',
                               'std_T1', 'std_T2',
                               'sem_T1', 'sem_T2',
                               'lvalue', 'lpvalue',
                               'delta_IO_T1', 'delta_IO_T2']

    statistic_comparisons = pd.DataFrame(index=weights_to_compare_between_ctx,
                                         columns=columns_for_stats_frame)

    paired_diffs = {}
    for weight_type in weights_to_compare_between_ctx:
        weights = pd.concat([distributionT1[weight_type].dropna(),
                             distributionT2[weight_type].dropna()],
                            axis=1).dropna()
        weights.columns = ['T1', 'T2']

        paired_diff_T1_T2 = weights['T1'] - weights['T2']
        paired_diffs[weight_type] = paired_diff_T1_T2
        statistic_comparisons.loc[weight_type, ['mean_diff', 'sd_diff', 'sem_diff']]\
            = paired_diff_T1_T2.mean(), paired_diff_T1_T2.std(), paired_diff_T1_T2.sem()
        statistic_comparisons.loc[weight_type, 'sem_diff']\
            = paired_diff_T1_T2.sem()
        statistic_comparisons.loc[weight_type, ['ci_inf_diff', 'ci_sup_diff']]\
            = np.percentile(paired_diff_T1_T2, [2.5, 97.5])

        tvalue, tpvalue \
            = scipy.stats.ttest_1samp(paired_diff_T1_T2.values, 0)
        statistic_comparisons.loc[weight_type, ['tvalue', 'tpvalue']]\
            = (tvalue, tpvalue)

        statistic_comparisons.loc[weight_type, 'D_cohen'] \
            = paired_diff_T1_T2.mean()/paired_diff_T1_T2.std()

        wvalue, wpvalue = scipy.stats.wilcoxon(paired_diff_T1_T2.values,
                                               alternative='two-sided')
        statistic_comparisons.loc[weight_type, ['wvalue', 'wpvalue']] \
            = (wvalue, wpvalue)

        lvalue, lpvalue = scipy.stats.levene(weights['T1'].values, weights['T2'].values)
        statistic_comparisons.loc[weight_type, ['lvalue', 'lpvalue']] \
            = (lvalue, lpvalue)

        statistic_comparisons.loc[weight_type, ["µ_T1", "µ_T2"]] \
            = (weights['T1'].mean(), weights['T2'].mean())

        statistic_comparisons.loc[weight_type, ["std_T1", "std_T2"]] \
            = (weights['T1'].std(), weights['T2'].std())
        statistic_comparisons.loc[weight_type, ["sem_T1", "sem_T2"]] \
            = (weights['T1'].sem(), weights['T2'].sem())

        # Reject the null hypothesis that distribution mean is equal to optimal if significant
        statistic_comparisons.loc[weight_type, ["delta_IO_T1"]]\
            = scipy.stats.ttest_1samp(weights['T1'].values, 1)[1]  # only pvalue
        statistic_comparisons.loc[weight_type, ["delta_IO_T2"]]\
            = scipy.stats.ttest_1samp(weights['T2'].values, 1)[1]  # only pvalue

        statistic_comparisons.loc[weight_type, 'diff_median_T1_T2']\
            = weights['T1'].median() - weights['T2'].median()
        statistic_comparisons.loc[weight_type, 'diff_mean_T1_T2']\
            = weights['T1'].mean() - weights['T2'].mean()

    paired_diffs = pd.DataFrame(paired_diffs)
    if diffs_to_test:
        diff_of_paired_diffs = paired_diffs.loc[:, diffs_to_test].diff(axis=1).dropna(axis=1)
        tval, pval = scipy.stats.ttest_1samp(diff_of_paired_diffs, 0)
        wval, wpval = scipy.stats.wilcoxon(np.hstack(diff_of_paired_diffs.values))
        statistic_comparisons.loc['diff_of_interctxt_diff', ['mean_diff', 'sd_diff', 'sem_diff']]\
            = np.concatenate([diff_of_paired_diffs.mean().values, diff_of_paired_diffs.std().values, diff_of_paired_diffs.sem().values])
        statistic_comparisons.loc['diff_of_interctxt_diff', ['tvalue', 'tpvalue']]\
            = (tval[0], pval[0])
        statistic_comparisons.loc['diff_of_interctxt_diff', ['wvalue', 'wpvalue']]\
            = (wval, wpval)
        statistic_comparisons.loc['diff_of_interctxt_diff', 'D_cohen']\
            = (diff_of_paired_diffs.mean()/diff_of_paired_diffs.std()).values
        statistic_comparisons.loc['diff_of_interctxt_diff', ['ci_inf_diff', 'ci_sup_diff']]\
            = np.percentile(diff_of_paired_diffs, [2.5, 97.5])

    statistic_comparisons.loc['lik_prior', ['lpvalue_T1_lik_prior']] \
        = scipy.stats.levene(distributionT1['likelihood_weight'].dropna().values,
                             distributionT1['priors_weight'].dropna().values)[1]  # only pvalue
    statistic_comparisons.loc['lik_prior', ['lpvalue_T2_lik_prior']] \
        = scipy.stats.levene(distributionT2['likelihood_weight'].dropna().values,
                             distributionT2['priors_weight'].dropna().values)[1]  # only pvalue

    statistic_comparisons['logreg_options'] = logreg_options
    statistic_comparisons['priorsT1'] = priors_pretty_names[logreg_options.split('_')[0]]
    statistic_comparisons['priorsT2'] = priors_pretty_names[logreg_options.split('_')[1]]
    statistic_comparisons['Evidence'] = evidence_pretty_names[logreg_options.split('_')[3]]

    return statistic_comparisons

def squeeze(x, exp=1/2):
    squeezed_x = x**exp
    squeezed_x[x < 0] = -((-x[x < 0]) ** exp)
    return squeezed_x

def unsqueeze(x, exp=2):
    squeezed_x = x**exp
    squeezed_x[x < 0] = -((-x[x < 0]) ** exp)
    return squeezed_x

def plot_binned_correlation(implicit_priors, explicited_priors, weights_to_correlate):

    figure, axes = plt.subplots(1, len(weights_to_correlate), sharey=False,
                                figsize=(4*len(weights_to_correlate), 4), dpi=800)
    plt.subplots_adjust(wspace=0.4)

    correlation_strength = pd.DataFrame(index=weights_to_correlate, columns=['R2', 'pvalue'])
    for i, weight_type in enumerate(weights_to_correlate):
        out, bins = pd.qcut(implicit_priors[weight_type], 10, retbins=True)
        weights = pd.concat([implicit_priors[weight_type],
                             explicited_priors[weight_type]], axis=1).dropna()
        weights.columns = ['implicit', 'explicited']
        weights['bins_of_density_implicit_priors'] = out

        bin_weights = weights.groupby('bins_of_density_implicit_priors').median()
        ax = axes[i] if type(axes) == np.ndarray else axes
        ax.errorbar(bin_weights['implicit'], bin_weights['explicited'],
                    color=COLORS_FOR_TASK_COMPARISON[weight_type][2], linewidth=0,
                    xerr=weights.groupby('bins_of_density_implicit_priors').sem()['implicit'],
                    yerr=weights.groupby('bins_of_density_implicit_priors').sem()['explicited'], elinewidth=1,
                    ecolor='black')
        ax.scatter(bin_weights['implicit'], bin_weights['explicited'], color=COLORS_FOR_TASK_COMPARISON[weight_type][2], s=60)
        ax.set_title(f'{PRETTY_NAMES[weight_type]}', x=-0.1)
        ax.set_ylabel('Explicited prior values', x=-0.03)
        ax.set_xlabel('Implicit prior values', y=-0.05)
        ax.axis('scaled')
        x2 = sm.add_constant(weights['implicit'])
        model = sm.OLS(weights['explicited'], x2)
        results = model.fit()
        R2 = results.rsquared
        pvalue = results.pvalues['implicit']
        ax.plot(0, 0, color='white', label=f'R² = {R2}')
        ax.legend(loc=('lower right'), ncol=1, edgecolor='white', framealpha=0.1)
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)

        correlation_strength.loc[weight_type, ['R2', 'pvalue']] = R2, pvalue

    return figure, correlation_strength
