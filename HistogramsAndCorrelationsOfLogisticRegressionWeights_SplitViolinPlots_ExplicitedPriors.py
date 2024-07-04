#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:30:55 2022
This script create the figure 5.
It first loads subjects results (with fitted priors and reports), 
the weights of generative priors in T1 (on all trials) 
and the weights of fitted priors and of explicited priors in T2 
on trials after the explicit report.
The panel A. corresponds to the correlation between implicit prior value and reports
Trials are binned by density and the s.e.m. is shown.
The panel B. Corresponds to the comparison, in a violinplot, of the weights of 
explicited and implicit priors in T2. 
The panel C. corresponds to the correlation between the weights of 
explicit and explicted priors (plotted in logscale).
@author: caroline
"""
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy
import numpy as np
from scipy.stats import pearsonr, spearmanr
import sys
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.stats.api import DescrStatsW

computer = os.uname().nodename
parser = argparse.ArgumentParser(description="Plot figures comparing explicit report of priors and implicit priors")
parser.add_argument('--computer', type=str, default=computer,
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
sys.path.append(args.BayesianBattery_path)
sys.path.append(args.BayesianBattery_completements_path)
os.chdir(args.BayesianBattery_path)
print(os.getcwd())

from Analysis.AnalysisOfGorrillaData.VariablesForContextPaperAnalysis\
    import COLORS_FOR_TASK_COMPARISON, PRETTY_NAMES, PENALIZATION_SUFFIX
from Analysis.AnalysisOfGorrillaData.AnalysisCommonFunctions\
    import run_statistic_comparisons, plot_correlation, plot_violinplot
from PathsForContextPaper import define_paths_for_context_paper

PATHS = define_paths_for_context_paper(computer)

subjects_to_include_both_exp = pd.read_csv(PATHS['SubjectsList_ErraticAnswers'],
                                           index_col=0)['participant_ID'].dropna().to_numpy()
subjects_to_include_both_exp = [s for s in subjects_to_include_both_exp
                                if s != '613d383474d383b495180e85'] # logistic regression failed for explicit report

#%% DEFINE VARIABLES SPECIFIC TO THE TASK
behavior_to_plot = 'subject_choices'
weights_in_violinplot = ['posteriors_weight', 'priors_weight', 'likelihood_weight']
combination_type = 'bayesian'
free_parameters = 'fitted_decision_and_learning_parameters'
location_to_Q = 'after_Q'
weights_in_violinplot = ['priors_weight']
pen_suffix = PENALIZATION_SUFFIX

#%% LOAD LOGISTIC WEIGHTS AND SELECT COLUMNS TO KEEP IN THE FRAME
logreg_weights_around_reports_exp3 = {
    'after_Q': pd.read_csv(os.path.join(PATHS['LogregWeightsFolder'],
                                        f"exp3_implicit_explicit_logreg_after_Q_{pen_suffix}.csv"),
                           index_col=0, engine='python').loc[subjects_to_include_both_exp, :]}

concatenated_subjects_results_exp3 \
    = pd.read_csv(PATHS['exp3_data'], index_col=0, engine='python')
concatenated_subjects_results_exp3 \
    = concatenated_subjects_results_exp3[
        concatenated_subjects_results_exp3['participant_ID'].isin(subjects_to_include_both_exp)]

logreg_weights_folder = PATHS['LogregWeightsFolder']
logreg_weights_all_trials = {
    'exp1': pd.read_csv(os.path.join(logreg_weights_folder,
                                     f"exp1_ideal_observer_with_generative_priors_all_levels_{pen_suffix}.csv"),
                        index_col=0).loc[subjects_to_include_both_exp, :],
    'exp3': pd.read_csv(os.path.join(logreg_weights_folder,
                                     f"exp3_ideal_observer_with_fitted_decision_and_learning_parameters_all_levels_{pen_suffix}.csv"),
                        index_col=0).loc[subjects_to_include_both_exp, :]}

#%% DEFINE FIGURES PARAMETERS
plt.rcParams.update({'font.size': 14})
figsave_options = dict(format='svg', bbox_inches='tight', pad_inches=1, dpi=10000)
figsave_options_png = dict(format='png', bbox_inches='tight', pad_inches=1, dpi=300)

#%% ANALYSE THE CORRELATION BETWEEN PRIOR VALUES AND EXPLICIT REPORTS
filename = f"T1T2_gen_priors_vs_fitted_priors_trials_n{len(subjects_to_include_both_exp)}"
bin_nb = 10
idx_prior_reports \
    = concatenated_subjects_results_exp3[concatenated_subjects_results_exp3['report_prior_value'] == 1].index

priors = pd.DataFrame({
    'participant_ID': concatenated_subjects_results_exp3.loc[idx_prior_reports, 'participant_ID'].values,
    'impl_after_Q': concatenated_subjects_results_exp3.loc[idx_prior_reports+1, 'implicit_priors'].values,
    'gen_after_Q' :concatenated_subjects_results_exp3.loc[idx_prior_reports+1, 'generative_priors'].values,
    'expl_after_Q': concatenated_subjects_results_exp3.loc[idx_prior_reports+1, 'explicited_priors'].values})

# bin the dataframe per quantiles of implicit prior, and partcipants
sub_avg_priors = priors.groupby([pd.qcut(priors['impl_after_Q'], bin_nb), 'participant_ID']).mean()
sub_avg_priors.dropna(inplace=True)
# average over participants
sub_avg_priors.mean(level="impl_after_Q")

# bin the dataframe per quantiles of implicit prior
bin_priors = priors.groupby([pd.qcut(priors['impl_after_Q'], bin_nb)]).mean()
rho, pval = spearmanr(bin_priors["impl_after_Q"],
                      bin_priors["expl_after_Q"])
# Compute correlation
for prior_type in ['gen_after_Q', 'impl_after_Q']:
    print(prior_type)
    subject_rhos = pd.Series()
    subject_slopes = pd.Series()
    subject_ridge_slopes = pd.Series()
    subject_rhos_pval = pd.Series()
    for participant in priors["participant_ID"].unique():
        participant_priors = priors[priors['participant_ID'] == participant].dropna()
        if participant_priors.any().all():
            rho, pval = spearmanr(participant_priors[prior_type],
                                  participant_priors["expl_after_Q"])
            subject_rhos[participant] = rho
            subject_rhos_pval[participant] = pval
            ols = LinearRegression().fit(participant_priors[prior_type].values.reshape(-1, 1),
                                         participant_priors["expl_after_Q"].values.reshape(-1, 1))
            bias, slope = ols.intercept_, ols.coef_[0][0]
            subject_slopes[participant] = slope
            clf = Ridge().fit(participant_priors[prior_type].values.reshape(-1, 1),
                              participant_priors["expl_after_Q"].values.reshape(-1, 1))
            bias, slope = clf.intercept_, clf.coef_[0][0]
            subject_ridge_slopes[participant] = slope
    
    print('rho mean=', subject_rhos.mean())
    print('sem=', subject_rhos.std()/np.sqrt(len(subject_rhos)))
    print(f'Cohen D against 0 = {subject_rhos.mean()/subject_rhos.std()}')
    print('rho ci=', np.percentile(subject_rhos, [2.5, 97.5]))
    
    print('ttest =', scipy.stats.ttest_1samp(subject_rhos, 0))
    print('Wilcoxon test =', scipy.stats.wilcoxon(subject_rhos))
    
    print('n subjects rhos pval > 0.05=',
          len(subject_rhos_pval[subject_rhos_pval > 0.05]),
          f'(out of {len(subject_rhos_pval)})')
    print('rho sd=', subject_rhos.std())
    print('median rho pval', subject_rhos_pval.median())
    print('min, max pval=', subject_rhos_pval.min(), subject_rhos_pval.max())
    print(f'median slope = {np.median(subject_slopes)}')
    print(f'median ridge slope = {np.median(subject_ridge_slopes)}')

pd.DataFrame({
    'rhos': subject_rhos,
    'slopes': subject_slopes,
    'ridge_slopes': subject_ridge_slopes,
    'rhos_pval': subject_rhos_pval}).to_csv(os.path.join(PATHS['PriorReportCorrelationsFolder'],
                                                         'prior_report_correlation_impl_context.csv'))
#%% PLOT THE CORRELATION (FIG 4.A)
scatterplot, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=800)
ax.errorbar(sub_avg_priors.mean(level=f"impl_{location_to_Q}")[f"impl_{location_to_Q}"],
            sub_avg_priors.mean(level=f"impl_{location_to_Q}")[f"expl_{location_to_Q}"],
            xerr=sub_avg_priors.sem(level=f"impl_{location_to_Q}")[f"impl_{location_to_Q}"],
            yerr=sub_avg_priors.sem(level=f"impl_{location_to_Q}")[f"expl_{location_to_Q}"],
            fmt='.',  ecolor=COLORS_FOR_TASK_COMPARISON['after_Q'][3],
            ms=20, alpha=0.4, lw=2,
            color=COLORS_FOR_TASK_COMPARISON['after_Q'][3])
ax.plot(np.linspace(0.1, 0.9, 10), np.linspace(0.1, 0.9, 10), lw=0.5, color='black')
ax.set_title(f"{PRETTY_NAMES['after_Q']}", x=-0.1)
ax.set_ylabel('Reported prior values', x=-0.03)
ax.set_xlabel('Implicit prior values', y=-0.05)
ax.axis('scaled')
ax.plot(0, 0, color='white', label=f'rho = {np.nanmean(subject_rhos): .2f}')
ax.legend(loc=('lower right'), ncol=1, edgecolor='white', framealpha=0.1)
ax.set_ylim([0, 1])
ax.set_xlim([0, 1])
for side in ['top', 'right']:
    ax.spines[side].set_visible(False)

if args.save_figures:
    scatterplot.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                                     'svg',
                                     'MainFigures',
                                     f"Binned_correlation_explicit_implicit_{filename}.svg"),
                        **figsave_options)
    scatterplot.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                                     'png',
                                     'MainFigures',
                                     f"Binned_correlation_explicit_implicit_{filename}.png"),
                        **figsave_options_png)

#%% PLOT THE COMPARISON OF WEIGHTS OF IMPLICIT AND EXPLICITED PRIORS (FIG 4.B)
prior_column_name = 'explicited implicit'

weights_implicit_priors \
    = logreg_weights_around_reports_exp3[location_to_Q]\
    [logreg_weights_around_reports_exp3[location_to_Q]['prior_type'] == 'implicit_priors']
weights_explicited_priors \
    = logreg_weights_around_reports_exp3[location_to_Q]\
    [logreg_weights_around_reports_exp3[location_to_Q]['prior_type'] == 'explicited_priors']

logreg_options_label = 'repT2_fitallT2_priors_alllevels'

stats_for_violinplot = \
        run_statistic_comparisons(weights_explicited_priors, weights_implicit_priors,
                                  logreg_options_label, weights_in_violinplot)


violinplots = plot_violinplot(weights_explicited_priors, weights_implicit_priors,
                              weights_in_violinplot, prior_column_name,
                              free_parameters, stats_for_violinplot)

if args.save_figures:
    violinplots.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                                     'svg',
                                     'MainFigures',
                                     f"Violinplot_explicit_implicit_{filename}.svg"),
                        **figsave_options)
    violinplots.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                                     'png',
                                     'MainFigures',
                                     f"Violinplot_explicit_implicit_{filename}.png"),
                        **figsave_options_png)

#%% PLOT THE SAME COMPARISON FOR PRIORS BELOW AND ABOVE 0.5 (RESPONSE TO REVIEWER)
logreg_weights_around_reports_exp3 = {
    'pinf05_after_Q': pd.read_csv(os.path.join(PATHS['LogregWeightsFolder'],
                                               f"exp3_implicit_explicit_logreg_pinf05_after_Q_{pen_suffix}_mod.csv"),
                                  index_col=0, engine='python'),
    'psup05_after_Q': pd.read_csv(os.path.join(PATHS['LogregWeightsFolder'],
                                               f"exp3_implicit_explicit_logreg_psup05_after_Q_{pen_suffix}_mod.csv"),
                                  index_col=0, engine='python')}

prior_column_name = 'explicited implicit'

for prior_half in logreg_weights_around_reports_exp3.keys():
    weights_implicit_priors \
        = logreg_weights_around_reports_exp3[prior_half]\
        [logreg_weights_around_reports_exp3[prior_half]['prior_type'] == 'implicit_priors']
    weights_explicited_priors \
        = logreg_weights_around_reports_exp3[prior_half]\
        [logreg_weights_around_reports_exp3[prior_half]['prior_type'] == 'explicited_priors']

    logreg_options_label = 'repT2_fitallT2_priors_alllevels'

    stats_for_violinplot = \
        run_statistic_comparisons(weights_explicited_priors, weights_implicit_priors,
                                  logreg_options_label, weights_in_violinplot)

    violinplots = plot_violinplot(weights_explicited_priors, weights_implicit_priors,
                                  weights_in_violinplot, prior_column_name,
                                  free_parameters, stats_for_violinplot)
    plt.title(prior_half)

    print(stats_for_violinplot.loc[:, ['tvalue', 'tpvalue', 'D_cohen', 'wpvalue']])

    if args.save_figures:
        violinplots.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                                         'svg',
                                         'MainFigures',
                                         f"Violinplot_explicit_implicit_{filename}_{prior_half}.svg"),
                            **figsave_options)
        violinplots.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                                         'png',
                                         'MainFigures',
                                         f"Violinplot_explicit_implicit_{filename}_{prior_half}.png"),
                            **figsave_options_png)

#%% PLOT THE CORRELATION BETWEEN WEIGHTS OF EXPLICIT AND EXPLICITED PRIORS (FIG 4.C)
prior_column_name = 'explicit explicited'
scatterplots, correlation_strength\
        = plot_correlation(logreg_weights_all_trials['exp1'].dropna(subset=['priors_weight']),
                           weights_explicited_priors,
                           ['priors_weight'], prior_column_name)
if args.save_figures:
    scatterplots.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                                      'svg',
                                      f"Correlation_explicit_implicit_{filename}_logscale.svg"),
                         **figsave_options)
    scatterplots.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                                      'png',
                                      'MainFigures',
                                      f"Correlation_explicit_implicit_{filename}_logscale.png"),
                         **figsave_options_png)

    to_save = pd.concat([stats_for_violinplot, correlation_strength], axis=1)
    to_save.T.to_csv(os.path.join(PATHS['ContextPaperFiguresPathRoot'], 'csv',
                                  'Violinplot_and_correlation_T1T2_statistic_comparisons_'
                                  + f'n{len(subjects_to_include_both_exp)}'
                                  + f'_{logreg_options_label}_explicited_explicit.csv'))

#%% COMPUTE CORRELATIONS OF REPORTS FOR IMPLICIT PRIORS
correlations = pd.DataFrame()
for subject in subjects_to_include_both_exp:
    s_data = concatenated_subjects_results_exp3[concatenated_subjects_results_exp3['participant_ID'] == subject]
    s_data.dropna(subset=['prior_reports'], inplace=True)
    rho, pval = spearmanr(s_data['prior_reports'], s_data['generative_priors'])

    ols = LinearRegression().fit(s_data['prior_reports'].values.reshape(-1, 1),
                                 s_data['generative_priors'].values.reshape(-1, 1))
    tvalue = (rho * np.sqrt(len(s_data)-2))/np.sqrt(1 - rho**2)
    correlations.loc[subject, 'rho'] = rho
    correlations.loc[subject, 'pvalue'] = pval
    correlations.loc[subject, 'tvalue'] = tvalue
    correlations.loc[subject, 'slope'] = ols.coef_[0][0]

print('rho mean=', correlations.rho.mean())
print('sem=', correlations.rho.sem())
print(f'Cohen D against 0 = {correlations.rho.mean()/correlations.rho.std()}')
print('ttest =', scipy.stats.ttest_1samp(correlations.rho, 0))
print('Wilcoxon test =', scipy.stats.wilcoxon(correlations.rho))
