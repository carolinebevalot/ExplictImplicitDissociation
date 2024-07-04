#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:31:38 2023

@author: caroline
"""
import argparse
import sys
import scipy
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge

computer = os.uname().nodename
parser = argparse.ArgumentParser(description='Parameter recovery')
parser.add_argument('-computer', type=str, default=computer,
                    help='Define on which computer the script is run to adapt paths')
parser.add_argument('-BayesianBattery_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery/',
                    help='Define the path for scripts directory')
parser.add_argument('-BayesianBattery_completements_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery_complements/',
                    help='Define the path for the data and results')
args = parser.parse_args()
sys.path.append(args.BayesianBattery_path)
sys.path.append(args.BayesianBattery_completements_path)
os.chdir(args.BayesianBattery_path)
print(os.getcwd())

from Analysis.AnalysisOfGorrillaData.VariablesForContextPaperAnalysis\
    import COLORS_FOR_TASK_COMPARISON

from PathsForContextPaper import define_paths_for_context_paper
PATHS = define_paths_for_context_paper(computer)

scatterplot_size = 15
scatterplot_alpha = 0.5
scatterplot_lw = 0.6
h_figsize = 4
v_figsize = 4
dpi_inlineplots = 500
penalization_suffix = 'pen'
logreg_weights_folder = PATHS['LogregWeightsFolder']
subjects_to_include_both_exp = pd.read_csv(PATHS['SubjectsList_ErraticAnswers'],
                                           index_col=0)['participant_ID'].dropna().to_numpy()

logreg_weights_subsets_exp1 \
    = pd.read_csv(f"{logreg_weights_folder}/exp1_ideal_observer_with_generative_priors_all_levels_"
                  + f"{penalization_suffix}.csv",
                  index_col=0).loc[subjects_to_include_both_exp, :]
logreg_weights_subsets_exp3 \
    = pd.read_csv(f"{logreg_weights_folder}/"
                  + f"exp3_ideal_observer_with_fitted_decision_and_learning_parameters_all_levels"
                  + f"_{penalization_suffix}.csv",
                  index_col=0).loc[subjects_to_include_both_exp, :]


report_df = pd.read_csv(PATHS['exp1_priorreport'], index_col=0)
report_df = report_df[report_df['participant_ID'].isin(subjects_to_include_both_exp)]

#%% COMPUTE FOR EACH SUBJECT THE CORELATION BETWEEN PRIOR REPORT AND ITS GENERATIVE VALUE

subject_rhos = pd.Series()
subject_slopes = pd.Series()
subject_ridge_slopes = pd.Series()
subject_rhos_pval = pd.Series()
for participant in report_df["participant_ID"].unique():
    participant_df = report_df[report_df['participant_ID'] == participant].dropna()
    if report_df.any().all():
        rho, pval = spearmanr(participant_df['pred_house'],
                              participant_df['reported_prior_value'])
        subject_rhos[participant] = rho
        subject_rhos_pval[participant] = pval

        ols = LinearRegression().fit(participant_df['pred_house'].values.reshape(-1, 1),
                                     participant_df['reported_prior_value'].values.reshape(-1, 1))
        bias, slope = ols.intercept_, ols.coef_[0][0]
        subject_slopes[participant] = slope

        clf = Ridge().fit(participant_df['pred_house'].values.reshape(-1, 1),
                          participant_df['reported_prior_value'].values.reshape(-1, 1))
        bias, slope = clf.intercept_, clf.coef_[0][0]
        subject_ridge_slopes[participant] = slope

pd.DataFrame({
    'rhos': subject_rhos,
    'slopes': subject_slopes,
    'ridge_slopes': subject_ridge_slopes,
    'rhos_pval': subject_rhos_pval}).to_csv(os.path.join(PATHS['PriorReportCorrelationsFolder'],
                                                         'prior_report_correlation_expl_context.csv'))

print('ttest =', scipy.stats.ttest_1samp(subject_rhos, 0))
print('wilcoxon test =', scipy.stats.wilcoxon(subject_rhos.values))
print('Cohens D against 0 =', subject_rhos.mean()/subject_rhos.std())
print('rho mean=', subject_rhos.mean())
print('rho median=', subject_rhos.median())
print('rho ci=', np.percentile(subject_rhos, [2.5, 97.5]))
print('rho sd=', subject_rhos.std())
print('sem=', subject_rhos.std()/np.sqrt(len(subject_rhos)))
print('median rho pval', subject_rhos_pval.median())
print('median slope', np.median(subject_slopes))
print('mean slope', np.mean(subject_slopes))
print('median ridge slope', np.median(subject_ridge_slopes))
print('mean ridge slope', np.mean(subject_ridge_slopes))
print('min, max pval=', subject_rhos_pval.min(), subject_rhos_pval.max())
print('n subjects rhos pval > 0.05=',
      len(subject_rhos_pval[subject_rhos_pval > 0.05]),
      f'(out of {len(subject_rhos_pval)})')
print('n subjects rhos pval < 0.05=',
      len(subject_rhos_pval[subject_rhos_pval < 0.05]),
      f'(out of {len(subject_rhos_pval)})')

# %% COMPUTE THE CORRELATION BETWEEN PRIOR REPORTS ACCURACY AND PRIORS WEIGHTS
corr_expl_ctxt = pd.read_csv(os.path.join(PATHS['PriorReportCorrelationsFolder'],
                                          'prior_report_correlation_expl_context.csv'),
                             index_col=0)
weight_priors = {
    'explicit': logreg_weights_subsets_exp1[logreg_weights_subsets_exp1['prior_type']
                                            == 'generative_priors']['priors_weight'].dropna()}

to_correlate = pd.concat([corr_expl_ctxt['rhos'], corr_expl_ctxt['slopes'],
                          weight_priors['explicit']], axis=1)
to_correlate.columns = ['rho_expl', 'slope_expl', 'weight_expl']

rho_priorweight_with_rhoexplpriors, pval_priorweight_with_rhoexplpriors \
    = spearmanr(to_correlate['weight_expl'], to_correlate['rho_expl'])


# %% PLOT BOTH CORRELATIONS (FIG. S1)
figure1, axes = plt.subplots(1, 1,
                            figsize=(h_figsize, v_figsize),
                            dpi=dpi_inlineplots)
significativity = 1.0e-200 if pval < 1.0e-200 else 0.005
axes.scatter(report_df['pred_house'].values,
             report_df['reported_prior_value'].values,
             color=COLORS_FOR_TASK_COMPARISON['priors_weight'][1],
             s=scatterplot_size, alpha=scatterplot_alpha)
axes.plot(0.2, 0.2, '.', color='white', label=f'rho = {round(subject_rhos.mean(), 3)}')
axes.plot(0.2, 0.2, '.', color='white', label=f'pval < {significativity} ')
axes.set_xlabel('Generative prior values')
axes.set_ylabel('Reported prior values')
axes.legend()
axes.set_title('Accuracy of report')
sns.despine()

figure2, axes = plt.subplots(1, 1,
                            figsize=(h_figsize, v_figsize),
                            dpi=dpi_inlineplots)
axes.scatter(to_correlate['weight_expl'].values,
             to_correlate['rho_expl'].values,
             color=COLORS_FOR_TASK_COMPARISON['priors_weight'][1],
             s=scatterplot_size, alpha=scatterplot_alpha)
axes.plot(0.2, 0.2, '.', color='white', label=f'rho = {round(rho_priorweight_with_rhoexplpriors, 3)}')
axes.plot(0.2, 0.2, '.', color='white', label=f'pval < {round(pval_priorweight_with_rhoexplpriors, 3)}')
axes.set_xlabel('Weight of explicit priors')
axes.set_ylabel('Accuracy of reports')
axes.legend()
axes.set_title('Accuracy and use of explicit priors')
sns.despine()

for nb, figure in enumerate([figure1, figure2]):
    figure.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                           'svg',
                           'SupplementaryFigures',
                           f"CorrelationBetweenPriorValueAndReport{nb}.svg"),
                   format='svg', bbox_inches='tight', pad_inches=0,
                   dpi=1200)
    figure.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                           'png',
                           'SupplementaryFigures',
                           f"CorrelationBetweenPriorValueAndReport{nb}.png"),
                   format='png', bbox_inches='tight', pad_inches=0,
                   dpi=1200)
