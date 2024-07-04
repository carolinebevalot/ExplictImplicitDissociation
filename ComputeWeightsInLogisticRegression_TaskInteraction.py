#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Fri Apr  8 11:29:57 2022

# @author: caroline

"""
Compute logistic regressions of the weights of priors and sensory likelihood for each subject
for both tasks at the same time having the tasks(=context) as a predictor
"""

import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import os
import sys

computer = os.uname().nodename
parser = argparse.ArgumentParser(
    description="Compute the weights of posteriors, priors, likelihood in answers")
parser.add_argument('-computer', type=str, default=os.uname().nodename,
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

from Analysis.AnalysisOfGorrillaData.VariablesForContextPaperAnalysis \
    import SIMULATION_OPTIONS_BOTH_EXP, PENALIZATION_SUFFIX, MODEL_OPTIONS_BOTH_EXP, COLORS_FOR_TASK_COMPARISON
from Analysis.AnalysisOfGorrillaData.AnalysisCommonFunctions import lo
from PathsForContextPaper import define_paths_for_context_paper

PATHS = define_paths_for_context_paper(computer)

concatenated_subjects_results\
    = {'exp1': pd.read_csv(PATHS['exp1_data'], index_col=0, engine='python'),
       'exp3': pd.read_csv(PATHS['exp3_data'], index_col=0, engine='python')}

pen_suffix = PENALIZATION_SUFFIX
subjects_to_include_both_exp = pd.read_csv(PATHS['SubjectsList_ErraticAnswers'],
                                           index_col=0)['participant_ID'].dropna().to_numpy()



#%%
def regression_analysis(exp_results, explicative_variables,
                        behavior_to_analyse, prior_type,
                        combination_type, free_parameters, evidence_level='all_levels',
                        pen_suffix='pen', compute_correlation_explicit_report=False):
    """Function which models with a logistic regression, weights of each factor
    in subjects'answers : the likelihood and the prior.
    Combination between priors and likelihood can be bayesian or linear (adding their interaction)
    A constant is add to capture a response bias"""

    penalty = 'l2' if pen_suffix == 'pen' else None

    results = []
    errors_nb = 0

    for subject in exp_results.participant_ID.unique():
        subject_data = exp_results[exp_results.participant_ID == subject].dropna()

        explicative_variables_df = subject_data[explicative_variables]

        variable_to_explain = subject_data[behavior_to_analyse]
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


NM_fitted_parameters = pd.read_csv(PATHS['NM_FittedParameters'], index_col=0).loc[subjects_to_include_both_exp, :]
fitted_param_stats = pd.DataFrame(index= NM_fitted_parameters.columns, columns=['mean', 'median', 'sem'])  
fitted_param_stats['mean'] = NM_fitted_parameters.mean()
fitted_param_stats['median'] = NM_fitted_parameters.median()
fitted_param_stats['sem'] = NM_fitted_parameters.sem()

fitted_param_stats.to_csv(PATHS['ContextPaperFiguresPathRoot']+'/stats_on_fitted_param.csv')
#%% TASK 1 AND TASK 2 : DECISION (+/- LEARNING)
# Fit weights of priors and likelihood in T1
logreg_weights_with_interactions = []
with_generative_priors = False
# Define variables that will be used
if with_generative_priors:
    bay_explicative_variables = ['lo_genT1T2_priors', 'lo_likelihood']
    other_explicative_variables = ['context', 'prior_context_interaction',
                                   'lik_context_interaction']

else:
    bay_explicative_variables = ['lo_genT1_inferedT2_priors', 'lo_likelihood']
    other_explicative_variables = ['context', 'prior_context_interaction',
                                   'lik_context_interaction']

# Specify model options
behavior_to_analyse = 'subject_choices'
evidence_level = 'all_levels'

if with_generative_priors:
    prior_column_name = 'genT1T2_priors'
    prior_type = 'generative_priors'
    posterior_column_name = 'post_with_genT1T2_priors'
else:
    prior_column_name = 'genT1_inferedT2_priors'
    prior_type = 'gen_and_inferred_priors'
    posterior_column_name = 'post_with_genT1_inferedT2_priors'

free_parameters = 'None_or_fitted_decision_and_learning_parameters'
combination_type = 'bayesian'

#%%
# Create a df with everything needed
## Variables used in the log-odd
concatenated_subjects_results['exp1'][prior_column_name] = \
    concatenated_subjects_results['exp1']['generative_priors']
concatenated_subjects_results['exp1'][posterior_column_name] = \
    concatenated_subjects_results['exp1']['posteriors_with_generative_priors']

if with_generative_priors:
    concatenated_subjects_results['exp3'][prior_column_name] = \
        concatenated_subjects_results['exp3']['generative_priors']
    concatenated_subjects_results['exp3'][posterior_column_name] = \
        concatenated_subjects_results['exp3']['posteriors_with_generative_priors']
else:
    concatenated_subjects_results['exp3']['genT1_inferedT2_priors'] = \
        concatenated_subjects_results['exp3']['inferred_priors_with_fitted_decision_and_learning_parameters']
    concatenated_subjects_results['exp3']['post_with_genT1_inferedT2_priors'] = \
        concatenated_subjects_results['exp3']['posteriors_with_fitted_decision_and_learning_parameters']

## Other variables
concatenated_subjects_results['exp1']['context'] = 0
concatenated_subjects_results['exp3']['context'] = 1

data_to_use = pd.concat([concatenated_subjects_results['exp1'],
                         concatenated_subjects_results['exp3']])\
    .loc[:, ['participant_ID', behavior_to_analyse, 'evidence', 'context',
             prior_column_name, 'likelihood', posterior_column_name]]

data_to_use[f'lo_{prior_column_name}'] = lo(data_to_use[prior_column_name])
data_to_use['lo_likelihood'] = lo(data_to_use['likelihood'])
data_to_use[f'lo_{posterior_column_name}'] = lo(data_to_use[posterior_column_name])
data_to_use['prior_context_interaction'] =\
        data_to_use.loc[:, [f'lo_{prior_column_name}', 'context']].product(axis=1)
data_to_use['lik_context_interaction'] =\
        data_to_use.loc[:, ['lo_likelihood', 'context']].product(axis=1)
data_to_use['post_context_interaction'] =\
        data_to_use.loc[:, [f'lo_{posterior_column_name}', 'context']].product(axis=1)

data_to_use.drop([prior_column_name, 'likelihood', posterior_column_name],
                 axis=1, inplace=True)

#%% LOGISTIC REGRESSIONS
#### For priors and likelihood
logistic_regression \
    = regression_analysis(data_to_use,
                          bay_explicative_variables + other_explicative_variables,
                          behavior_to_analyse,
                          prior_type, combination_type, free_parameters, evidence_level)
logistic_regression.rename({'context' : 'context_priorlik'}, axis=1, inplace=True)
logreg_weights_with_interactions.append(logistic_regression)

#### For posteriors
if with_generative_priors:
     bay_explicative_variables = ['lo_post_with_genT1T2_priors']
else:
    bay_explicative_variables = ['lo_post_with_genT1_inferedT2_priors']
other_explicative_variables = ['context', 'post_context_interaction']
logistic_regression \
    = regression_analysis(data_to_use,
                          bay_explicative_variables + other_explicative_variables,
                          behavior_to_analyse,
                          prior_type, combination_type, free_parameters, evidence_level)
logistic_regression.rename({'context' : 'context_post'}, axis=1, inplace=True)
logreg_weights_with_interactions.append(logistic_regression)

logreg_weights_with_interactions = pd.concat(logreg_weights_with_interactions)

logreg_filename = f"{PATHS['LogregWeightsFolder']}/logregweights_ContextInteraction_pen_with{prior_column_name}.csv"
logreg_weights_with_interactions.to_csv(logreg_filename)

#%%
logreg_filename = f"{PATHS['LogregWeightsFolder']}/logregweights_ContextInteraction_pen_with{prior_column_name}.csv"
logreg_weights_with_interactions = pd.read_csv(logreg_filename, index_col=0)

#%%
import matplotlib.pyplot as plt
import seaborn as sns

weights = pd.concat([logreg_weights_with_interactions[f'lo_{prior_column_name}'].dropna(),
                     logreg_weights_with_interactions['prior_context_interaction'].dropna(),
                     logreg_weights_with_interactions['lo_likelihood'].dropna(),
                     logreg_weights_with_interactions['lik_context_interaction'].dropna(),
                     logreg_weights_with_interactions['context_priorlik'].dropna()
                     ],
                     axis=1).loc[subjects_to_include_both_exp, :]


bar_colors = COLORS_FOR_TASK_COMPARISON['priors_weight']\
    + COLORS_FOR_TASK_COMPARISON['likelihood_weight']\
    + ['grey'] \
    + COLORS_FOR_TASK_COMPARISON['posteriors_weight']\
    + ['grey']

figsize = (6,6)  # if exp_to_plot == 'exp3' else (1.2, 10)
fig, ax = plt.subplots(1, 1, figsize=figsize)

x = range(len(weights.columns))
plt.bar(x,
        weights.mean(),
        yerr=weights.sem(),
        color=bar_colors)
plt.xticks(x, weights.columns, rotation=45, ha='right')
plt.ylim([0, 1])
sns.despine()

fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                          'svg',  'SupplementaryFigures',
                          f'task_logweights_interaction.svg'),
            format='svg', bbox_inches='tight', pad_inches=1, dpi=10000)

fig.savefig(os.path.join(PATHS['ContextPaperFiguresPathRoot'],
                          'png',  'MainFigures',
                          f'task_logweights_interaction.png'), format='png',
            bbox_inches='tight', pad_inches=1, dpi=300)

