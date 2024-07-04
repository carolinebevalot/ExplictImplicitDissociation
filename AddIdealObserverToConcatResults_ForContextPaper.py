#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 11:18:19 2021
This script compute the value, trial by trial, of the priors and the posteriors.
These variables can be computed with generative or inferred priors, with optimal or fitted parameter
The script also computes derivative from these variables such as surprise and prior entropy
@author: caroline
"""

# Define file configurations
import argparse
import numpy as np
import os
import pandas as pd
import sys

computer = os.uname().nodename
parser = argparse.ArgumentParser(
    description="Add to the data the predictions of the ideal observer at each trial")
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

from Analysis.AnalysisOfGorrillaData.VariablesForContextPaperAnalysis \
    import SIMULATION_OPTIONS_BOTH_EXP
from SimulationOptimization.VariablesForFit \
    import DISTRIBUTIONS_RESOLUTION,\
    DEFAULT_PARAMETERS_VALUES_BY_MODEL
from SimulationOptimization.OptimizationFunctions\
    import get_sequences_from_results, infer_category, initializer_for_forward_algorithm
from PathsForContextPaper import define_paths_for_context_paper

computer = os.uname().nodename
PATHS = define_paths_for_context_paper(computer)

#%% Import variables
concatenated_subjects_results = {
            'exp1': pd.read_csv(PATHS['exp1_data'], index_col=0),
            'exp3': pd.read_csv(PATHS['exp3_data'], index_col=0)}
for experiment in ['exp1', 'exp3']:
    concatenated_subjects_results[experiment].index \
        = range(len(concatenated_subjects_results[experiment]))

#%% Begin simulations for each subjects
# Get fitted parameters for both experiments
fitted_parameters = {
    'ideal_observer_with_fitted_decision_and_learning_parameters':
        pd.read_csv(PATHS['NM_FittedParameters'], index_col=0, engine='python')}

for key in fitted_parameters:
    fitted_parameters[key].columns = [col.replace('fit_', '')
                                      for col in fitted_parameters[key].columns]

for experiment in ['exp3']:
    inference_options = SIMULATION_OPTIONS_BOTH_EXP[experiment]

    for sequence_to_infer in inference_options.keys(): 
        model_options = {'distributions_resolution': DISTRIBUTIONS_RESOLUTION}
        model_options['model_type'] = inference_options[sequence_to_infer]['model_type']
        model_options['default_parameters'] \
            = DEFAULT_PARAMETERS_VALUES_BY_MODEL[model_options['model_type']]
        combination = 'linear' if 'linear' in model_options['model_type'] else 'bayesian'
        posterior_column_name = inference_options[sequence_to_infer]['posterior_column_name']
        prior_column_name = inference_options[sequence_to_infer]['prior_column_name']

        subjects_list = concatenated_subjects_results[experiment]['participant_ID'].unique()

        for subject in subjects_list:
            subject_idx_in_df \
             = concatenated_subjects_results[experiment]\
             [concatenated_subjects_results[experiment]['participant_ID'] == subject].index

            # We will use fitted decision (and learning) parameters if we use fitted priors
            # or if we have a linear combination since there is no optimal values in this case
            if 'fitted' in sequence_to_infer:
                parameters_to_use = fitted_parameters[sequence_to_infer].loc[subject, :]
            else:
                parameters_to_use = model_options['default_parameters']

            # Compute initial prior distribution and initial transition matrix
            initial_distributions = initializer_for_forward_algorithm(parameters_to_use,
                                                                      model_options)
            # Get sequences of evidence and generative priors
            gather_blocks_within_part = False if experiment == 'exp1' else True
            observed_sequences \
                = get_sequences_from_results(subject, concatenated_subjects_results[experiment],
                                             gather_blocks_within_part, drop_on_choices=False,
                                             return_dict=True)

            inferred_sequences \
                = infer_category(parameters_to_use, observed_sequences, initial_distributions,
                                 model_options)

            concatenated_subjects_results[experiment].loc[subject_idx_in_df, prior_column_name]\
                = np.hstack(inferred_sequences['p(c_k=H|obs(1:k-1))'])
            concatenated_subjects_results[experiment].loc[subject_idx_in_df, posterior_column_name]\
                = np.hstack(inferred_sequences['p(c_k=H|obs(1:k))'])
            if prior_column_name == 'inferred_priors_with_optimal_parameters':
                concatenated_subjects_results[experiment]\
                    .loc[subject_idx_in_df, 'ideal_observers_with_optimal_parameters_choices']\
                    = np.hstack(inferred_sequences['choices'])

            if sequence_to_infer == 'ideal_observer_with_fitted_decision_and_learning_parameters':
                concatenated_subjects_results[experiment]\
                    .loc[subject_idx_in_df, 'ideal_observers_with_fitted_decision_and_learning_parameters_choices']\
                    = np.hstack(inferred_sequences['choices'])

            if 'learning' in model_options['model_type']:
                concatenated_subjects_results[experiment].loc[subject_idx_in_df, 'sd_optimal_prior']\
                    = np.hstack(inferred_sequences['sd_p(c_k=H|obs(1:k-1))'])

            if ('fitted' in sequence_to_infer
                and sequence_to_infer != 'ideal_observer_with_fitted_decision_and_learning_parameters'):
                # We want two columns for posteriors :
                # one with fitted decision and learning parameter
                # and one with optimal decision parameters and fitted learning parameters
                posterior_column_name_opt_decision \
                    = inference_options[sequence_to_infer]['posterior_column_name_opt_decision']

                param_for_post_with_opt_decision = parameters_to_use.copy()
                opt_decision_parameters \
                    = pd.Series(DEFAULT_PARAMETERS_VALUES_BY_MODEL[model_options['model_type']
                                .replace('_and_learning', '')])
                param_for_post_with_opt_decision[opt_decision_parameters.index] \
                    = opt_decision_parameters

                inferred_sequences \
                    = infer_category(param_for_post_with_opt_decision, observed_sequences,
                                     initial_distributions, model_options)
                concatenated_subjects_results[experiment]\
                    .loc[subject_idx_in_df, posterior_column_name_opt_decision] \
                    = np.hstack(inferred_sequences['p(c_k=H|obs(1:k))'])

for exp in ['exp1', 'exp3']:
    for prior_type_exp3, suffix in [('inferred_priors_with_optimal_parameters',
                                     'opt_priors'), ('generative_priors', 'gen_priors')]:
        prior_name = 'generative_priors' if exp == 'exp1' else prior_type_exp3
        concatenated_subjects_results[exp][f'prior_lik_incongruency_{suffix}'] = [
            0 if (prior > 0.5 and lik > 0.5) or (prior < 0.5 and lik < 0.5) else
            1 if (prior > 0.5 and lik < 0.5) or (prior < 0.5 and lik > 0.5) else np.nan
            for prior, lik in zip(concatenated_subjects_results[exp][prior_name],
                                  concatenated_subjects_results[exp]['likelihood'])]

    prior_name \
        = 'generative_priors' if exp == 'exp1' else 'inferred_priors_with_optimal_parameters'
    concatenated_subjects_results[exp]['prior_lik_congruency_value'] = [
       (np.absolute(prior - 0.5) + np.absolute(lik - 0.5))
       if (prior >= 0.5 and lik >= 0.5) or (prior <= 0.5 and lik <= 0.5) else
       -np.absolute(prior - lik)
       if (prior >= 0.5 and lik <= 0.5) or (prior <= 0.5 and lik >= 0.5) else np.nan
       for prior, lik in zip(concatenated_subjects_results[exp][prior_name],
                             concatenated_subjects_results[exp]['likelihood'])]

    # obvious morphs <0.1463 or > 0.8571
    concatenated_subjects_results[exp]['evidence_4levels'] \
        = ['very_obvious' if ((lik < 0.05) or (lik > 0.95)) else
           'very_ambiguous' if ((lik > 0.05) and (lik < 0.95)) else np.nan
           for lik in concatenated_subjects_results[exp]['likelihood']]

    concatenated_subjects_results[exp]['strength_of_evidence']\
        = [np.absolute(lik - 0.5)
           for lik in concatenated_subjects_results[exp]['likelihood']]

    concatenated_subjects_results[exp]['surprise'] \
        = [-(np.log(lik*prior + (1-lik)*(1-prior)))
           for prior, lik in zip(concatenated_subjects_results[exp][prior_name],
                                 concatenated_subjects_results[exp]['likelihood'])]

    concatenated_subjects_results[exp]['prior_entropy'] \
        = [-(np.log(prior)*prior + np.log(1-prior)*(1-prior))
           for prior in concatenated_subjects_results[exp][prior_name]]

    concatenated_subjects_results[exp]['prior_strength'] \
        = [np.absolute(prior - 0.5)
           for prior in concatenated_subjects_results[exp][prior_name]]

    if exp == 'exp3':
        concatenated_subjects_results[exp]['confidence'] \
            = [-np.log(sd)
               for sd in concatenated_subjects_results[exp]['sd_optimal_prior']]

    concatenated_subjects_results[experiment].to_csv(PATHS[f'{experiment}_data'].replace('.csv', '_mod.csv'))
