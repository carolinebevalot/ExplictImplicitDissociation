#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# "Created on Mon Aug 16 18:51:52 2021
# @author: caroline


"""
Created on Wed Mar 18 11:07:16 2020
Functions to simulate choices and fit the model with the
Nelder-Mead or the L-BFGS-B algorithms from a predefined sequence of morphs and tones for one particular subject.
@author: carolinebevalot
"""
from itertools import combinations
import numpy as np
import pandas as pd
import pickle
import random
from scipy.optimize import minimize
import string

import IdealObserver.InferenceNoTone as io_no_tone
from SimulationOptimization.VariablesForFit import  \
    PLAUSIBLE_PARAMETER_VALUES, LEARNING_PARAMETERS, DECISION_PARAMETERS,\
    LINEAR_DECISION_PARAMETERS, GORRILLA_EXP_STRUCTURE, DEFAULT_PARAMETERS_VALUES_BY_MODEL


# %% Functions to get the sequence of evidence from subjects data
def create_part(sequence_by_blocks, flat=True):
    """
    Create lists of block trials that belong to the same part.

    This fonction takes a list of sequences (of observation, priors, etc.)
    corresponding to blocks and gather blocks belonging to the same part.

    Parameters
    ----------
    sequence_by_blocks : list of lists or array
        General list containing sublists by block
    flat : bool, optional
        If set to False, each part contains several lists that correspond to each block.
        If set to True, trials of one part are belonging to the same array.
        The default is True.

    Returns
    -------
    sequence_by_parts : list
        List of observations, priors, choices, etc that constitute one part of the experiment.

    """

    sequence_by_parts = []
    for part in GORRILLA_EXP_STRUCTURE:
        if flat:
            sequence_by_parts.append(np.hstack([sequence_by_blocks[i] for i in part]))
        else:
            sequence_by_parts.append([sequence_by_blocks[i] for i in part])
    return sequence_by_parts


def get_sequences_from_results(subject, concatenated_subjects_results, 
                               gather_blocks_within_part=True, drop_on_choices=False, return_dict=False):
    """
    Get observable and behavioral sequences from the datafile.

    This function get the sequences of evidence, choices made by the subject
    and generative priors if given to subjects.
    For convenience, it can return a list of these sequences or a dictionnary.

    Parameters
    ----------
    subject : str
        subject identity.
    concatenated_subjects_results : DataFrame
        DataFrame with all subjects results.
    gather_blocks_within_part : bool, optional
        Control whether sequences are returned by part (if True) or by blocks (if False).
        The default is True.
    return_dict : bool, optional
        Control whether sequences are returned as three lists or as a dictionnary. The default is False.

    Returns
    -------
    list or dictionnary
        sequence of evidence, choices and generative priors.
    """

    sequences = {'evidence': [], 'choices': [], 
                 'generative_priors': [], 'optimal_priors': []}
    sequences_by_block = {'evidence': [], 'choices': [],
                          'generative_priors': [],'optimal_priors': []}

    subject_df = \
        concatenated_subjects_results[concatenated_subjects_results.loc[:, 'participant_ID'] == subject]\
        .dropna(subset=['block_nb'])

    for block_name in subject_df['block_nb'].unique():
        if drop_on_choices:
            subject_block_df = subject_df[subject_df['block_nb'] == block_name].dropna(subset=['subject_choices'])
        else:
            subject_block_df = subject_df[subject_df['block_nb'] == block_name]
        sequences_by_block['evidence'].append(subject_block_df['likelihood'].values)
        sequences_by_block['choices'].append(subject_block_df['subject_choices'].values)
        sequences_by_block['generative_priors'].append(subject_block_df['prior_values'].values)
        if 'inferred_priors_with_optimal_parameters' in subject_block_df.columns:
            sequences_by_block['optimal_priors'].append(subject_block_df['inferred_priors_with_optimal_parameters'].values)

    for seq_type in sequences_by_block.keys():
        if gather_blocks_within_part:
            seq = create_part(sequences_by_block[seq_type])
        else:
            seq = sequences_by_block[seq_type]
        sequences[seq_type] = seq

    if return_dict:
        return sequences
    else:
        return sequences['evidence'], sequences['choices'], sequences['generative_priors']


#%% Functions to initiate minimization
def draw_a_set_of_parameter_values_from_parameters_distributions(free_parameters_list, parameters_distributions_type,
                                                                 paths):
    """
    This function draws a random set of values from the distribution of plausibles values
    of each parameters.
    The list of free parameters is provided in first argument.
    The distribution of plausible values are either 'uniform' distributions
    or 'gaussian&gamma' distributions as defined in parameters_ditributions_type.
    """

    random_set_of_parameter_values = {}

    if parameters_distributions_type == 'uniform':
        for parameter in free_parameters_list:
            random_set_of_parameter_values[parameter] = \
                random.uniform(PLAUSIBLE_PARAMETER_VALUES[parameter][0],
                               PLAUSIBLE_PARAMETER_VALUES[parameter][-1])
    else:
        distribution_param_values = get_moments_of_distribution_of_optimal_parameters(paths)
        for parameter in free_parameters_list:
            if parameter in ['volatility', 'strength_evidence', 'prior_weight', 'lik_weight']:
                random_set_of_parameter_values[parameter] = \
                       np.random.gamma(distribution_param_values[parameter][0], distribution_param_values[parameter][1])

            elif parameter in ['bias_evidence', 'resp_bias']:
                random_set_of_parameter_values[parameter] = \
                       np.random.normal(distribution_param_values[parameter][0], distribution_param_values[parameter][1])

    return random_set_of_parameter_values


def create_remap(remap_type='min_remap'):
    """
    Return 2 dictonnaries of lambda functions.

    remap_to_model_space: project the real number line to the parameter space (with its constraints)
    remap_to_real_space: transform the parameter value (with its constrains) into the minimization algorithm space of real numbers
    By using such a remapping inside the objective function, the optimization algorithm can
    explore smoothly the entire range of real number.
    """
    remap_to_model_space, remap_to_real_space = {}, {}
    parameters_names = LEARNING_PARAMETERS + LINEAR_DECISION_PARAMETERS + DECISION_PARAMETERS
    for parameter in parameters_names:
        remap_to_model_space[parameter] = lambda x: x  # R
        remap_to_real_space[parameter] = lambda x: x  # R

    remap_to_model_space['volatility'] = lambda x: (1 / (1 + np.exp(-0.1 * (x+10))))  # R to ]0, 1[
    remap_to_real_space['volatility'] = lambda x: (-1 / 0.1 * (np.log((1/x)-1))) - 10  # ]0, 1[ to R

    return remap_to_model_space, remap_to_real_space


def initializer_for_forward_algorithm(model_parameters, model_options):
    """
    Initialize the beta distribution and the transition matrix for the first trial.

    The beta distribution is the prior distribution for the first trial in the forward algorithm.
    """
    initial_distributions = {}
    # Get probability grid
    pgrid = np.linspace(0, 1, model_options['distributions_resolution'])
    initial_distributions['pgrid'] = pgrid

    # Initialize Alpha
    Alpha0 = io_no_tone.init_Alpha(model_parameters, pgrid)
    initial_distributions['Alpha0'] = Alpha0

    # Compute transition matrix
    TransMat = io_no_tone.compute_transition_matrix(Alpha0)
    initial_distributions['TransMat'] = TransMat

    return initial_distributions


def get_moments_of_distribution_of_optimal_parameters(paths):
    with open(paths['moments_of_distribution_of_optimal_parameters_file'], 'rb') as file:
        moments_of_distribution_of_optimal_parameters = pickle.load(file)
    return moments_of_distribution_of_optimal_parameters


def get_moments_of_distribution_of_subjects_fitted_parameters(paths):
    file = paths['moments_of_distribution_of_subjects_fitted_parameters_file']
    moments_of_distribution_of_subjects_fitted_parameters = pd.read_csv(file, index_col=0, engine='python')
    return moments_of_distribution_of_subjects_fitted_parameters


#%% Fonctions for logliklihood minimization
def infer_category(model_parameters, observed_sequences, initial_distributions, model_options=None, print_parameters=False):
    """
    Simulate choices for a a sequence of observations +- given priors with a set of parameters.

    previously named simulate_behavior
    The function compute the sequence of priors and posteriors with the ideal observer
    and sample choices from the posterior probability of choices.
    The model used can be a bayesian or a liner model.
    """
    if not model_options:
        model_options = {'model_type': 'bayesian_model_of_decision_and_learning',
                         'default_parameters': DEFAULT_PARAMETERS_VALUES_BY_MODEL['bayesian_model_of_decision_and_learning'],
                         'fixed_values': DEFAULT_PARAMETERS_VALUES_BY_MODEL['bayesian_model_of_decision_and_learning']}

    # Complete the set of parameters with optimal values if the some parameter are fixed
    for parameter in model_options['default_parameters'].keys():
        if parameter not in model_parameters.keys():
            model_parameters[parameter] = model_options['default_parameters'][parameter]

    # Compute priors, posteriors and choices for each part containing several blocks
    inferred_sequences = {'p(c_k=H|obs(1:k))': [],
                          'p(c_k=H|obs(1:k-1))': [],
                          'choices': [],
                          'sd_p(c_k=H|obs(1:k-1))': []}

    for i, subdivision in enumerate(observed_sequences['evidence']):
        if 'learning' not in model_options['model_type']:
            res = {'p(c_k=H|obs(1:k-1))': observed_sequences['generative_priors'][i]}
            res['sd_p(c_k=H|obs(1:k-1))'] = None
        else:
            res = io_no_tone.compute_inference(subdivision, model_parameters, initial_distributions)
        res['p(c_k=H|obs(1:k))'] = \
            io_no_tone.compute_choice_probability(subdivision,  model_parameters, res['p(c_k=H|obs(1:k-1))'],
                                                  model_type=model_options['model_type'])['p(c_k=H|obs(1:k))']

        res['choices'] = io_no_tone.sample_choice_from_posterior(res['p(c_k=H|obs(1:k))'])

        for seq_type in inferred_sequences.keys():
            inferred_sequences[seq_type].append(res[seq_type])

    return inferred_sequences


def compute_prior_lik_combination(model_parameters, observed_sequences, model_options):
    """
    Simulate choices for a a sequence of observations +- given priors with a set of parameters.

    previously named simulate_behavior
    The function compute the sequence of priors and posteriors with the ideal observer
    and sample choices from the posterior probability of choices.
    The model used can be a bayesian or a liner model.
    model_options is a dictionnary with the following entries
    {'model_type': 'linear_without_interaction' or 'linear_with_interaction'or 'bayesian'
     'context': 'implicit_context' or 'explicit_context'}
    """
        

    # Compute priors, posteriors and choices for each part containing several blocks
    inferred_sequences = {'p(c_k=H|obs(1:k))': [],
                          'p(c_k=H|obs(1:k-1))': [],
                          'choices': []}

    for i, subdivision in enumerate(observed_sequences['evidence']):
        if model_options['context'] == 'explicit_context':
            res = {'p(c_k=H|obs(1:k-1))': observed_sequences['generative_priors'][i]}
        elif model_options['context'] == 'implicit_context':
            res = {'p(c_k=H|obs(1:k-1))': observed_sequences['optimal_priors'][i]}
        else:
            print('This context does not exist')
            
        res['p(c_k=H|obs(1:k))'] = \
            io_no_tone.compute_choice_probability(subdivision,  model_parameters, res['p(c_k=H|obs(1:k-1))'],
                                                  model_type=model_options['model_type'])['p(c_k=H|obs(1:k))']

        res['choices'] = io_no_tone.sample_choice_from_posterior(res['p(c_k=H|obs(1:k))'])

        for seq_type in inferred_sequences.keys():
            inferred_sequences[seq_type].append(res[seq_type])

    return inferred_sequences


def compute_choice_loglikelihood_given_parameters(sequence_of_choices, posterior_probabilities_of_choices,
                                                  verbose=False):
    """
    Compute the log likelihood of parameters for a sequence of choices
    that is the probability of choices given the model and the parameters.

     # previously named compute_parameters_loglikelihood_for_choices
    The function subdivide each experiment
    into blocks or parts to apply the ideal observeur model with the given parameters.
    """
    # Compute log likelihood of subject's choices given the fitted model

    posterior_probabilities_of_choices \
        = np.clip(np.hstack(posterior_probabilities_of_choices), 0.0001, 0.9999)
    
    sequence_of_choices = np.hstack(sequence_of_choices)
    len_seq_choices = len(sequence_of_choices[~np.isnan(sequence_of_choices)])
    parameters_lik_for_choices \
        = [p if choice == 1 else (1-p) if choice == 0 else np.nan
           for p, choice in zip(posterior_probabilities_of_choices, sequence_of_choices)]
    sum_parameters_lik_for_choices = np.nansum(parameters_lik_for_choices)
    mean_parameters_lik_for_choices = sum_parameters_lik_for_choices/len_seq_choices
    
    parameters_loglik_for_choices \
        = [np.log(p) if choice == 1 else np.log(1-p) if choice == 0 else np.nan
           for p, choice in zip(posterior_probabilities_of_choices, sequence_of_choices)]
    sum_parameters_loglik_for_choices = np.nansum(parameters_loglik_for_choices)
    mean_parameters_loglik_for_choices = sum_parameters_loglik_for_choices/len_seq_choices
    
    # Return joint log likelihood (sum of log likelihood)
    if verbose:
        return {'parameters_lik_for_choices': '_'.join([str(n) for n in parameters_lik_for_choices]), 
                'sum_parameters_lik_for_choices': sum_parameters_lik_for_choices,
                'mean_parameters_lik_for_choices': mean_parameters_lik_for_choices,
                'parameters_loglik_for_choices': '_'.join([str(n) for n in parameters_loglik_for_choices]), 
                'sum_parameters_loglik_for_choices': sum_parameters_loglik_for_choices,
                'mean_parameters_loglik_for_choices': mean_parameters_loglik_for_choices,
                'len_seq_choices': len_seq_choices}
    else:
        return sum_parameters_loglik_for_choices


def objective_fun(model_parameters_in_real_space,  *args):
    """
    Compute the negative loglikelihood function to minimize.

    The function transforms the algorithm initial guess about the parameter values
    from the real space to the parameter space.
    It calls the function to compute the loglikelihood
    and turns it into negative loglikelihood.
    """
    sequence_of_choices, observed_sequences,\
        parameters_remap_to_model_space, initial_distributions, model_options = args

    model_parameters_in_model_space = \
        {parameter: parameters_remap_to_model_space[parameter](model_parameters_in_real_space[i])
         for i, parameter in enumerate(model_options['free_parameters_list'])}

    # Compute sequences of prior and posterior probabilities of house given the model parameters to evaluate
    inferred_sequences = infer_category(model_parameters_in_model_space, observed_sequences,
                                        initial_distributions, model_options)
    parameters_neg_loglikelihood \
        = - compute_choice_loglikelihood_given_parameters(sequence_of_choices,
                                                          inferred_sequences['p(c_k=H|obs(1:k))'])

    return parameters_neg_loglikelihood


#%% Wrapper for the minimization         
def find_best_set_of_parameter_values_to_explain_choices(sequence_of_choices, observed_sequences, 
                                                         model_options, minimization_options, paths):
    """
    This fitting function look for the set of model free parameters that minimize the negative
    loglikelihood of the sequence of choices given the observerved sequence of evidence.

    The sequence of evidence is observed_sequence['evidence'].

    The list of free parameters if found in model_options['free_parameters_list'].
    The option model_options['manage_absent_parameters'] explain what to do
    if the list of provided free parameters is smaller than the list of possible
    free parameters. Parameters are :
        - fixed to the optimal value if 'fix_to_optimal'
        - provided with the value of another parameter if 'combine'.

    The model used to simulate choices can be a bayesian or a linear model and
    it can simulate decision only (task 1) or decision and learning (task 2).
    The type of model is defined in model_options['model_type'] which can be
    for example 'bayesian_model_of_decision_and_learning'.

    The resolution used for distribution is defined in model_options['distributions_resolution']

    The fitting function uses scipy.minimize which tests parameter values from the real space.
    As there are constraints on some parameters of the model, the function uses
    a set of transformations (parameters_remap_to_model_space) that project
    the real number line onto the parameter space and a set of the inverse
    transformations (parameters_remap_to_real_space).
    This remap can either be minimal with minimization_options['remap_type'] set
    to 'min_remap' (only the volatility is in ]0,1[) or it can constrains all parameters :
        - volatility to [0, 0.3]
        - other parameters in the range [-5;5] with minimization_options['remap_type']='remap_range_5_5'
        - other parameters in the range [-10,40] with minimization_options['remap_type']='remap_range_10,40'
        - bias_evidence and resp_bias in [-1;1] and the others in [0.19;5]  
                with minimization_options['remap_type']='remap_two_ranges'

    The method (optimization algorithm) is defined by with minimization_options['method']
    and can be :
            'Nelder-Mead': Simplex algorithm
            'BFGS': quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno
    A maximum of iterations (N) for either optimization methods can be defined in
    minimization_options['options']={'maxiter': N}.
    
    The optimization algorithm starts with an a set of initial guesses about 
    parameter values. These guesses are drawn from the distribution of plausible
    values for each parameter. The distribution can be a uniform distribution
    to explore a wide range of values or a gaussian or gamma distribution to 
    select values closer to optimal. This option 'uniform' or 'gaussian&gamma' 
    is set in minimization_options['distribution_type_for_initial_guess']
    
    To avoid finding a local minimum, the algorithm is restarted n times.
    n is defined in minimization_options['restart_n_times'].  
   
    """
    parameters_remap_to_model_space, parameters_remap_to_real_space = \
                                create_remap(minimization_options['remap_type'])

    fits_from_every_iterations = []
    sim_ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 8))
    
    for k_iteration in range(minimization_options['restart_n_times']):
        # Draw randomly a set of learning parameters that will be the starting point 
        # of the optimization algorithm and remap them into the algorithm space       
        initial_guess_about_parameters \
           = draw_a_set_of_parameter_values_from_parameters_distributions(model_options['free_parameters_list'],
                                                                          minimization_options['distribution_type_for_initial_guess'],
                                                                          paths)
        # Compute initial prior distribution and initial transition matrix 
        initial_guess_about_parameters_in_real_space\
            = {parameter: parameters_remap_to_real_space[parameter](initial_guess_about_parameters[parameter])
                        for parameter in initial_guess_about_parameters.keys()}

        initial_distributions \
            = initializer_for_forward_algorithm(initial_guess_about_parameters, model_options)

        # Fit learning and the decision parameters with the optimization algorithm 
        fit = minimize(objective_fun, list(initial_guess_about_parameters_in_real_space.values()),
                       args=(sequence_of_choices, observed_sequences, 
                             parameters_remap_to_model_space, initial_distributions, model_options),
                       method=minimization_options['method'], options=minimization_options['options'])
                             
            
        # Save result of the fit of learning and decision parameters with their starting point
        fits_from_every_iterations.append(
            {'negative_loglikelihood': fit['fun'], 
             'initial_guess_about_parameters': initial_guess_about_parameters,
             'fitted_parameters': {parameter: parameters_remap_to_model_space[parameter](fit.x[i])
                                for i, parameter in enumerate(model_options['free_parameters_list'])},
             'fit': fit,
             'sim_ID': sim_ID})
    
    for fit_nb in range(len(fits_from_every_iterations)):
        fits_from_every_iterations[fit_nb]['fit_status'] = \
            'best_fit' if fit_nb == np.argmin([iteration['negative_loglikelihood'] for iteration in fits_from_every_iterations]) else 'bad_fit'
    
    return {'all_fits' : fits_from_every_iterations,
            'best_fit' : fits_from_every_iterations[np.argmin([iteration['negative_loglikelihood'] for iteration in fits_from_every_iterations])]}


def define_models_to_test(experiment, type_of_models_to_test):
    models_to_test = {}  # key : {} for key in ['model_type','parameters_to_fit', 'fixed_parameters']}

    for model_class in type_of_models_to_test:
        learning_parameters = set(LEARNING_PARAMETERS if experiment == 'exp3' else [])

        if model_class == 'linear':
            decision_parameters = set(LINEAR_DECISION_PARAMETERS)
            complete_list_of_parameters = learning_parameters.union(decision_parameters)
            varying_parameters = decision_parameters - {'resp_bias'}
            additional_param_to_fit = {'resp_bias'}

        # add the model varying both priors and likelihood without the interaction
            parameters_to_fit = additional_param_to_fit.union({'prior_weight', 'lik_weight'})
            fixed_parameters = complete_list_of_parameters - parameters_to_fit
            model_name = f"{model_class}_model_with_fitted__{('__').join(sorted(parameters_to_fit))}"
            model_type = f"{model_class}_model_of_decision"
            if experiment == 'exp3':
                model_type += '_and_learning'
            models_to_test[model_name] = {}
            models_to_test[model_name]['model_type'] = model_type
            models_to_test[model_name]['free_parameters_list'] = parameters_to_fit
            models_to_test[model_name]['fixed_parameters'] = fixed_parameters

        # add the model with 4 free parameters (weights of priors, likelihood, interaction and response bias)
            parameters_to_fit = decision_parameters
            fixed_parameters = complete_list_of_parameters - parameters_to_fit
            model_name = f"{model_class}_model_with_fitted__{('__').join(sorted(parameters_to_fit))}"
            model_type = f"{model_class}_model_of_decision"
            if experiment == 'exp3':
                model_type += '_and_learning'
            models_to_test[model_name] = {}
            models_to_test[model_name]['model_type'] = model_type
            models_to_test[model_name]['free_parameters_list'] = parameters_to_fit
            models_to_test[model_name]['fixed_parameters'] = fixed_parameters

        elif model_class == 'bayesian':
            decision_parameters = set(DECISION_PARAMETERS)
            complete_list_of_parameters = learning_parameters.union(decision_parameters)
            parameters_to_fit = decision_parameters
            fixed_parameters = complete_list_of_parameters - parameters_to_fit
            model_name = f"{model_class}_model_with_fitted__{('__').join(sorted(parameters_to_fit))}"
            model_type = f"{model_class}_model_of_decision"
            if experiment == 'exp3':
                model_type += '_and_learning'
            models_to_test[model_name] = {}
            models_to_test[model_name]['model_type'] = model_type
            models_to_test[model_name]['free_parameters_list'] = parameters_to_fit
            models_to_test[model_name]['fixed_parameters'] = fixed_parameters

        else:
            print(f'ERROR !!! : Unrecognized model type {model_class}')

    return models_to_test


def define_models_to_test_opt_priors(experiment, type_of_models_to_test):
    models_to_test = {} 

    for model_class in type_of_models_to_test:
        learning_parameters = set(LEARNING_PARAMETERS if experiment == 'exp3' else [])

        if model_class == 'linear':
            decision_parameters = set(LINEAR_DECISION_PARAMETERS)
            complete_list_of_parameters = decision_parameters.union(learning_parameters)
            varying_parameters = decision_parameters - {'resp_bias'}
            additional_param_to_fit = {'resp_bias'}

        # add the model varying both priors and likelihood without the interaction
            parameters_to_fit = additional_param_to_fit.union({'prior_weight', 'lik_weight'})
            fixed_parameters = complete_list_of_parameters - parameters_to_fit
            model_name = f"{model_class}_model_with_fitted__{('__').join(sorted(parameters_to_fit))}"
            model_type = f"{model_class}_model_of_decision"
            if experiment == 'exp3':
                model_type += '_and_learning'
            models_to_test[model_name] = {}
            models_to_test[model_name]['model_type'] = model_type
            models_to_test[model_name]['free_parameters_list'] = parameters_to_fit
            models_to_test[model_name]['fixed_parameters'] = fixed_parameters

        elif model_class == 'bayesian':
            decision_parameters = set(DECISION_PARAMETERS)
            complete_list_of_parameters = decision_parameters.union(learning_parameters)
            parameters_to_fit = decision_parameters
            fixed_parameters = complete_list_of_parameters - parameters_to_fit
            model_name = f"{model_class}_model_with_fitted__{('__').join(sorted(parameters_to_fit))}"
            model_type = f"{model_class}_model_of_decision"
            if experiment == 'exp3':
                model_type += '_and_learning'
            models_to_test[model_name] = {}
            models_to_test[model_name]['model_type'] = model_type
            models_to_test[model_name]['free_parameters_list'] = parameters_to_fit
            models_to_test[model_name]['fixed_parameters'] = fixed_parameters

            # add the optimal model if bayesian, there is no optimal linear model
            parameters_to_fit = set()
            fixed_parameters = complete_list_of_parameters
            model_name = f"{model_class}_model_with_optimal_parameters"
            model_type = f"{model_class}_model_of_decision"
            if experiment == 'exp3':
                model_type += '_and_learning'
            models_to_test[model_name] = {}
            models_to_test[model_name]['model_type'] = model_type
            models_to_test[model_name]['free_parameters_list'] = parameters_to_fit
            models_to_test[model_name]['fixed_parameters'] = fixed_parameters
        else:
            print(f'ERROR !!! : Unrecognized model type {model_class}')

    return models_to_test


