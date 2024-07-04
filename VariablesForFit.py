#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:43:01 2020

@author: caroline
"""

SUBJECT_GROUP = 'Gorilla_V4'


len_of_task = 600

GORRILLA_EXP_STRUCTURE = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14]


DISTRIBUTIONS_RESOLUTION = 20

PLAUSIBLE_PARAMETER_VALUES = {
    'volatility': [k/len_of_task for k in [0.001, 0.1, 5, 15, 30, 70, 150]],
    'strength_evidence': [0.2, .5, .7, 1, 1.4, 2, 5],
    'bias_evidence': [-0.9, -0.7, -0.3, 0, 0.3, 0.7, 0.9],
    'resp_bias': [-0.9, -0.7, -0.3, 0, 0.3, 0.7, 0.9],
    'prior_weight': [0.2, .5, .7, 1, 1.4, 2, 5],
    'lik_weight': [0.2, .5, .7, 1, 1.4, 2, 5],
    'interaction_weight': [0.2, .5, .7, 1, 1.4, 2, 5],
    'linear_prior_weight': [0.0001, 0.2, .5, .7, 1, 1.4, 2],
    'linear_lik_weight': [0.0001, 0.2, .5, .7, 1, 1.4, 2]}


LEARNING_PARAMETERS = ['volatility','strength_evidence','bias_evidence']
DECISION_PARAMETERS = ['resp_bias','prior_weight','lik_weight']
LINEAR_DECISION_PARAMETERS = DECISION_PARAMETERS+['interaction_weight']


OPTIMAL_LEARNING_PARAMETERS = {'volatility': 15/600,
                               'strength_evidence': 1,
                               'bias_evidence': 0}

OPTIMAL_DECISION_PARAMETERS = {'resp_bias' : 0, 
                               'prior_weight': 1,
                               'lik_weight': 1}



DEFAULT_LINEAR_DECISION_PARAMETERS = {'resp_bias' : 0, 
                                      'prior_weight': 0,
                                      'lik_weight': 0,
                                      'interaction_weight': 0}
DEFAULT_PARAMETERS_VALUES_BY_MODEL = {
    'bayesian_model_of_decision_and_learning': {**OPTIMAL_DECISION_PARAMETERS, **OPTIMAL_LEARNING_PARAMETERS},
    'bayesian_model_of_decision': OPTIMAL_DECISION_PARAMETERS,
    'linear_model_of_decision_and_learning': {**DEFAULT_LINEAR_DECISION_PARAMETERS, **OPTIMAL_LEARNING_PARAMETERS},
    'linear_model_of_decision': DEFAULT_LINEAR_DECISION_PARAMETERS
    }

RECOVERY_OPTIONS = {'method': 'Nelder-Mead',
                    'restart_n_times': 50,
                    'remap_type': 'min_remap',
                    'distribution_type_for_initial_guess': 'uniform',
                    'generative_parameters_distributions': 'uniform',
                    'options': None}  # {'options' : {'maxiter':100}



