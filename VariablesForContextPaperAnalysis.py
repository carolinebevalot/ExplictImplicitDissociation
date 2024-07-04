#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 19:00:56 2022

@author: caroline
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

EXPERIMENTS = ['exp1', 'exp3']
COMPUTER = 'laptop'
SUBJECT_GROUP = 'Gorilla_V4'
DISTRIBUTIONS_RESOLUTION = 20


SIMULATION_OPTIONS_BOTH_EXP \
  = {'exp1': {'bayesian_model_with_generative_priors':
                {'model_type': 'bayesian_model_of_decision',
                  'combination_type': 'bayesian',
                  'prior_type': 'generative_priors',
                  'prior_column_name': 'generative_priors',
                  'posterior_column_name': 'posteriors_with_generative_priors',
                  'free_parameters': 'None',
                  'logreg_options_label': 'genT1_genT2_priors'},
            'linear_model_without_interaction_with_generative_priors':
                {'model_type': 'linear_model_of_decision',
                  'combination_type': 'linear_without_interaction',
                  'prior_type': 'generative_priors',
                  'prior_column_name': 'generative_priors',
                  'posterior_column_name': 'linear_posteriors_with_generative_priors',
                  'free_parameters': 'None',
                  'logreg_options_label': 'lingenT1_lingenT2_priors'}
                },
        'exp3': {'bayesian_model_with_optimal_priors':
                {'model_type': 'bayesian_model_of_decision_and_learning',
                  'combination_type': 'bayesian',
                  'prior_type': 'inferred_priors',
                  'prior_column_name': 'inferred_priors_with_optimal_parameters',
                  'posterior_column_name': 'posteriors_with_optimal_parameters',
                  'confidence_column_name': 'confidence_in_priors_with_optimal_parameters',
                  'free_parameters': 'optimal_parameters',
                  'logreg_options_label': 'genT1_optT2_priors'},
                'linear_model_without_interaction_with_optimal_priors':
                    {'model_type': 'linear_model_of_decision_and_learning',
                      'combination_type': 'linear_without_interaction',
                      'prior_type': 'inferred_priors_with_linear_param',
                      'prior_column_name': 'inferred_priors_with_optimal_parameters',
                      'posterior_column_name': 'posteriors_with_optimal_parameters',
                      'confidence_column_name': 'confidence_in_priors_with_optimal_parameters',
                      'free_parameters': 'optimal_parameters',
                      'logreg_options_label': 'lingenT1_linoptT2_priors'},
                    
                'ideal_observer_with_fitted_decision_and_learning_parameters':
                  {'model_type': 'bayesian_model_of_decision_and_learning',
                    'combination_type': 'bayesian',
                    'prior_type': 'inferred_priors',
                    'prior_column_name': 'inferred_priors_with_fitted_decision_and_learning_parameters',
                    'posterior_column_name': 'posteriors_with_fitted_decision_and_learning_parameters',
                    'confidence_column_name': 'confidence_in_priors_with_fitted_decision_and_learning_parameters',
                    'free_parameters': 'fitted_decision_and_learning_parameters',
                    'logreg_options_label': 'genT1_fitallT2_priors'}

                }
        }

                
MODEL_OPTIONS_BOTH_EXP \
            = {exp: {'ideal_observer_with_generative_priors':
                      {'model_type': 'bayesian_model_of_decision',
                      'prior_type': 'generative_priors',
                                          'combination_type': 'bayesian',

                      'prior_column_name': 'generative_priors',
                      'posterior_column_name': 'posteriors_with_generative_priors',
                      'free_parameters': 'None',
                      'logreg_options_label': 'genT1_genT2_priors'},
                      'linear_ideal_observer_with_generative_priors':
                      {'model_type': 'linear_model_of_decision',
                        'prior_type': 'generative_priors',
                        'prior_column_name': 'generative_priors',
                        'posterior_column_name': 'linear_posteriors_with_generative_priors',
                        'free_parameters': 'None',
                        'logreg_options_label': 'lingenT1_lingenT2_priors'}
                      }
                for exp in EXPERIMENTS}
MODEL_OPTIONS_BOTH_EXP['exp3'] \
            = {**MODEL_OPTIONS_BOTH_EXP['exp3'],
                **{'ideal_observer_with_optimal_parameters':
                  {'model_type': 'bayesian_model_of_decision_and_learning',
                    'prior_type': 'inferred_priors',
                    'prior_column_name': 'inferred_priors_with_optimal_parameters',
                    'posterior_column_name': 'posteriors_with_optimal_parameters',
                    'confidence_column_name': 'confidence_in_priors_with_optimal_parameters',
                    'free_parameters': 'optimal_parameters',
                    'logreg_options_label': 'genT1_optT2_priors'},

                  'ideal_observer_with_fitted_decision_and_learning_parameters':
                  {'model_type': 'bayesian_model_of_decision_and_learning',
                    'prior_type': 'inferred_priors',
                    'combination_type': 'bayesian',
                    'prior_column_name': 'inferred_priors_with_fitted_decision_and_learning_parameters',
                    'posterior_column_name': 'posteriors_with_fitted_decision_and_learning_parameters',
                    'posterior_column_name_opt_decision': 'posteriors_with_opt_decision_parameters_and_learning_parameters',
                    'confidence_column_name': 'confidence_in_priors_with_fitted_decision_and_learning_parameters',
                    'free_parameters': 'fitted_decision_and_learning_parameters',
                    'logreg_options_label': 'genT1_fitallT2_priors'},

                  'linear_ideal_observer_with_optimal_parameters':
                  {'model_type': 'linear_model_of_decision_and_learning',
                    'prior_type': 'inferred_priors_with_linear_param',
                    'prior_column_name': 'inferred_priors_with_optimal_parameters',
                    'posterior_column_name': 'posteriors_with_optimal_parameters',
                    'confidence_column_name': 'confidence_in_priors_with_optimal_parameters',
                    'free_parameters': 'optimal_parameters',
                    'logreg_options_label': 'lingenT1_linoptT2_priors'},
                  }}


MODEL_CROSS_VALIDATION_COLORS_COMPARISONS = np.array([(97/255,22/255,23/255),
                                                      (165/255,39/255,35/255),
                                                      (191/255,69/255,38/255),
                                                      (217/255,151/255,0/255), 
                                                      (241/255,182/255,79/255),
                                                      (217/255,151/255,0/255), 
                                                      (241/255,182/255,79/255),
                                                      (97/255,22/255,23/255),
                                                      (165/255,39/255,35/255),
                                                      (191/255,69/255,38/255),
                                                      (217/255,151/255,0/255), 
                                                      (241/255,182/255,79/255)])
APRICOT = (251/255, 145/255, 77/255)
LIME = (131/255, 253/255, 56/255)
BLUEGREY = (146/255, 181/255, 204/255)
MODEL_CROSS_VALIDATION_COLORS_CMAP = LinearSegmentedColormap.from_list("", [LIME, APRICOT])
MODEL_CROSS_VALIDATION_COLORS_CMAP_INV = LinearSegmentedColormap.from_list("", [APRICOT, LIME])


MODEL_CROSS_VALIDATION_COLORS_COMPARISONS2 = np.array([(97/255, 22/255, 23/255),
                                                       (165/255, 39/255, 35/255),
                                                       (191/255, 69/255, 38/255),
                                                       (217/255, 151/255, 0/255),
                                                       (241/255, 182/255, 79/255),
                                                       (242/255, 215/255, 185/255),
                                                       (139/255, 84/255, 85/255),
                                                       (204/255, 133/255, 130/255)])

MODEL_CROSS_VALIDATION_COLORS_CORRELATIONS = [(181/255,181/255,180/255,255/255),
                                              (132/255,132/255,146/255,255/255),
                                              (92/255,97/255,118/255,255/255),
                                              (62/255,70/255,93/255,255/255),
                                              (38/255,45/255,67/255,255/255)]


LOGODDS_COLORS = [(157/255,222/255,225/255), (89/255,190/255,216/255), 
                  (60/255,142/255,205/255), (39/255,103/255,153/255), (13/255,70/255,113/255)]

PRIOR_UPDATE_COLORS = {'prior_before_change': (254/255,133/255,50/255),
                     'prior_after_change': (255/255,208/255,90/255),
                     'likelihood': (89/255,190/255,216/255)}


COLORS_FOR_TASK_COMPARISON = {
    'likelihood_weight':[(157/255,222/255,225/255), (89/255,190/255,216/255)],
    'before_Q': [(39/255,103/255,153/255), (13/255,70/255,113/255)],
    'after_Q': [(254/255,133/255,50/255), (249/255,87/255,76/255), (249/255,87/255,76/255), (254/255,133/255,50/255)],
    'priors_weight': [(254/255,133/255,50/255), (249/255,87/255,76/255)],
    'posteriors_weight': [(255/255, 230/255 ,177/255), (255/255,208/255,90/255)]}

PRETTY_NAMES = {'priors_weight': 'Weights of priors',
                'likelihood_weight': 'Weights of likelihood',
                'posteriors_weight': 'Weights of posteriors',
                'generative_priors': 'generative priors',
                'inferred_priors_with_optimal_parameters': 'optimal priors',
                'inferred_priors_with_fitted_decision_and_learning_parameters': 'fitted priors',
                'inferred_priors_with_fitted_decision_parameters_and_volatility': 'fitted priors with best model',
                'inferred_priors_with_linear_fitted_decision_and_learning_parameters': 'fitted priors with linear model',
                'before_Q': 'Before prior report',
                'after_Q': 'A.'}
THETA = {'exp1': [0.1, 0.3, 0.5, 0.7, 0.9],
         'exp3': [0.9, 0.3, 0.5, 0.1, 0.7,
                  0.3, 0.9, 0.1, 0.7, 0.5, 0.3,
                  0.7, 0.9, 0.5, 0.1]}

PENALIZATION_SUFFIX = 'pen'
NO_PENALIZATION_SUFFIX = 'nopen'

