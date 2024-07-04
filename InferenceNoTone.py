#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ideal observer code for:
    A sequence of noisy images (faces or house). The probability of the
    stimulus category (\theta_k, proprotion of house that prevails at trial k)
    can be volatile or fixed.
    The observer estimate \theta based on the observation received.

    This estimation is done by casting the problem as a hidden markov chain,
    and using the forward algorithm (in which the posterior distribution is
    called Alpha by convention; here, Alpha is the \theta that we want to
    estimate)

@author: Florent Meyniel & Caroline Bévalot
"""

import numpy as np
from scipy.stats import beta


def compute_alpha_and_beta(mean_beta_dist, var_beta_dist):
    """
    Function which computes alpha and beta, the parameters
    of a beta distribution, from the mean and variance of this distribution.
    Alpha corresponds to beta_H (or 'beta_HighTone_H')
    and beta to beta_F (or 'beta_LowTone_H'), which are the pseudo-count+1
    of houses and faces.
    """
    alpha = round(((1 - mean_beta_dist) / var_beta_dist - 1 /
                   mean_beta_dist) * mean_beta_dist**2)
    beta = round(alpha * (1 / mean_beta_dist - 1))
    return alpha, beta


def init_Alpha(options, pgrid):
    """
    Initialize Alpha, which is the probability distribution of \theta, the
    bernoulli parameter corresponding to the probability of house at a given
    trial, using the discretization grid pgrid.
    This initialization takes into account potential biases, parameterized
    as the parameter (mean and variance) of the beta distribution for \theta:
        options['mean_beta_dist']
        options['var_beta_dist']
    By default, Alpha is flat.
    """
    if all([param in options.keys() for param in ['mean_beta_dist', 'var_beta_dist']]):
        beta_H, beta_F = compute_alpha_and_beta(options['mean_beta_dist'],
                                                options['var_beta_dist'])
    else:
        beta_H, beta_F = 1, 1
    beta_dist = beta.pdf(pgrid, beta_H, beta_F)

    # return the normalized, discretized probability distribution
    return beta_dist / np.sum(beta_dist)


def compute_transition_matrix(Alpha0):
    """
    Compute the transition matrix, in which columns correspond to \theta_k and
    rows correspond \theta{k-1}
    The diagonal is set to 0 (because the matrix corresponds to the occurence
    of a change point) and the value over columns is one.
    """
    # Repeat the prior
    T = np.vstack([Alpha0 for k in range(Alpha0.shape[0])])

    # Set diagonal to 0 and normalize per row
    for k in range(Alpha0.shape[0]):
        T[k, k] = 0
        T[k, :] = T[k, :] / np.sum(T[k, :])

    return T


def change_marginalize(Alpha, T):
    """
    Compute the integral:
        \int p(\theta_{t-1}|y_{1:t-1})p(\theta_t|\theta_{t-1})) d\theta_{t-1}
        in which the transition matrix has zeros on the diagonal, and
        the prior Alpha0 elsewhere. In other words, it computes the updated
        distribution in the case of a change point (but does not multiply by
        the prior probability of change point).
    """
    return np.matmul(Alpha, T)


def alteration_of_evidence(seq_image_lik, options):
    """
    Modify the likelihood values (formally defined as p(I_k | c_k=H)).
    The transformation is affine in log-odd:
        - options['bias_evidence'] bias the likelihood function in favor of
          houses (when >0) or faces (when <0).
        - options['strength_evidence'] exacerbate the likelihood function when >1,
          and blunt it when <1.
    If options['bias_evidence'] or options['strength_evidence'] is not provided,
    no transformation is applied.
    """
    seq_image_lik = np.array(seq_image_lik)
    if (options['strength_evidence'] == 1) & (options['bias_evidence'] == 0):
        dist_seq_image_lik = seq_image_lik
    else:
        # affine transformation in log-odd space.
        dist_seq_image_lik_lo = \
            options['bias_evidence'] + options['strength_evidence'] * lo(seq_image_lik)
        # remap to the probability space.
        dist_seq_image_lik = 1/(1 + np.exp(-dist_seq_image_lik_lo))
    return dist_seq_image_lik


def compute_mean_of_dist(dist, pgrid):
    """ Compute mean of probability distribution"""
    return dist.transpose().dot(pgrid)


def compute_sd_of_dist(dist, pgrid):
    """ Compute SD of probability distribution"""
    m = compute_mean_of_dist(dist, pgrid)
    v = dist.transpose().dot(pgrid**2) - m**2
    return np.sqrt(v)


def turn_posterior_into_prediction(Alpha, p_c, T):
    """
    Turn the posterior at the current trial (included) into a prediction
    about the next trial, taking into account the possibility of a change point.
    """
    # Initialize containers
    pred_Alpha = np.ndarray(Alpha.shape)

    # Update
    for t in range(Alpha.shape[1]):
        # Update Alpha, without a new observation but taking into account
        # the possibility of a change point
        pred_Alpha[:, t] = (1-p_c) * Alpha[:, t] + \
                           p_c * change_marginalize(Alpha[:, t], T)

    return pred_Alpha


def lo(probability):
    """ Covert a probability into its log-odd."""
    probability = np.clip(probability, 0.001, 0.999)
    return np.log(probability/(1-probability))


def forward_updating(dist_seq_image_lik, options, init=None):
    """
    Update iteratively the joint probability of observations and parameters
    values, moving forward in the sequence of likelihoods of images being houses.

    Return Alpha (the running posterior estimate), pgrid (the probability
    grid used for numerical estimation), and the options used for the inference.

    init is an optional dictionary with precomputed values, if not provided, then
    the following quantities are initialized in this function:
        - pgrid
        - alpha0, the prior for theta
        - TransMat, the transition matrix in case of change point
    """
    # Get change point probability
    p_c = options['volatility']
    # Get probability grid
    pgrid = np.linspace(0, 1, options['resol']) if init is None else init['pgrid']
    # Initialize Alpha distribution
    Alpha0 = init_Alpha(options, pgrid) if init is None else init['Alpha0']
    # Initialize containers
    Alpha = np.zeros((len(Alpha0), len(dist_seq_image_lik)))
    # Compute transition matrix
    TransMat = compute_transition_matrix(Alpha0) if init is None else init['TransMat']

    # Update iteratively
    for t in range(len(dist_seq_image_lik)):
        # Compute the observation likelihood: the probability of the stimulus
        # being perceived as a house
        lik = pgrid*dist_seq_image_lik[t] + (1-pgrid)*(1-dist_seq_image_lik[t])
        if t == 0:
            # Update Alpha0 with the new observation
            Alpha[:, t] = (1-p_c) * lik * Alpha0 + \
                          p_c * lik * change_marginalize(Alpha0, TransMat)
        else:
            # Update Alpha with the new observation
            Alpha[:, t] = (1-p_c) * lik * Alpha[:, t-1] + \
                          p_c * lik * change_marginalize(Alpha[:, t-1], TransMat)
        # Normalize
        cst = np.sum(Alpha[:, t])
        Alpha[:, t] = Alpha[:, t]/cst

    return {'theta': Alpha, 'pgrid': pgrid,
            'options': options, 'T': TransMat, 'Alpha0': Alpha0}


def compute_inference(seq_image_lik, options, init=None):
    """
    Wrapper function to compute the inference of the generative proportion of houses.

    INPUT:
        - seq_image_lik: the (possibly) distorted likelihood of images being houses
        - options: a dictionary
          options['volatility']: assumed volatility
          options['prior_weight']: bias the weight of the prior in the posterior, which
            can be attenuated (<1) or exacerbated (>1) compared to optimal inference
          options['lik_weight']: bias the weight of the likelihood in the posterior, which
            can be attenuated (<1) or exacerbated (>1) compared to optimal inference
          options['resp_bias']: bias the posterior toward houses (>0) or faces (<0)
          options['bias_evidence']: bias the likelihood function in favor of
            houses (when >0) or faces (when <0)
          options['strength_evidence']: exacerbate the likelihood function when >1,
            and blunt it when <1
        - init: a dictionary
          init['pgrid']: the discretization grid for probability
          init['alpha0']: the prior about the proportion of houses
          init['TransMat']: the transition matrix for the proportion of houses,
             in case of change point

    OUTPUTS -- a dictionary with the following elements:
        - 'theta': the posterior estimate of the generative proprotion of houses.
            theta[k] is given previous observations, obs[k] included.
        - 'p(c_k=H|past)': the prediction about the category house on trial k, given
            previous trials (taking into account volatility from past to present).
        - 'sd_p_past': the SD of this prediction
        - pgrid: the discretized probability grid used for numerical estimation
        - 'p(c_k=H|image_k, past)': the probability of the category house on the current trial k,
            given past trials and the current image
        - 'p(c_k=H|obs(1:k))': the probability of the category being house on the current trial k,
            given past trials and the current sensory data. It is the same as
            'p(c_k=H|image_k, past)' because only images are presented to subjects
        """
    # learning stage
    dist_seq_image_lik = alteration_of_evidence(seq_image_lik, options)
    res = forward_updating(dist_seq_image_lik, options, init=init)
    pred_Alpha = turn_posterior_into_prediction(res['theta'], options['volatility'], res['T'])
    pred_house = np.hstack((compute_mean_of_dist(res['Alpha0'], res['pgrid']),
                            compute_mean_of_dist(pred_Alpha, res['pgrid'])[:-1]))
    sd_pred_house = np.hstack((compute_sd_of_dist(res['Alpha0'], res['pgrid']),
                               compute_sd_of_dist(pred_Alpha, res['pgrid'])[:-1]))


    return {'theta': res['theta'],
            'p(c_k=H|obs(1:k-1))': pred_house,
            'sd_p(c_k=H|obs(1:k-1))': sd_pred_house,
            'pgrid': res['pgrid']}


def odd_distorsion(p, beta):
    p = np.clip(p, 0.001, 0.999)
    odd = p / (1-p)
    return (odd ** beta) / (1 + odd ** beta)


def compute_choice_probability(seq_image_lik, options, th_trials, model_type='bayesian'):
    """
    Wrapper function that computes:
        - the (possibility) distorted subject's representation of the image likelihood
        - p(c_k=H|past) is the prior probability to on trial k, the category will be H, given
          previous trials
        - p(c_k=H|image_k, past) is the probability of the category being H on the current trial k,
          given past trials and the current image
        - p(c_k=H|obs(1:k)) is the probability of the category being H on the current trial k,
          given past trials and the current sensory data (here, sensory data are restricted to the
          image, so this quantity is the same as the former. In other experiments, there is also
          a tone).
        """
    if 'bayesian' in model_type:
        posterior_logodd_ratio = options['resp_bias'] \
            + options['prior_weight'] * lo(th_trials) \
            + options['lik_weight'] * lo(seq_image_lik)
        post_house = 1/(1 + np.exp(-posterior_logodd_ratio))

    elif model_type in ['linear', 'linear_with_interaction']:
        posterior = options['prior_weight'] * th_trials \
                    + options['lik_weight'] * np.array(seq_image_lik)\
                    + options['interaction_weight'] * th_trials * np.array(seq_image_lik)
        post_house = 1/(1 + np.exp(-(posterior + options['resp_bias'])))

    elif model_type == 'linear_without_interaction':
        posterior = options['prior_weight'] * th_trials \
                    + options['lik_weight'] * np.array(seq_image_lik)
        post_house = 1/(1 + np.exp(-(posterior + options['resp_bias'])))

    else:
        print('model not available')

    return {#'p(c_k=H|past)': th_trials,
            'p(c_k=H|image_k, past)': post_house,
            'p(c_k=H|obs(1:k))': post_house}


def sample_choice_from_posterior(p_house_post):
    return [1 if np.random.rand() < p else 0 for p in p_house_post]


