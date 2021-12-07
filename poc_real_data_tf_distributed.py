import numpy as np
import scipy.sparse as sparse
import scipy.stats
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import os

from poc_real_data import read_site_profiles, setup_visit_data, reverse_map, create_priors


def normalize_logprob(unnormalized):
    rowmax = tf.reshape(tf.reduce_max(unnormalized, axis=1), (-1, 1))
    probs = tf.exp(unnormalized - rowmax)
    return probs / tf.reshape(tf.reduce_sum(probs, axis=1), (-1, 1))


def Estep(rates: np.ndarray,
          visit_counts: sparse.csr_matrix,
          prior: np.ndarray):
    """
    Args:
      rates: Element (j,k) is the poisson rate for site j for users in category
        k.
      visit_counts: A scipy.sparse matrix.  Element (i, j) is the number of
        visits by user i to site j.
      prior: Element (i, k) is the prior probability that user i belongs to
        category k.
    """
    unnormalized_probs = (
        tf.math.log(prior)
        + tf.sparse.sparse_dense_matmul(visit_counts, tf.cast(tf.math.log(rates), tf.double))
        - tf.reshape(tf.reduce_sum(rates, axis=0), (1, -1))
    )
    return normalize_logprob(unnormalized_probs)



def Mstep(cookie_distributions: np.ndarray,
          visit_counts: sparse.csc_matrix,
          prior_a: np.ndarray,
          prior_b: np.ndarray):
    """
    Args:
      cookie_distributions: Element (i, k) is the probability that cookie i
        belongs to cateogory k.
      visit_counts: A sparse matrix.  Element (i, j) is the number of visits by
        user i to site j.
      prior_a: Element (j, k) is the prior pseudo-count of visits to site j by
        users from category k.
      prior_b: Element (j, k) is the prior pseudo-count of potential users from
        category k.
    """
    # counts = prior_a + tf.tensordot(tf.transpose(visit_counts) ,cookie_distributions, axes=(1,0))
    # counts = prior_a + tf.sparse.sparse_dense_matmul(tf.transpose(visit_counts) ,cookie_distributions)
    counts = prior_a + tf.transpose(tf.sparse.sparse_dense_matmul(tf.transpose(cookie_distributions), visit_counts))
    exposure = prior_b + tf.reduce_sum(cookie_distributions, axis=0)
    return counts / exposure

def fit_model(visit_counts, rates, cookie_prior,
              prior_a, prior_b, niter=50):

    visit_counts_csr = visit_counts
    visit_counts_csc = visit_counts
    ncat = rates.shape[1]
    # num_cookies = visit_counts.shape[0]
    # num_sites = visit_counts.shape[1]
    history = np.empty((niter, ncat))
    for i in range(niter):
        print(f"Iteration {i}")
        cookie_distributions = Estep(rates, visit_counts_csr, cookie_prior)
        history[i, :] = tf.reduce_sum(cookie_distributions, axis=0)
        old_rates = tf.identity(rates)
        rates = Mstep(cookie_distributions, visit_counts_csc, prior_a, prior_b)

        convergence = np.max(np.abs(rates - old_rates))
        print(np.all(old_rates == rates), convergence)
        if convergence < 1e-4:
            history = history[:i, :]
            break

    return rates, cookie_distributions, history
    
def run_em_distributed(dev_mode:bool = False):
  
    if dev_mode:
        # For dev purposes start with shards = [0].
        shards = [0]
    else:
        shards = range(10)

    visit_counts, cookie_map, site_map, data = setup_visit_data(shards)
    site_index = reverse_map(site_map)
    num_cookies, num_sites = visit_counts.shape

    visits_per_site = pd.Series(np.array(visit_counts.sum(axis=0)).ravel(),
                                index=site_index)

    age_profiles, gender_profiles = read_site_profiles()
    known_sites = np.array([False] * num_sites)
    known_sites[:1000] = True
    cookie_prior, prior_a, prior_b = create_priors(
        # gender_profiles,
        age_profiles,
        visits_per_site,
        known_sites=known_sites,
        num_cookies=num_cookies)
    rates = prior_a/prior_b

    dense_shape = (num_cookies, num_sites)
    coo = visit_counts.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    values = coo.data
    site_counts = tf.sparse.SparseTensor(indices, values, dense_shape)

    fitted_rates, fitted_distributions, history = fit_model(site_counts, 
                              tf.convert_to_tensor(rates), 
                              cookie_prior, 
                              tf.convert_to_tensor(prior_a), 
                              tf.convert_to_tensor(prior_b))

if __name__ == "__main__":
    run_em_distributed()