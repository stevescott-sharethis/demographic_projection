import numpy as np
import pandas as pd
import scipy.sparse as sparse


def normalize_logprob(unnormalized):
    rowmax = np.max(unnormalized, axis=1).reshape((-1, 1))
    probs = np.exp(unnormalized - rowmax)
    return probs / np.sum(probs, axis=1).reshape((-1, 1))


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

    Returns:
      The cookie-level demographic distributions as a numpy array.  Element (i,
      k) is the probability that cookie i belongs to class k.
    """
    unnormalized_probs = (
        np.log(prior)
        + visit_counts.dot(np.log(rates))
        - rates.sum(axis=0).reshape(1, -1)
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

    Returns:
      The array of rates as a numpy array.  Element (j, k) is the per-user
      Poisson rate for users of class k on site j.
    """
    counts = prior_a + visit_counts.transpose().dot(cookie_distributions)
    exposure = prior_b + cookie_distributions.sum(axis=0)
    return counts / exposure


def depanda(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.values
    return obj


def fit_model(visit_counts, rates, cookie_prior, prior_a, prior_b, niter,
              initialize=True):
    """
    Given raw data, priors, and starting values for cookie distributions and
    rates, run the EM algorithm for up to a specified number of iterations.

    Args:
      visit_counts: a scipy sparse matrix with rows corresponding to users and
        columns corresponding to sites.  Element (i, j) is the number of times
        user i visited site j.

      rates:  A numpy


    """
    print("Converting sparse matrices.")
    visit_counts_csr = visit_counts.tocsr()
    visit_counts_csc = visit_counts.tocsc()
    ncat = rates.shape[1]
    # num_cookies = visit_counts.shape[0]
    # num_sites = visit_counts.shape[1]
    history = np.empty((niter, ncat))

    rates = depanda(rates)
    cookie_prior = depanda(cookie_prior)
    prior_a = depanda(prior_a)
    prior_b = depanda(prior_b)

    if initialize:
        rates = Mstep(cookie_prior, visit_counts_csc, prior_a, prior_b)

    for i in range(niter):
        print(f"Iteration {i}")
        cookie_distributions = Estep(rates, visit_counts_csr, cookie_prior)
        history[i, :] = cookie_distributions.sum(axis=0)
        old_rates = rates.copy()
        rates = Mstep(cookie_distributions, visit_counts_csc, prior_a, prior_b)
        convergence = np.max(np.abs(rates - old_rates))
        if i > 10 and convergence < 1e-4:
            history = history[:i, :]
            break

    return rates, cookie_distributions, history
