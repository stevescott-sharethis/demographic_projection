import numpy as np
import scipy.sparse as sparse
import scipy.stats
import pandas as pd

from model import fit_model

import pdb
import matplotlib.pyplot as plt


def run_simulation(site_rates: np.ndarray,
                   num_cookies: int,
                   true_demographics: pd.Series,
                   num_known_sites: int,
                   niter: int,
                   cookie_prior=None):
    """
    Args:
      site_rates: A number-of-sites by number-of-categories array.  Element j,k
        is the the expected number of visits to site j by a single person of
        category k.
      num_cookies: The number of users to simulate.
      true_demographics: A discrete probability distribution.  The series index
        describes the categories.  The series values are the proportion of the
        population in those categories.
      num_known_sites: The number of sites whose rates are to be treated as
        known.
      niter:  The max number of iterations in the EM algorithm.

    Returns:
      TBD
    """
    ncat = len(true_demographics)
    if cookie_prior is None:
        cookie_prior = np.concatenate(
            [true_demographics.values.reshape((1, -1))] * num_cookies)

    true_cookie_categories = np.random.choice(
        range(ncat), size=num_cookies, p=true_demographics, replace=True)

    print("populating site_counts")
    site_counts = sparse.lil_matrix((num_cookies, num_sites))
    for i in range(num_cookies):
        true_category = true_cookie_categories[i]
        site_counts[i, :] = scipy.stats.poisson.rvs(
            site_rates[:, true_category])

    cookie_distributions = cookie_prior.copy()

    print("populating site priors")
    prior_a = np.concatenate(
        [true_demographics.values.reshape((1, -1))] * num_sites)
    prior_b = np.ones_like(prior_a)
    if num_known_sites > 0:
        idx = range(num_known_sites)
        prior_a[idx, :] = 1e+100 * site_rates[idx, :]
        prior_b[idx, :] *= 1e+100

    starting_rates = np.random.rand(num_sites, ncat)

    rates, cookie_distributions, history = fit_model(
        site_counts, starting_rates, cookie_prior,
        prior_a, prior_b, niter=niter)

    return rates, cookie_distributions, history, true_cookie_categories


num_cookies = 50000
num_sites = 10000
true_demographics = pd.Series((.4, .3, .2, .1),
                              index=["Red", "Blue", "Green", "Yellow"])
ncat = len(true_demographics)
print("simulating rates")
rates = scipy.stats.gamma.rvs(a=true_demographics * 100,
                              scale=1.0 / (np.ones(ncat) * num_sites),
                              size=(num_sites, ncat))

cookie_prior = np.sqrt(true_demographics)
cookie_prior = cookie_prior / cookie_prior.sum()
cookie_prior = np.concatenate([np.array([cookie_prior.values])] * num_cookies)

fitted_rates, fitted_distributions, history, true_cookie_categories = (
    run_simulation(
        rates, num_cookies, true_demographics, num_known_sites=50,
        niter=200, cookie_prior=cookie_prior)
)

fig, ax = plt.subplots(1, 1)
for i in range(history.shape[1]):
    ax.plot(history[:, i],
            label=true_demographics.index[i],
            color=true_demographics.index[i].lower())
ax.legend()
ax.set_xlabel("EM Iteration")
ax.set_ylabel("Number of Cookies")
ax.set_title("Estimated number of cookies by type")
ax.hlines(true_demographics * num_cookies, linestyles='dotted',
          xmin=0, xmax=history.shape[0])
fig.show()
