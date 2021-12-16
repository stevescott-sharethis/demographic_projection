import numpy as np
import pandas as pd
import scipy.sparse as sparse
import os

import model

import matplotlib.pyplot as plt
import pdb


def reverse_map(dict_of_ints):
    """
    Construct the reverse mapping of 'dict_of_ints'.  In other words, return a
    list 'ans' such that ans[i] = k if and only if dict_of_ints[k] = i.
    """
    ans = [None] * len(dict_of_ints)
    for k, v in dict_of_ints.items():
        ans[v] = k
    return ans


def read_visits(shards=range(3)):
    """
    Args:
      shards:  A subset of the integers 0-9, indicating which data shards
        should be read in.

    Returns:
      A pandas DataFrame
    """
    data = pd.DataFrame()
    dname = "/Users/stevescott/Downloads/demographics-2"
    for which_part in shards:
        fname = f"part-00000000000{which_part}.csv.gz"
        fname = os.path.join(dname, fname)
        print(f"reading {fname}")
        shard = pd.read_csv(fname)
        data = data.append(shard)
    return data


def remove_bots(data, bot_visit_threshold):
    """
    Any cookie that visits any site more than bot_visit_threshold times is
    assumed to to be a bot and will be deleted.

    Args:

      data: A data frame to be screened.  The data has the following variables.

                               estid             domain  visits
            ZGwAB2EdKZsAAAAIF0RUAw==       vmodtech.com  155195
            ZHyAAmFejEIAAAAIF7S9Aw==         legacy.com   17569
            ZGYAAWGA+9kAAAAIDYEjAw==   jump.mingpao.com    7300

      bot_visit_threshold: Any cookie with a 'visits' field greater than this
        amount is assumed to be a bot.

    Returns:
      The 'data' data frame is returned, after any cookies thought to be bots
      are removed.
    """
    print("removing bots")
    too_many = data["visits"] > bot_visit_threshold
    bots = set(data.loc[too_many, "estid"])
    bot_flags = data["estid"].isin(bots)
    data = data[~bot_flags]
    return data


def setup_visit_data(shards):
    """
    Reads and cleans the visit count data.  Populates the data structures, and
    returns the visit-related bits needed to run the model.
    """
    data = read_visits(shards)
    data = remove_bots(data, bot_visit_threshold=100)

    print("creating cookie map")
    cookie_set = set(data["estid"].values)
    num_cookies = len(cookie_set)
    cookie_map = {cookie: idx for idx, cookie in enumerate(cookie_set)}
    del(cookie_set)

    print("creating site map")
    site_set = set(data["domain"].values)
    num_sites = len(site_set)
    site_map = {domain: idx for idx, domain in enumerate(site_set)}
    del(site_set)

    visit_counts = sparse.lil_matrix((num_cookies, num_sites))

    def emplace_data(row):
        visit_counts[cookie_map[row["estid"]],
                     site_map[row["domain"]]] = row["visits"]
        return None

    print("Loading visits data into a sparse matrix")
    data.apply(emplace_data, axis=1)

    print(f"There are {num_sites} sites and {num_cookies} cookies.")

    return visit_counts, cookie_map, site_map, data


def read_site_profiles():
    """

    """
    dname = "/Users/stevescott/Downloads/demographics"
    fname = os.path.join(dname, "age_gender.csv")
    data = pd.read_csv(fname, index_col="domain")
    age_profiles = data.iloc[:, range(6)]
    gender_profiles = data.iloc[:, range(6, 8)]
    return age_profiles, gender_profiles


def create_priors(profile, overall_site_counts, known_sites, num_cookies):
    """
    Args:
      profile: A data frame indexed by domain.  Each row of the data frame is a
        discrete probability distribution over the set of demographic buckets
        represented by the data frame columns.
      overall_site_counts: A numpy array (vector) containing the overall number
        of visits to each site.
      known_sites: A numpy array (vector) of boolean values indicating whether
        the values in the site profile are to be treated as known.
    """
    profile = profile.loc[overall_site_counts.index, :]
    num_sites = profile.shape[0]
    if len(overall_site_counts) != num_sites:
        raise Exception("The vector of overall_site_counts must match "
                        "the number of rows in 'profile'.")
    if len(known_sites) != num_sites:
        raise Exception("The vector of known_sites must match "
                        "the number of rows in 'profile'.")

    scaled_profile = profile * overall_site_counts.values.reshape((-1, 1))

    cookie_prior = scaled_profile.sum(axis=0)
    cookie_prior = cookie_prior / np.sum(cookie_prior)

    prior_a = scaled_profile / num_cookies
    prior_b = pd.DataFrame(np.ones_like(prior_a), index=prior_a.index,
                           columns=prior_a.columns)
    prior_a.iloc[known_sites, :] *= 1e+100
    prior_b.iloc[known_sites, :] *= 1e+100
    prior_a.iloc[~known_sites, :] = (
        cookie_prior.values.reshape((1, -1)) * overall_site_counts[
            ~known_sites].values.reshape((-1, 1)) / num_cookies
    )

    cookie_prior = np.concatenate(
        [cookie_prior.values.reshape((1, -1))] * num_cookies)

    return cookie_prior, prior_a, prior_b


# For dev purposes start with shards = [0].
shards = [0]

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


rates = prior_a / prior_b

rates, cookie_distribution, history = model.fit_model(
    visit_counts, rates, cookie_prior, prior_a, prior_b, niter=200)

# ===========================================================================
# This bit does an analysis of the fitted model by comparing the fitted rates
# to the masked rates
masked_sites = site_index[1000:]


def plot_site(domain_name, site_map, fitted_rates, age_profiles):
    pos = site_map[domain_name]
    site_rates = fitted_rates[pos, :]
    site_profile = site_rates / np.sum(site_rates)
    true_profile = age_profiles.loc[domain_name, :]
    profiles = pd.DataFrame({
        "true": true_profile,
        "estimated": site_profile,
        })
    profiles.plot(kind="barh")
    plt.show()


def plot_user(data, user):
    """

    """
    subset = data[data.estid == user]



def plot_users(data, users):
    """
    """
