import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lib.ts import acorr

# From lecture derivation of the full conditional probabilities
def update_mu(n, ybar, sig_sq, mu_0, sig_sq_0):
    denom = (n / sig_sq + 1 / sig_sq_0)
    sig_sq_1 = 1 / denom
    mu_1 = (n * ybar / sig_sq + mu_0 / sig_sq_0) / denom
    return np.random.default_rng().normal(mu_1, np.sqrt(sig_sq_1))


def update_sig_sq(n, y, mu, nu_0, beta_0):
    nu_1 = nu_0 + n / 2
    sum_of_sq = np.sum((y-mu) ** 2)
    beta_1 = beta_0 + sum_of_sq / 2
    return 1 / np.random.default_rng().gamma(shape=nu_1, scale=1/beta_1) # Reciprocal is inverse gamma

def gibbs(y, n_iter, init, prior):
    """
    :param y: Data
    :param n_iter: Iterations
    :param init: Initial parameters
    :param prior: Prior parameters
    :return:
    """
    ybar = np.mean(y)
    n = len(y)
    mu_out = np.empty(n_iter)
    sig_sq_out = np.empty(n_iter)

    mu_now = init['mu']

    # Gibbs sampler
    for i in range(n_iter):
        sig_sq_now = update_sig_sq(n=n, y=y, mu=mu_now, nu_0=prior['nu_0'], beta_0=prior['beta_0'])
        mu_now = update_mu(n=n, ybar=ybar, sig_sq=sig_sq_now, mu_0=prior['mu_0'], sig_sq_0=prior['sig_sq_0'])
        sig_sq_out[i], mu_out[i] = sig_sq_now, mu_now

    return mu_out, sig_sq_out


if __name__ == '__main__':

    # Data
    #y = np.array([1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1, 0.1, 1.3, 1.9])
    y = np.array([-0.2, -1.5, -5.3, 0.3, -0.8, -2.2])

    init = {'mu': 0}
    prior = {'mu_0': 1,
             'sig_sq_0': 1,
             'n_0': 2,          # Prior effective sample size
             's2_0': 1}         # Prior guess for sigma squared
    prior['nu_0'] = prior['n_0'] / 2
    prior['beta_0'] = prior['n_0'] * prior['s2_0'] / 2

    plt.hist(y, density=True)
    plt.plot(y, [0] * len(y), 'o')
    plt.plot(np.mean(y), [0], 'ro')

    mu_post, sig_sq_post = gibbs(y, n_iter=5000, init=init, prior=prior)
    sns.kdeplot(mu_post)
    sns.kdeplot(sig_sq_post)
    print("mu mean/sd: {} / {}".format(np.mean(mu_post), np.std(mu_post)))
    print("sig mean/sd: {} / {}".format(np.mean(sig_sq_post), np.std(sig_sq_post)))
    plt.plot(mu_post, 'k')
    plt.legend(['data', 'mean', 'post_mu', 'post_sig', 'trace'])

    print("Autocorrelation: {}".format(acorr(mu_post, lags=[1, 5, 10, 20, 30, 50, 100])))
    plt.show()