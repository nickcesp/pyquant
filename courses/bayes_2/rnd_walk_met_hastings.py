# y_i | mu ~iid~ N(mu, 1), i = 1,...,n
# mu ~ t(0,1,1)
import math
from scipy.stats import t
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# This model is not conjugate
# We need to setup Markov chain whose stationary distribution is the posterior distribution we want
# Posterior has for P(mu | y1...yn) prop exp[n*(y_bar*mu - mu*mu/2)] / (1+mu*mu) = g(mu)
# Some of these numbers will be really small, so let's work with log(g) for stability
# log(g(mu)) = n*(y_bar*mu - mu*mu/2) - log(1+mu*mu)

def log_g_mu(mu, n, y_bar):
    """ Log of the posterior """
    mu_sqrd = mu ** 2
    return n * (y_bar * mu - mu_sqrd/2) - math.log(1 + mu_sqrd)

def rw_met_hastings(n, y_bar, n_iter, theta_0, cand_sampling_func, cand_sampling_kwargs):
    """ Runs metropolis-hastings with normal candidate distribution """
    # Step 1 initialize
    theta_out = np.empty(n_iter)
    accpt = 0
    theta_now = theta_0
    lg_now = log_g_mu(theta_now, n, y_bar)
    rnd = np.random.default_rng()

    # Step 2 simulate
    for i in range(n_iter):
        # 2.a Draw from candidate sampling function
        theta_cand = cand_sampling_func(loc=theta_now, **cand_sampling_kwargs)
        lg_cand = log_g_mu(theta_cand, n, y_bar)
        log_alpha = lg_cand - lg_now
        alpha = np.exp(log_alpha)

        if rnd.uniform() < alpha:
            theta_now = theta_cand
            lg_now = lg_cand
            accpt += 1

        theta_out[i] = theta_now

    return list(theta_out), accpt / n_iter

if __name__ == '__main__':

    # Run sim
    rdn = np.random.default_rng()
    y = [1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1, 0.1, 1.3, 1.9]
    y_bar = np.mean(y)
    n = len(y)

    # Plot the histogram of the data, points and mean in y axis and the prior t distribution
    plt.hist(y, density=True, range=(-1, 3), bins=6)
    plt.plot(y, [0] * n, 'o')
    plt.plot(y_bar, [0], 'ro')
    x_curve = np.linspace(-1, 3, 500)
    plt.plot(x_curve, t.pdf(x_curve, df=1), 'k-.')
    plt.legend(['y', 'y_bar', 'prior'])

    # Do the sampling
    # If you run it with sd = 3 we get an acceptance rate of about 13% (too small step size, lower sd)
    # If you run with sd 0.05 we get too high an acceptance rate
    rdn = np.random.default_rng()
    post, accpt_rate = rw_met_hastings(n=n,
                                       y_bar=y_bar,
                                       n_iter=1000,
                                       theta_0=0,
                                       cand_sampling_func=rdn.normal,
                                       cand_sampling_kwargs={'scale': 0.9})

    print("Acceptance rate: {}".format(accpt_rate))

    # Post analysis
    mu_keep = post[:900]
    sns.kdeplot(mu_keep)
    plt.legend(['y', 'y_bar', 'prior', 'post'])
    plt.show()