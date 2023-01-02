import numpy as np
import numpy.random
from scipy.stats import norm

if __name__ == '__main__':
    print("10.4: {}".format(norm.ppf([.975], loc=96.17, scale=np.sqrt(.042))))
    print("10.5: {}".format(norm.cdf([100], loc=96.17, scale=np.sqrt(.042))))

    # Random number generator
    sim_N = 10000
    gen = numpy.random.default_rng()

    # Quiz 10.8: Simulate random samples from inverse gamma distribution
    sample = np.reciprocal(gen.gamma(3, scale=1/200, size=sim_N))
    print("10.8: Rnd sample mean: {}".format(np.mean(sample)))

    # Quiz 10.9: Simulate and come up with 95% confidence interval
    n = 27
    a_post = 16.5
    b_post = 6022.9
    m_post = 609.3
    w = 0.1

    # Start by drawing samples from sigma_squared
    # sig^2 | y ~ Inverse-Gamme(a_p, b_p)
    sig_sqrd_B = np.reciprocal(gen.gamma(a_post, scale=1 / b_post, size=sim_N))

    # Use sigma_sqrd to draw samples
    mu_samples_B = gen.normal(loc=m_post, scale=np.sqrt(np.divide(sig_sqrd_B, n + w)), size=sim_N)

    # Create 95% confidence
    print("10.9: 95% confidence interval for mu: {}".format(np.quantile(mu_samples_B, [0.025, 0.975])))

    # Quiz 10.10: Repeat simulation for restaurant A data
    n = 30
    m_pr = 500
    a_pr, b_pr = 3, 200
    y_bar = 622.8
    s_sqrd = 403.1
    a_post = a_pr + n/2
    b_post = b_pr + s_sqrd*(n-1)/2 + (0.5*w*n/(w+n))*((y_bar - m_pr)**2)
    m_post = (n*y_bar + w*m_pr) / (w+n)

    sig_sqrd_A = np.reciprocal(gen.gamma(a_post, scale=1 / b_post, size=sim_N))
    mu_samples_A = gen.normal(loc=m_post, scale=np.sqrt(np.divide(sig_sqrd_A, n + w)), size=sim_N)
    print("10.10: % mu_A > mu_B = {}".format(np.sum(mu_samples_A > mu_samples_B) / sim_N))


