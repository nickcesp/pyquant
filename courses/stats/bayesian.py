import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

"""
Prior p(H): Our prior reflects what we know about the value of some parameter before seeing data. This could refer to previous trials and distributions.

Likelihood p(D|H): what is the plausibility that our data is observed, given our prior?

Posterior p(H|D): This is result of the Bayesian analysis and reflects all that we know about a problem (given our data and model).

Evidence p(D): Evidence is the probability of observing the data averaged over all the possible values the parameters can take. Also knowns as the noramlziing factor. The normalising constant makes sure that the resulting posterior distribution is a true probability distribution by ensuring that the sum of the distribution is equal to 1.

Because p(D) is considered a normalizing constant we can say: p(H|D)∝p(D|H)∗p(H)

"""

def bayes_ex_coins():
    coin_flips_prior = np.random.binomial(n=1, p=0.5, size=1000)
    p_guess = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(3, 1)

    # Bernoulli pmf gives us... st.bernoulli.pmf(coin_flips_prior, p)
    # For a given coin flip, what is the probability that the probability is p?
    # The product of that gives us the prob of seeing that entire sequence of flips
    p_prior = [np.product(st.bernoulli.pmf(coin_flips_prior, p)) for p in p_guess]
    p_prior = np.array(p_prior) / np.sum(p_prior)
    axes[0].plot(p_guess, p_prior)
    axes[0].set_title('p_prior')
    sns.despine()  # Not sure what this does

    # Now, let's introduce some observations from trials with an unfair coin.
    # Let's say the probability is now weight 80-20, where the probability a head is shown is 0.8.
    coin_flips_observed = np.random.binomial(n=1, p=0.8, size=1000)
    p_observed = [np.product(st.bernoulli.pmf(coin_flips_observed, p)) for p in p_guess]
    p_observed = np.array(p_observed) / np.sum(p_observed)
    axes[1].plot(p_guess, p_observed)
    axes[1].set_title('p_observed')

    # While our observations from our sampling distribution indicate a probability around 0.8, because our prior is 0.5,
    # we have to assess the likelihood that these values could be observed and find our posterior distribution.
    p_post = p_prior * p_observed
    p_post = p_post / np.sum(p_post)
    axes[2].plot(p_guess, p_post)
    axes[2].set_title('p_post')
    plt.show()


def norm_stud_iq():
    #We'll do another example where we have some prior belief about the IQ of University of Michigan students.
    #For our prior distribution, we'll have a normal distribution with a mean IQ of 100 and a standard deviation of 10.
    fig, axes = plt.subplots(3, 1)
    prior_distribution = np.random.normal(100, 10, 1000)
    plt.hist(prior_distribution)

    #Now, let's say we are collecting some observations of student IQs which takes the shape of a normal distribution
    #with mean 115 and standard deviation of 7.5 and want to construct our posterior distribution.
    np.random.seed(5)
    observed_distribution = np.random.normal(115, 10, 1000)
    mu = [100] * 1000
    sigma = [10] * 1000

    mu[0] = (10 ** 2 * observed_distribution[0] + (10 ** 2) * 100) / (10 ** 2 + 10 ** 2)
    sigma[0] = (10 ** 2 * 10 ** 2) / (10 ** 2 + 10 ** 2)

    for i in range(1000):
        if i == 999:
            break
        mu[i + 1] = (sigma[i] * observed_distribution[i + 1] + (10 ** 2) * mu[i]) / (sigma[i] + 10 ** 2)
        sigma[i + 1] = (sigma[i] * 10 ** 2) / (sigma[i] + 10 ** 2)

    posterior_distributions = [[]] * 20

    for i in range(20):
        posterior_distributions[i] = np.random.normal(mu[i], sigma[i], 1000)

    plt.hist(prior_distribution)
    plt.hist(observed_distribution, alpha=0.75)
    plt.hist(posterior_distributions[14], alpha=0.5)
    sns.despine()


if __name__ == '__main__':
    bayes_ex_coins()