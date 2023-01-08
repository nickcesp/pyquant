import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

if __name__ == '__main__':

    # Parameters of gamma
    m = 10000
    a = 2
    b = 1/3

    # Initialize random number gen
    gen = np.random.default_rng()
    theta_sample = gen.gamma(shape=a, scale=1/b, size=m)

    # Plot sample gamma and distribution | density=True plots density instead of counts
    plt.hist(theta_sample, bins=20, density=True)
    x = np.linspace(0, 20, 1000)
    y = gamma.pdf(x, a, scale=1/b)
    theta_bar = np.mean(theta_sample)
    plt.plot(x, y)
    print("Approx exp value vs real: {} vs {}".format(theta_bar, a / b))
    plt.show()

    # se of estimate
    se = np.std(theta_sample) / np.sqrt(m)
    conf_95 = (theta_bar - 2 * se, theta_bar + 2 * se)