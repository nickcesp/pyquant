import numpy as np
from scipy.stats import beta, gamma, norm
import matplotlib.pyplot as plt

def plot_beta(theta, a=1, b=1, ax=None):
    """ Plots Beta """
    if ax is not None:
        ax.plot(theta, beta.pdf(theta, a, b))
        ax.set_title("Beta({}, {})".format(a, b))
    else:
        plt.plot(theta, beta.pdf(theta, a, b), title="Beta({}, {})".format(a, b))


def plot_gamma(lambd, a=1, b=1, ax=None):
    if ax is not None:
        ax.plot(lambd, gamma.pdf(lambd, a, scale=1/b))
        ax.set_title("Gamma({}, {})".format(a, b))
    else:
        plt.plot(lambd, gamma.pdf(lambd, a, scale=1/b), title="Gamma({}, {})".format(a, b))


if __name__ == '__main__':

    fig, axes = plt.subplots(2, 2)
    theta = np.linspace(0, 1, num=100)
    lambd = np.linspace(0, 20, num=100)

    # Plot the Uniform distribution (Beta distribution w/ 1, 1)
    #plot_beta(theta, 1, 1, ax=axes[0, 0])
    plot_beta(theta, 2, 4, ax=axes[0, 1])
    plot_beta(theta, 8, 16, ax=axes[1, 0])
    #3333333plot_beta(theta, 1, 5, ax=axes[1, 1])
    print(beta.cdf([0.5], 1, 5))
    print(beta.ppf([.975], 8, 16))
    print(beta.cdf([0.35], 8, 16))
    print(beta.cdf([0.35], 8, 21))
    plot_gamma(lambd, 8, 1, ax=axes[0, 0])
    plot_gamma(lambd, 67, 6, ax=axes[0, 0])
    print(gamma.ppf([0.05], 67, scale=1/6))
    # Quiz 9: q3
    print(9.3)
    print(gamma.cdf([.1], 6, scale=1/93.5))
    print(9.9)
    # Quiz 9: q9
    print(gamma.ppf([.975], 9, scale=1/360))
    plot_gamma(np.linspace(0, 120, num=100), 9, 360, ax=axes[1, 1])

    #plt.show()