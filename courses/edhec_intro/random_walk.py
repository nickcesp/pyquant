import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_yr=12, s_0=1):
    """ Geometric Brownian motion generator

        :param n_years: For how many years?
        :param n_scenarios: How many scenarios?
        :param mu: Expected return

    """

    dt = 1 / steps_per_yr
    n_steps = int(n_years * steps_per_yr)
    rets_p1 = np.random.normal(loc=mu*dt + 1, scale=sigma * np.sqrt(dt), size=(n_steps, n_scenarios))
    rets_p1[0] = 1
    prices = s_0 * np.cumprod(rets_p1, axis=0)
    return pd.DataFrame(prices)

if __name__ == '__main__':

    p = gbm(10, n_scenarios=20)
    print(p)
    p.plot()
    plt.show()
