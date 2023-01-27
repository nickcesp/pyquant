import pymc as pm
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 1. Specify the model
    # Example y_i | mu ~ N(mu, 1)   i = 1,..,n
    # mu ~ t(0, 1, 1)
    y_obs = [1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1, 0.1, 1.3, 1.9]

    # 2. Setup the model
    with pm.Model() as model:
        # Prior
        mu = pm.StudentT('mu', nu=1, mu=0, sigma=1)

        # Likelyhood
        p_y = pm.Normal('p_y', mu=mu, sigma=1, observed=y_obs)

        # 3. Sample
        samples = pm.sample(draws=1000, tune=1000)

    print(samples)
    sns.kdeplot(samples.posterior['mu'])
    plt.show()