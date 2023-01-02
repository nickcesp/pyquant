import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    df = pd.read_csv('http://www.randomservices.org/random/data/Challenger2.txt', sep='\t')

    # Do a standard linear regression
    olsfit = sm.OLS.from_formula('I~T', data=df).fit()
    #print(olsfit.summary())
    p = olsfit.get_prediction({'T': [31]})
    #print("Prediction at 31 degrees: {}".format(p))
    #print(p.summary_frame(alpha=0.05))

    # Quiz 11:
    df = pd.read_csv('pgalpga2008.dat', names=['dist', 'accuracy', 'gender'])
    df['is_male'] = df['gender'].eq(2)
    df[df['is_male']].plot('dist', 'accuracy', title='Male', kind='scatter')
    df[~df['is_male']].plot('dist', 'accuracy', title='Female', kind='scatter')
    #plt.show()

    femfit = sm.OLS.from_formula('accuracy~dist', data=df[~df['is_male']]).fit()
    print("Female Posterior Parameters: {}".format(femfit.params))
    pr = femfit.get_prediction({'dist': [260]})
    sf = pr.summary_frame(alpha=0.05)
    print("Prediction for 260 yards: {}".format(pr.predicted_mean))
    print("95% Interval: ({}, {})".format(sf['obs_ci_lower'], sf['obs_ci_upper']))

    # Honors Quiz 4
    df['gender'] = df['gender'] - 1
    fullfit = sm.OLS.from_formula('accuracy~dist+gender', data=df).fit()
    print("H1: Full Posterior Parameters: {}".format(fullfit.params))
    sm.graphics.plot_partregress_grid(fullfit)
    plt.show()