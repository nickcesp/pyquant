from courses.edhec.edhec_risk_kit import get_ind_returns, style_analysis
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Let's construct a manager that invests in 30% Beer, 50% in Smoke and 20% in other things that have an average
    # return of 0% and an annualized vol of 15%
    ind = get_ind_returns()  # Monthly returns
    print(ind)
    exit()
    mgr_r = 0.3 * ind["Beer"] + .5 * ind["Smoke"] + 0.2 * np.random.normal(scale=0.15 / (12 ** .5), size=ind.shape[0])

    # Now, assume we knew absolutely nothing about this manager and all we observed was the returns.
    # How could we tell what she was invested in?
    weights = style_analysis(mgr_r, ind) * 100
    weights.sort_values(ascending=False).head(6).plot.bar()
    plt.show()

    # One of the most common ways in which Sharpe Style Analysis can be used is to measure style drift. If you run the
    # style analysis function over a rolling window of 1 to 5 years, you can extract changes in the style
    # exposures of a manager.
    # We'll look at Rolling Windows in the next lab session.
    # As an exercise to the student, download a set of returns from Yahoo Finance, and try and measure the style drift
    # in your favorite fund manager. Use reliable Value and Growth ETFs such as "SPYG" and "SPYV" along with a
    # SmallCap ETF such as "SLY" and LargeCap ETF such as "OEF".
    # Alternately, the Fama-French research factors and use the Top and Bottom portfolios by Value (HML) and Size (SMB)
    # to categorize mutual funds into categories. This is very similar to the "Style Box" methodology employed by
    # Morningstar and displayed on their website. Compare your results with their results to see if they agree!