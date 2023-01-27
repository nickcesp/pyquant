import numpy as np
import pandas as pd
import courses.edhec.edhec_risk_kit as erk
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ind_cw = erk.get_ind_returns(weighting='vw')
    ind_ew = erk.get_ind_returns(weighting='ew')

    kw = {"riskfree_rate": 0.03, "periods_per_year": 12}
    cw_rolling_sharpe = ind_cw.rolling(60).apply(erk.sharpe_ratio, raw=True, kwargs=kw).mean(axis=1)
    ax = cw_rolling_sharpe["1945":].plot(figsize=(12, 5), label="CW", legend=True)
    ew_rolling_sharpe = ind_ew.rolling(60).apply(erk.sharpe_ratio, raw=True, kwargs=kw).mean(axis=1)["1945":]
    ew_rolling_sharpe.plot(ax=ax, label="EW", legend=True)
    ax.set_title("Average Trailing 5 year Sharpe Ratio across 30 Industry Portfolios 1945-2018")
    plt.show()
