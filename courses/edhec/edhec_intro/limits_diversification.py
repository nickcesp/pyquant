""" Lab 118: Limits of Diversification """
from edhec_data import get_ind_size, get_ind_nfirms, get_ind_returns
from courses.edhec import edhec_risk_kit as erk
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Figures
    fig, axes = plt.subplots(3, 2)

    # Load the data
    ind_return = get_ind_returns()
    ind_nfirms = get_ind_nfirms()
    ind_size = get_ind_size()

    # Calculate market capitalization
    ind_mktcap = ind_nfirms * ind_size

    # Add up total market cap over time
    total_mktcap = ind_mktcap.sum(axis=1)
    total_mktcap.plot(ax=axes[0, 0], title='Total Market Cap')

    # Compute the capital weights of the index
    ind_capwt = ind_mktcap.divide(total_mktcap, axis='rows')
    assert all(abs(ind_capwt.sum(axis="columns") - 1) < 1E-10)

    # Fraction of steel / fin over time
    ind_capwt[['Steel', 'Fin']].plot(ax=axes[1, 0], title="% Steel/Fin in Indx")

    # Construct Cap wtd index
    total_market_return = (ind_capwt * ind_return).sum(axis="columns")
    total_market_index = erk.drawdown(total_market_return)['Wealth']
    total_market_index.plot(ax=axes[2, 0], title="Total Market Cap Weighted Index 1926-2018")

    # ROLLING WINDOWS
    # Compute moving avg over a trailing 36-month period
    total_market_index["1980":].plot(ax=axes[0, 1], title="1980 Moving Avg")
    total_market_index["1980":].rolling(window=36).mean().plot(ax=axes[0, 1])

    tmi_tr36rets = total_market_return.rolling(window=36).aggregate(erk.annualize_rets, periods_per_year=12)
    tmi_tr36rets.plot(ax=axes[1, 1], label="Tr 36 mo Returns", legend=True)
    total_market_return.plot(ax=axes[1, 1], label="Returns", legend=True)

    # How does correlation behave (measure avg. correlation accross industries)
    # Is there a relationship between avg correlation and returns?
    # Rolling correlation computation - MultiIndexes and .groupby()
    ts_mat_corr = ind_return.rolling(window=36).corr()  # This gives us a MultiIndex (dt, ind) and cols ind w/ corr mat
    ts_mat_corr.index.names = ['date', 'industry']
    ts_avg_corr = ts_mat_corr.groupby(level='date').apply(lambda cormat: cormat.values.mean())
    tmi_tr36rets.plot(ax=axes[2, 1], label="Tr 36 mo return", legend=True, secondary_y=True)
    ts_avg_corr.plot(ax=axes[2, 1], legend=True, label="Tr 36 mo Avg Correlation")

    print("Correlation between correlation and returns: {}".format(tmi_tr36rets.corr(ts_avg_corr)))

    plt.show()