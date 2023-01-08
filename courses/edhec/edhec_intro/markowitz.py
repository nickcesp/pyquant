import pandas as pd
import numpy as np
from courses.edhec import edhec_risk_kit as erk
import matplotlib.pyplot as plt
import seaborn as sns


def get_ind_returns():
    ind = pd.read_csv("resources/ind30_m_vw_rets.csv", header=0, index_col=0) / 100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def part1():
    """ Lab 107 """
    ind = get_ind_returns()
    fig, axes = plt.subplots(nrows=3, ncols=2)

    # Plot drawdowns of Food Industry, Gaussian VaR, And Sharpe Ratios
    erk.drawdown(ind["Food"])["Drawdown"].plot.line(ax=axes[0, 0], title='Food Drawdown')
    erk.var_gaussian(ind).sort_values().plot.bar(ax=axes[0, 1], title='Gaussian VaR')
    erk.sharpe_ratio(ind, 0.03, 12).sort_values().plot.bar(ax=axes[1, 0], title='Sharpe (1926-2018)')
    erk.sharpe_ratio(ind["2000":], 0.03, 12).sort_values().plot.bar(ax=axes[1, 1], title='Sharpe (2000-)')

    # Get the covariance matrix
    cov = ind["1995":"2000"].cov()
    sns.heatmap(cov, ax=axes[2, 0])

    # Format chart and show
    fig.tight_layout(pad=0.1)
    fig.set_size_inches((10, 7))
    plt.show()


def part2():
    """ Lab 108: 2 Asset efficient frontier """
    ind = get_ind_returns()
    er = erk.annualize_rets(ind["1996":"2000"], 12)
    cov = ind["1996":"2000"].cov()

    # Example 1: Equally weight 4 assets
    assets = ["Food", "Beer", "Smoke", "Coal"]
    ex1_wts = np.repeat(0.25, 4)
    print("Example1 - Assets {}".format(assets))
    print("Returns: {}".format(erk.port_returns(ex1_wts, er[assets])))
    print("Vol: {}\n".format(erk.port_vol(ex1_wts, cov.loc[assets, assets])))

    # Now let's get efficient frontier for 2 assets
    n_pts = 20
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_pts)]
    assets = ["Games", "Fin"]
    ef = pd.DataFrame(data={'R': [erk.port_returns(w, er[assets]) for w in weights],
                            'V': [erk.port_vol(w, cov.loc[assets, assets]) for w in weights]})
    ef.plot.scatter(x='V', y='R')
    plt.show()


def part3():
    """ N-asset efficient frontier """
    ind = get_ind_returns()
    er = erk.annualize_rets(ind["1996":"2000"], 12)
    cov = ind["1996":"2000"].cov()

    # The Efficient Frontier for the protfolio that has a target return of 0.15 is approx 0.056,
    # so let's see if our optimizer is able to locate it.
    l = ["Games", "Fin"]
    erk.plot_ef2(20, er[l], cov.loc[l, l])

    tgt_ret = 0.15
    opt_wts = erk.minimize_vol(tgt_ret, er[l], cov.loc[l, l])
    vol_15 = erk.port_vol(opt_wts, cov.loc[l, l])
    assert round(vol_15, 3) == 0.056

    # Now plot the whole shibang
    l = ["Smoke", "Fin", "Games", "Coal"]
    erk.plot_ef(50, er[l], cov.loc[l, l])
    plt.show()


def part4():
    """ Max Sharpe Ratio Portfolio lab 110 """
    ind = get_ind_returns()
    er = erk.annualize_rets(ind["1996":"2000"], 12)
    cov = ind["1996":"2000"].cov()
    # plot EF
    ax = erk.plot_ef(20, er, cov)
    ax.set_xlim(left=0)
    # get MSR
    rf = 0.1
    w_msr = erk.msr(rf, er, cov)
    r_msr = erk.port_returns(w_msr, er)
    vol_msr = erk.port_vol(w_msr, cov)
    # add CML
    cml_x = [0, vol_msr]
    cml_y = [rf, r_msr]
    ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
    plt.show()


def part5():
    """ Estimating GMV (Global-Minimum-Variance Portfolio) Lab 111
        Not sensitive to bad expected returns inputs
    """
    ind = get_ind_returns()
    er = erk.annualize_rets(ind["1996":"2000"], 12)
    cov = ind["1996":"2000"].cov()

    # Show the efficient frontier w/ the equally weighted port
    erk.plot_ef(20, er, cov, show_ew=True, show_gmv=True)
    plt.show()



if __name__ == '__main__':
    part5()