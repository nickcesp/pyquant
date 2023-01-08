from edhec_data import get_ind_returns, get_total_market_index_returns
from courses.edhec.edhec_risk_kit import annualize_rets, annualize_vol, sharpe_ratio, drawdown, skewness, kurtosis, var_gaussian, var_cond_historic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#TODO: Fucked up
def run_cppi_faster(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)

    if isinstance(risky_r, pd.Series) or isinstance(risky_r, pd.DataFrame):
        risky_r = risky_r.values

    if isinstance(risky_r, pd.Series) or isinstance(risky_r, pd.DataFrame):
        safe_r = safe_r.values
    else:
        safe_r = np.ones(risky_r.shape) * (riskfree_rate/12 if safe_r is None else safe_r)

    account_value = np.ones(risky_r.shape) * start
    floor_value = start * floor

    # set up some DataFrames for saving intermediate values
    account_history = [None] * n_steps
    risky_w_history = [None] * n_steps
    cushion_history = [None] * n_steps

    for step in range(n_steps):
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # recompute the new account value at the end of this step
        #print(risky_r[step])
        #print(safe_r[step])
        account_value = risky_alloc*(1+risky_r[step]) + safe_alloc*(1+safe_r[step])
        # save the histories for analysis and plotting
        cushion_history[step] = cushion
        risky_w_history[step] = risky_w
        account_history[step] = tuple(account_value)

    # Create DFs
    #account_history = pd.DataFrame(account_history).reindex_like(risky_r)
    #risky_w_history = pd.DataFrame(risky_w_history).reindex_like(risky_r)
    print(account_history)
    print(risky_w_history)
    #cushion_history = pd.DataFrame(cushion_history).reindex_like(risky_r)

    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r
    }
    return backtest_result


def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = start

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 # fast way to set all values to a number
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # recompute the new account value at the end of this step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r
    }
    return backtest_result



def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(var_cond_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })


if __name__ == '__main__':
    # Load the industry returns and the total market index we previously created
    ind_return = get_ind_returns()
    tmi_return = get_total_market_index_returns()

    r_risky = ind_return["2000":][["Steel", "Fin", "Beer"]]

    cppi = run_cppi(r_risky, riskfree_rate=0.03)

    cppi["Risky Allocation"].plot()
    plt.show()


