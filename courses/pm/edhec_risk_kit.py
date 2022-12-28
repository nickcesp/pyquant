import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def annualize_rets(r: pd.Series or pd.DataFrame,
                   periods_per_year: int) -> pd.Series or pd.DataFrame:
    """ Takes a time series or dataframe of returns, annualizes them
        :param r: DataFrame or Series of returns as decimals, with index as PeriodIndex
        :param periods_per_year: eg. if monthly returns = 12

        :return: Series or DataFrame of annualized returns
    """

    total_return = (r + 1).prod()
    years_passed = r.shape[0] / periods_per_year
    return total_return ** (1 / years_passed) - 1


def annualize_vol(r: pd.Series or pd.DataFrame,
                  periods_per_year: int,
                  ddof=1) -> pd.Series or pd.DataFrame:
    """ Takes a time series or dataframe of returns, annualizes them
        :param r: DataFrame or Series of returns as decimals
        :param periods_per_year: eg. if monthly returns = 12

        :return: Series or DataFrame of annualized returns
    """

    return r.std(ddof=ddof) * (periods_per_year ** 0.5)


def compound_fast(r):
    return np.expm1(np.log1p(r).sum())


def compound_slow(r):
    return (r+1).prod()-1


def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index,
       the previous peaks, and
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index,
                         "Previous Peak": previous_peaks,
                         "Drawdown": drawdowns})

def get_ffme_returns(cols=('Lo 10', 'Hi 10'), rn_cols=('SmallCap', 'LargeCap')):
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv("resources/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, na_values=-99.99)
    if cols is not None:
        me_m = me_m[cols]
    if rn_cols is not None:
        me_m.columns = rn_cols

    me_m = me_m/100
    me_m.index = pd.to_datetime(me_m.index, format="%Y%m").to_period('M')
    return me_m


def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    # If you assume all returns are the same the optimizer minimizes variance and you get GMV
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level

def infer_periods_per_year(r: pd.Series or pd.DataFrame) -> int:
    """ Infers periods per year for series or DataFrame """
    # Todo: Improve
    if isinstance(r.index, pd.PeriodIndex):
        if r.index.freqstr == 'M':
            return 12
        elif r.index.freqstr == 'Y':
            return 1
        elif r.index.freqstr == 'D':
            return 365
        else:
            raise ValueError("Cannot infer periods per year")
    else:
        raise TypeError("Can only infer periods with PeriodIndex index")


def kurtosis(r: pd.Series or pd.DataFrame):
    """ Computes kurtosis of returns """
    demeaned = r - r.mean()
    demeaned_exp = demeaned ** 4
    pop_std = r.std(ddof=0)
    return demeaned_exp.mean() / (pop_std ** 4)


def minimize_vol(target_return, est_returns, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = est_returns.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0, 1),) * n

    # Construct constraints
    wts_sum_to_1 = {'type': 'eq',
                    'fun': lambda weights: np.sum(weights) - 1}
    tgt_return = {'type': 'eq',
                  'args': (est_returns,),
                  'fun': lambda weights, er: target_return - weights.T @ er}

    weights = minimize(port_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(wts_sum_to_1, tgt_return),
                       bounds=bounds)
    return weights.x


def msr(rf_rate, er, cov):
    """
        Returns the weights of the portfolio that gives you the maximum sharpe ratio
        given the riskfree rate and expected returns and a covariance matrix
        :param er: Expected returns (annualized)
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0, 1),) * n

    def neg_sharpe(weights, rf_rate, er, cov):
        """ Negative sharpe of port """
        r = port_returns(weights, er)
        vol = port_vol(weights, cov)
        return -(r - rf_rate)/vol

    wts_sum_to_1 = {'type': 'eq',
                    'fun': lambda weights: np.sum(weights) - 1}

    weights = minimize(neg_sharpe, init_guess,
                       args=(rf_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(wts_sum_to_1,),
                       bounds=bounds)
    return weights.x


def optimal_weights(n_points, er, cov):
    """
    Generates efficient frontier weights
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def plot_ef(n_points, er, cov, show_ew=False, show_gmv=False):
    """ Plots the multi-asset efficient frontier
        :param show_ew: Show naive equally weighted portfolio
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [port_returns(w, er) for w in weights]
    vols = [port_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    ef.plot.line(x="Volatility", y="Returns", style='.-',
                 title="{}-Asset Effic. Frontier".format(er.shape[0]))
    if show_ew:
        n = er.shape[0]
        e_wts = np.repeat(1 / n, n)
        eq_er = port_returns(e_wts, er)
        eq_vol = port_vol(e_wts, cov)
        plt.plot([eq_vol], [eq_er], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = port_returns(w_gmv, er)
        vol_gmv = port_vol(w_gmv, cov)
        plt.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)


def plot_ef2(n_points, er, cov):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [port_returns(w, er) for w in weights]
    vols = [port_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    return ef.plot.line(x="Volatility", y="Returns", style=".-", title="2-Asset Efficient Frontier")


def port_returns(weights: pd.Series or pd.DataFrame,
                 returns: pd.Series or pd.DataFrame):
    """
        :param weights: Series of asset weights (Usually == 1 but you might have some leverage)
        :param returns: Returns
    """
    return weights.T @ returns


def port_vol(weights, covmat):
    """ Computes portfolio volatility
        :param weights: Series of asset weights (Usually == 1 but you might have some leverage)
        :param covmat: Covariance matrix
    """
    return (weights.T @ covmat @ weights) ** 0.5


def semidev_est(r: pd.Series or pd.DataFrame):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    Approximation when mean returns are 0 see below for full correct formula
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)


def semideviation(r: pd.Series or pd.DataFrame):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    excess = r-r.mean()                                        # We demean the returns
    excess_negative = excess[excess<0]                        # We take only the returns below the mean
    excess_negative_square = excess_negative**2               # We square the demeaned returns below the mean
    n_negative = (excess<0).sum()                             # number of returns under the mean
    return (excess_negative_square.sum()/n_negative)**0.5     # semideviation


def sharpe_ratio(r: pd.Series or pd.DataFrame,
                 rf_rate: float,
                 periods_per_year: int) -> float or pd.Series:
    """ Computes the Sharpe ratio of the strategy

        :param r: Returns (Series or DataFrame with columns being different instruments)
        :param rf_rate: risk-free rate (annual)
        :param periods_per_year: Periods per
    """
    # Convert the risk_free rate to a per-period rate
    rf_period = (1 + rf_rate) ** (1 / periods_per_year) - 1
    excess = r - rf_period
    return annualize_rets(excess, periods_per_year) / annualize_vol(r, periods_per_year)


def skewness(r: pd.Series or pd.DataFrame):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp / (sigma_r**3)


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



def var_cond_historic(r, level=5):
    """
    Beyond Var. Expected loss for scenarios worse than VaR
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_cond_historic, level=level)
    elif isinstance(r, pd.Series):
        pctile_val = np.percentile(r, level)
        is_below_pctile = r < pctile_val
        return -r[is_below_pctile].mean()
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z += (z**2 - 1)*s/6 + (z**3 -3*z)*(k-3)/24 - (2*z**3 - 5*z)*(s**2)/36

    return -(r.mean() + z*r.std(ddof=0))


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
