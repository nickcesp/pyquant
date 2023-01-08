import statsmodels.api as sm
import pandas as pd
from courses.edhec.edhec_risk_kit import compound_fast

def get_fff_returns():
    """
    Load the Fama-French Research Factor Monthly Dataset
    """
    rets = pd.read_csv("data/F-F_Research_Data_Factors_m.csv",
                       header=0, index_col=0, na_values=-99.99)/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets


if __name__ == '__main__':
    # We have business day returns of BRKA
    brka_d = pd.read_csv("data/brka_d_ret.csv", parse_dates=True, index_col=0)

    # Convert to monthly using resample
    brka_m = brka_d.resample('M').apply(compound_fast).to_period('M')

    # Get Fama-French monthly returns
    fff = get_fff_returns()

    # Next, we need to decompose the observed BRKA 1990-May 2012 as in Ang(2014) into the portion that's due to the
    # market and the rest that is not due to the market, using the CAPM as the explanatory model.
    #𝑅𝑏𝑟𝑘𝑎,𝑡−𝑅𝑓,𝑡=𝛼+𝛽(𝑅𝑚𝑘𝑡,𝑡−𝑅𝑓,𝑡)+𝜖𝑡
    brka_excess = brka_m["1990":"2012-05"] - fff.loc["1990":"2012-05", ['RF']].values
    mkt_excess = fff.loc["1990":"2012-05", ['Mkt-RF']]

    # Explanatory variable (Market excess return, to find coefficient Beta)
    exp_var = mkt_excess.copy()
    exp_var["Constant"] = 1
    lm = sm.OLS(brka_excess, exp_var).fit()
    print("Summary of CAPM Fit for BRKA excess return:")
    print(lm.summary())
    # This implies that the CAPM benchmark consists of 46 (1-Beta) cents in T-Bills and 54 cents (Beta) in the market.
    # i.e. each dollar in the Berkshire Hathaway portfolio is equivalent to 46 cents in T-Bills and 54 cents in the
    # market. Relative to this, the Berkshire Hathaway is adding (i.e. has 𝛼 of) 0.61% (per month!) although the
    # degree of statistica significance is not very high.

    # Now, let's add in some additional explanatory variables, namely Value and Size.
    exp_var["Value"] = fff.loc["1990":"2012-05", ['HML']]
    exp_var["Size"] = fff.loc["1990":"2012-05", ['SMB']]
    lm = sm.OLS(brka_excess, exp_var).fit()
    print("Summary of French-Fama fit")
    print(lm.summary())
    # The alpha has fallen from .61% to about 0.55% per month. The loading on the market has moved up from 0.54 to 0.67,
    # which means that adding these new explanatory factors did change things. If we had added irrelevant variables,
    # the loading on the market would be unaffected.
    # We can interpret the loadings on Value being positive as saying that Hathaway has a significant Value tilt - which
    # should not be a shock to anyone that follows Buffet. Additionally, the negative tilt on size suggests that
    # Hathaway tends to invest in large companies, not small companies. In other words, Hathaway appears to be a Large
    # Value investor. Of course, you knew this if you followed the company, but the point here is that numbers reveal it
    # The new way to interpret each dollar invested in Hathaway is: 67 cents in the market, 33 cents in Bills,
    # 38 cents in Value stocks and short 38 cents in Growth stocks, short 50 cents in SmallCap stocks and long 50
    # cents in LargeCap stocks. If you did all this, you would still end up underperforming Hathaway by about 55 basis
    # points per month.