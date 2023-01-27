import courses.edhec.edhec_risk_kit as erk
import statsmodels.api as sm
import pandas as pd

def part1():
    df_49vw = erk.get_ind_returns(n_inds=49).loc["1991":]
    fff = erk.get_fff_returns().loc["1991":"2018"]
    assert (df_49vw.index == fff.index).sum() == df_49vw.shape[0]

    # Question 1: What is the CAPM (Single Factor) Beta when evaluated over the entire period (1991-2018) of Beer?
    beer_excess = df_49vw['Beer'] - fff['RF']
    df_exp_var = fff[['Mkt-RF']].copy()
    df_exp_var['constant'] = 1
    ols = sm.OLS(beer_excess, df_exp_var).fit()
    # print(ols.summary())

    # Question 2: what is the CAPM Beta when evaluated over the entire period (1991-2018) of Steel?
    df_exp_var = fff[['Mkt-RF']].copy()
    df_exp_var['constant'] = 1
    ols = sm.OLS(df_49vw['Steel'] - fff['RF'], df_exp_var).fit()
    # print(ols.summary())

    # Question 3: what is the CAPM Beta when evaluated over the 2013-2018 (both included) period of Beer?
    beer_excess = df_49vw['Beer'].loc["2013":] - fff['RF'].loc["2013":]
    df_exp_var = fff.loc["2013":, ['Mkt-RF']].copy()
    df_exp_var['constant'] = 1
    ols = sm.OLS(beer_excess, df_exp_var).fit()
    # print(ols.summary())

    # Question 4: what is the CAPM Beta when evaluated over the 2013-2018 (both included) period of Steel?
    df_exp_var = fff.loc["2013":, ['Mkt-RF']].copy()
    df_exp_var['constant'] = 1
    ols = sm.OLS(df_49vw['Steel'].loc["2013":] - fff['RF'].loc["2013":], df_exp_var).fit()
    # print(ols.summary())

    # Question 5/6: Which of the 49 industries had the highest/lowest CAPM Beta when evaluated over the 1991-1993
    # (both included) period?
    df_excess = df_49vw.loc["1991":"1993"].sub(fff.loc["1991":"1993", 'RF'], axis=0)
    df_exp_var = fff.loc["1991":"1993", ['Mkt-RF']].copy()
    df_exp_var['constant'] = 1
    s_min, s_max, betas = None, None, dict()
    for sect in df_excess.columns:
        betas[sect] = sm.OLS(df_excess[sect], df_exp_var).fit().params.loc['Mkt-RF']
        if betas[sect] < betas.get(s_min, 99999):
            s_min = sect
        if betas[sect] > betas.get(s_max, -999999):
            s_max = sect

    print('Max beta sector 1991-1993: {}'.format((s_max, betas[s_max])))
    print('Min beta sector 1991-1993: {}'.format((s_min, betas[s_min])))


def part2():
    # Import data
    df_ffr = erk.get_fff_returns().loc['1991':]
    df_49vw = erk.get_ind_returns(n_inds=49).loc["1991":]
    df_49_excess = df_49vw.sub(df_ffr['RF'], axis=0)
    df_res = pd.DataFrame()

    exp_vars = df_ffr[['SMB', 'HML', 'Mkt-RF']].copy()
    exp_vars['const'] = 1
    for sect in df_49vw.columns:
        df_res[sect] = sm.OLS(df_49_excess[sect], exp_vars).fit().params

    df_res = df_res.transpose()

    print("Highest SmallCap Tilt: {}".format(df_res['SMB'].idxmax()))
    print("Highest Large Cap Tilt: {}".format(df_res['SMB'].idxmin()))
    print("Highest Value Tilt: {}".format(df_res['HML'].idxmax()))
    print("Highest Growth Tilt: {}".format(df_res['HML'].idxmin()))


if __name__ == "__main__":
    part2()