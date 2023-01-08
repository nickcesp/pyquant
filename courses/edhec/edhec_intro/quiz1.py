import pandas as pd

from courses.edhec.edhec_risk_kit import get_ffme_returns

def get_returns(r: pd.DataFrame or pd.Series):
    total_return = (r + 1).cumprod().iloc[-1]
    monthly_return = total_return ** (1 / r.shape[0])
    annualized_return = monthly_return ** 12

    print("Total Return: {}".format(total_return - 1))
    print("Monthly Return: {}".format(monthly_return - 1))
    print("Annualized Return: {}\n\n".format(annualized_return - 1))


def get_vol(r: pd.DataFrame or pd.Series):
    monthly_vol = r.std(ddof=0)
    annualized_vol = monthly_vol * (12 ** (1 / 2))
    print("Monthly Vol: {}".format(monthly_vol))
    print("Annualized Vol: {}\n\n".format(annualized_vol))


def drawdown(r: pd.DataFrame or pd.Series):
    wealth_idx = (r + 1).cumprod()
    wealth_idx_top = wealth_idx.cummax()
    drawdown = 100 * (wealth_idx_top - wealth_idx) / wealth_idx_top

    return wealth_idx, wealth_idx_top, drawdown


if __name__ == '__main__':

    # Question 1/3: What's the annualized return of Lo 20?
    df_lh_20 = get_ffme_returns(cols=['Lo 20', 'Hi 20'], rn_cols=['L20', 'H20'])
    get_returns(df_lh_20)

    # Question 2/4: What's annualized vol?
    get_vol(df_lh_20)

    # Question 5/6: Annualized returns 1999-2015
    df_period = df_lh_20.loc["1999":"2015"]
    get_returns(df_period)
    get_vol(df_period)

    # Question 9-12: What was max draw-down over 1999-2015?
    _, _, dd = drawdown(df_period)
    when_max = dd.idxmax()
    print("Max drawdown between 1999-2015")
    print(dd.loc[when_max])
    print('\n\n')

    # Questions 13 - 16: Using fund data
    df_hfi = pd.read_csv('resources/edhec-hedgefundindices.csv', index_col=0)
    df_hfi.index = pd.to_datetime(df_hfi.index, format='%d/%m/%Y').to_period('M')
    df_hfi_09 = df_hfi.loc["2009":"2018"]

    excess_ret = df_hfi_09 - df_hfi_09.mean()
    is_neg_er = excess_ret < 0
    semidev = (excess_ret[is_neg_er] ** 2).mean() ** (1/2)

    print(semidev.sort_values())
    print('\n\n')

    # Question 15: Skew
    skew = ((df_hfi_09 - df_hfi_09.mean()) ** 3).mean() / (df_hfi_09.std(ddof=0) ** 3)
    print('Skewness')
    print(skew.sort_values())
    print('\n\n')

    # Question 16: Kurtosis
    kurt = ((df_hfi_09 - df_hfi_09.mean()) ** 4).mean() / (df_hfi_09.std(ddof=0) ** 4)
    print('Kurtosis')
    print(kurt.sort_values())

