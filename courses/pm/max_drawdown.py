import pandas as pd
import matplotlib.pyplot as plt


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

if __name__ == '__main__':
    me_m = pd.read_csv("resources/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, parse_dates=True, na_values=-99.99)

    # Check if the date was converted
    if me_m.index.dtype != 'datetime64[ns]':
        me_m.index = pd.to_datetime(me_m.index, format='%Y%m')

    # Exctract the returns
    df_rets = me_m[['Lo 10', 'Hi 10']].rename(columns={'Lo 10': 'SmallCap', 'Hi 10': 'LargeCap'})

    # Convert index from timestamp to monthly period
    df_rets.index = df_rets.index.to_period('M')

    # Convert from % to real number
    df_rets /= 100

    # We want maxdrawdown, so let's compute the wealth index
    df_wealth = (1 + df_rets).cumprod()
    fig, axes = plt.subplots(nrows=3, ncols=2)
    df_wealth['SmallCap'].plot(ax=axes[0, 0], title='SmallCap W.I')
    df_wealth['LargeCap'].plot(ax=axes[0, 1], title='LargeCap W.I.')

    # Now we want to get previous peaks
    df_wealth[['SC_max', 'LC_max']] = df_wealth.cummax()
    df_wealth[['SmallCap', 'SC_max']].plot(ax=axes[1, 0], title='Large Cap Max')
    df_wealth[['LargeCap', 'LC_max']].plot(ax=axes[1, 1], title='Large Cap Max')

    # Max Drawdown:
    df_wealth[['SC_dd', 'LC_dd']] = -1 + (df_wealth[['SmallCap', 'LargeCap']].values / df_wealth[['SC_max', 'LC_max']].values)
    df_wealth['SC_dd'].plot(ax=axes[2, 0])
    df_wealth['LC_dd'].plot(ax=axes[2, 1])

    print("Small Cap Max Drawdown: {}".format(df_wealth['SC_dd'].min()))
    print("Large Cap Max Drawdown: {}".format(df_wealth['LC_dd'].min()))

    print("Small Cap Max Drawdown since 1975: {}".format(df_wealth['SC_dd']["1975":].min()))
    print("Large Cap Max Drawdown since 1975: {}".format(df_wealth['LC_dd']["1975":].min()))

    plt.show()