import edhec_risk_kit as erk
import pandas as pd


if __name__ == '__main__':
    # Let's start by building 500 scenarios for interest rates, an duration matched bond portfolio
    # (proxied by a zero coupon bond) and a stock portfolio.
    n_scenarios = 5000
    rates, zc_prices = erk.cir(10, n_scenarios=n_scenarios, b=0.03, r_0=0.03, sigma=0.02)
    price_eq = erk.gbm(n_years=10, n_scenarios=n_scenarios, mu=0.07, sigma=0.15)
    rets_eq = price_eq.pct_change().dropna()
    rets_zc = zc_prices.pct_change().dropna()


    rets_7030b = erk.bt_mix(rets_eq, rets_zc, allocator=erk.fixedmix_allocator, w1=0.7)
    rets_floor75 = erk.bt_mix(rets_eq, rets_zc, allocator=erk.floor_allocator, floor=.75, zc_prices=zc_prices[1:])
    rets_floor75m1 = erk.bt_mix(rets_eq, rets_zc, allocator=erk.floor_allocator, zc_prices=zc_prices[1:], floor=.75,
                                m=1)
    rets_floor75m5 = erk.bt_mix(rets_eq, rets_zc, allocator=erk.floor_allocator, zc_prices=zc_prices[1:], floor=.75,
                                m=5)
    rets_floor75m10 = erk.bt_mix(rets_eq, rets_zc, allocator=erk.floor_allocator, zc_prices=zc_prices[1:], floor=.75,
                                 m=10)

    # Do max DD allocator
    rets_tmi = erk.get_total_market_index_returns()["1990":]
    dd_tmi = erk.drawdown(rets_tmi)
    cashrate = 0.03
    monthly_cashreturn = (1 + cashrate) ** (1 / 12) - 1
    rets_cash = pd.DataFrame(data=monthly_cashreturn, index=rets_tmi.index, columns=[0])  # 1 column dataframe
    rets_maxdd25 = erk.bt_mix(pd.DataFrame(rets_tmi), rets_cash, allocator=erk.drawdown_allocator, maxdd=.25, m=5)
    dd_25 = erk.drawdown(rets_maxdd25[0])

    # Print all tests
    print(pd.concat([erk.terminal_stats(rets_zc, name="ZC", floor=0.75),
           erk.terminal_stats(rets_eq, name="Eq", floor=0.75),
           erk.terminal_stats(rets_7030b, name="70/30", floor=0.75),
           erk.terminal_stats(rets_floor75, name="Floor75", floor=0.75),
           erk.terminal_stats(rets_floor75m1, name="Floor75m1", floor=0.75),
           erk.terminal_stats(rets_floor75m5, name="Floor75m5", floor=0.75),
           erk.terminal_stats(rets_floor75m10, name="Floor75m10", floor=0.75)
          ],
          axis=1).round(2))