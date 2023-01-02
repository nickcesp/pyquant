import pandas as pd
import edhec_risk_kit as erk

import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # We are now ready to rerun the experiment we ran last time ... a bond portfolio of 60% in the 10 year bond
    # and 40% in the 30-year bond to generate a fixed mix bond portfolio.
    rates, zc_prices = erk.cir(10, 500, b=0.03, r_0=0.03)
    price_10 = erk.bond_price(10, 100, .05, 12, rates)
    price_30 = erk.bond_price(30, 100, .05, 12, rates)
    rets_30 = erk.bond_total_return(price_30, 100, .05, 12)
    rets_10 = erk.bond_total_return(price_10, 100, .05, 12)
    rets_bonds = erk.bt_mix(rets_10, rets_30, allocator=erk.fixedmix_allocator, w1=.6)
    mean_rets_bonds = rets_bonds.mean(axis='columns')
    print("Bond 60% 10YR 40% 30yr result")
    print(erk.summary_stats(pd.DataFrame(mean_rets_bonds)))

    # 70-30 STK BND
    price_eq = erk.gbm(n_years=10, n_scenarios=500, mu=0.07, sigma=0.15)
    rets_eq = price_eq.pct_change().dropna()
    rets_zc = zc_prices.pct_change().dropna()
    rets_7030b = erk.bt_mix(rets_eq, rets_bonds, allocator=erk.fixedmix_allocator, w1=0.7)
    rets_7030b_mean = rets_7030b.mean(axis='columns')
    print("70/30 Bond Stock")
    print(erk.summary_stats(pd.DataFrame(rets_7030b_mean)))

    plt.figure(figsize=(12, 6))
    sns.displot(erk.terminal_values(rets_eq), color="red", label="100% Equities")
    sns.displot(erk.terminal_values(rets_bonds), color="blue", label="100% Bonds")
    sns.displot(erk.terminal_values(rets_7030b), color="orange", label="70/30 Equities/Bonds")
    plt.legend()
    plt.show()

    rets_g8020 = erk.bt_mix(rets_eq, rets_bonds, allocator=erk.glidepath_allocator, start_glide=.8, end_glide=.2)

    rets_7030z = erk.bt_mix(rets_eq, rets_zc, allocator=erk.fixedmix_allocator, w1=0.7)
    plt.figure(figsize=(12, 6))
    # sns.distplot(terminal_values(rets_eq), color="red", label="100% Equities")
    # sns.distplot(terminal_values(rets_bonds), color="blue", label="100% Bonds")
    sns.distplot(erk.terminal_values(rets_7030b), color="orange", label="70/30 Equities/Bonds")
    # sns.distplot(terminal_values(rets_g8020), color="green", label="Glide 80 to 20")
    sns.distplot(erk.terminal_values(rets_7030z), color="grey", label="70/30 Equities/Zeros")