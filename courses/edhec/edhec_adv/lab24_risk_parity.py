import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import courses.edhec.edhec_risk_kit as erk

if __name__ == '__main__':
    inds = ['Beer', 'Hlth', 'Fin','Rtail','Whlsl']
    ind_rets = erk.get_ind_returns(weighting="vw", n_inds=49)["1974":]
    ind_mcap = erk.get_ind_market_caps(49, weights=True)["1974":]

    rets = ind_rets["2013":][inds]
    cov = rets.cov()
    fig, axes = plt.subplots(3)

    erk.risk_contribution(erk.weight_ew(rets), cov).plot.bar(title="Risk Contributions of an EW portfolio",
                                                             ax=axes[0])
    erk.risk_contribution(erk.equal_risk_contributions(cov), cov).plot.bar(title="Risk Contributions of an ERC portfolio",
                                                                           ax=axes[1])

    # Run a backtest of industry portfolios w/ new erk portfolio
    ewr = erk.backtest_ws(ind_rets, estimation_window=36, weighting=erk.weight_ew)
    cwr = erk.backtest_ws(ind_rets, estimation_window=36, weighting=erk.weight_cw, cap_weights=ind_mcap)
    mv_erc_r = erk.backtest_ws(ind_rets, estimation_window=36, weighting=erk.weight_erc, cov_estimator=erk.sample_cov)
    btr = pd.DataFrame({"EW": ewr, "CW": cwr, 'ERC-Sample': mv_erc_r})
    (1 + btr).cumprod().plot(figsize=(12, 6), title="Industry Portfolios", ax=axes[2])
    print(erk.summary_stats(btr.dropna()))
    plt.show()