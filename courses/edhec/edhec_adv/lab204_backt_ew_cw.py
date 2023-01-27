import numpy as np
import pandas as pd
import courses.edhec.edhec_risk_kit as erk
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ind49_rets = erk.get_ind_returns(weighting="vw", n_inds=49)["1974":]
    ind49_mcap = erk.get_ind_market_caps(49, weights=True)["1974":]
    ewr = erk.backtest_ws(ind49_rets)
    ewtr = erk.backtest_ws(ind49_rets, cap_weights=ind49_mcap, max_cw_mult=5, microcap_threshold=.005)
    cwr = erk.backtest_ws(ind49_rets, weighting=erk.weight_cw, cap_weights=ind49_mcap)
    btr = pd.DataFrame({"EW": ewr, "EW-Tethered": ewtr, "CW": cwr})
    (1+btr).cumprod().plot(figsize=(12,5))
    erk.summary_stats(btr.dropna())
    plt.show()

    # One of the motivations of adding the tethering constraint is to improve tracking error to the cap-weighted
    # portfolio. Let's see if we did manage to achieve that:
    erk.tracking_error(ewr, cwr), erk.tracking_error(ewtr, cwr)