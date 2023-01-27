import courses.edhec.edhec_risk_kit as erk
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ind_rets = erk.get_ind_returns(weighting="vw", n_inds=30)["1997":"2018"]
    ind_mcap = erk.get_ind_market_caps(30, weights=True)["1997":"2018"]

    # Equally weighted
    ewr = erk.backtest_ws(ind_rets, estimation_window=36, weighting=erk.weight_ew)

    # Cap weighted portfolio
    cwr = erk.backtest_ws(ind_rets, estimation_window=36, weighting=erk.weight_cw, cap_weights=ind_mcap)

    # Tethered EW portfolios
    tewr = erk.backtest_ws(ind_rets, estimation_window=36, weighting=erk.weight_ew, microcap_threshold=0.01,
                           max_cw_mult=2, cap_weights=ind_mcap)

    # GMV portfolio
    gmvr = erk.backtest_ws(ind_rets, estimation_window=36, weighting=erk.weight_gmv, cov_estimator=erk.sample_cov)

    # GMV_R
    gmv_shr = erk.backtest_ws(ind_rets, estimation_window=36, weighting=erk.weight_gmv, cov_estimator=erk.shrinkage_cov,
                              delta=0.25)

    btr = pd.DataFrame({"EW": ewr, "CW": cwr, 'TEWR': tewr, 'GMV': gmvr, 'GMVR_SH': gmv_shr})
    #(1 + btr).cumprod().plot(figsize=(12, 6), title="Industry Portfolios - CW vs EW")
    print(erk.summary_stats(btr.dropna()))

    print("Tracking error between EW and CW: {}".format(erk.tracking_error(ewr, cwr)))
    print("Tracking error between TEW and CW: {}".format(erk.tracking_error(tewr, cwr)))