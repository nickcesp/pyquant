import courses.edhec.edhec_risk_kit as erk
import pandas as pd
import matplotlib.pyplot as plt
# We'll now move on the more sophisticated portfolio construction techniques, but they will get us involved in the
# estimation game, something we've avoided so far ... so let's start by pulling in the data we need and start with
# the CW and EW portfolios, since they are the baseline portfolios.

if __name__ == '__main__':
    inds = ['Food', 'Beer', 'Smoke', 'Games', 'Books', 'Hshld', 'Clths', 'Hlth',
           'Chems', 'Txtls', 'Cnstr', 'Steel', 'FabPr', 'ElcEq', 'Autos', 'Carry',
           'Mines', 'Coal', 'Oil', 'Util', 'Telcm', 'Servs', 'BusEq', 'Paper',
           'Trans', 'Whlsl', 'Rtail', 'Meals', 'Fin', 'Other']
    #inds=['Beer', 'Hlth', 'Fin','Rtail','Whlsl']
    ind_rets = erk.get_ind_returns(weighting="ew", n_inds=49)["1974":]
    ind_mcap = erk.get_ind_market_caps(49, weights=True)["1974":]

    # Equally weighted portfolio
    ewr = erk.backtest_ws(ind_rets, estimation_window=36, weighting=erk.weight_ew)

    # Cap weighted portfolio
    cwr = erk.backtest_ws(ind_rets, estimation_window=36, weighting=erk.weight_cw, cap_weights=ind_mcap)

    # Minimum variance portfolio
    mv_s_r = erk.backtest_ws(ind_rets, estimation_window=36, weighting=erk.weight_gmv, cov_estimator=erk.sample_cov)

    # Now, let's try a new estimator - Constant Correlation. The idea is simple, take the sample correlation matrix,
    # compute the average correlation and then reconstruct the covariance matrix. The relation between correlations 𝜌
    # and covariance 𝜎 is given by: rho_ij = sig_ij / sqrt(sig_ii * sig_jj)
    mv_cc_r = erk.backtest_ws(ind_rets, estimation_window=36, weighting=erk.weight_gmv, cov_estimator=erk.cc_cov)

    # We can mix the model and sample estimates by choosing a shrinkage parameter. You can either let the numbers
    # dictate an optimal shrinkage value for 𝛿 although in practice many practitioners choose 0.5. Let's implement
    # a simple shrinkage based covariance estimator that shrinks towards the Constant Correlation estimate.
    mv_sh_r = erk.backtest_ws(ind_rets, estimation_window=36, weighting=erk.weight_gmv, cov_estimator=erk.shrinkage_cov,
                              delta=0.5)

    # Put them all together and calculate returns
    btr = pd.DataFrame({"EW": ewr, "CW": cwr, "GMV-Sample": mv_s_r, "GMV-CC": mv_cc_r, 'GMV-Shrink 0.5': mv_sh_r})
    (1 + btr).cumprod().plot(figsize=(12, 6), title="Industry Portfolios - CW vs EW")
    print(erk.summary_stats(btr.dropna()))
    plt.show()


