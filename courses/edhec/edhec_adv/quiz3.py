import courses.edhec.edhec_risk_kit as erk
import numpy as np
import pandas as pd

if __name__ == '__main__':

    # Given code
    ind49_rets = erk.get_ind_returns(weighting="vw", n_inds=49)["2013":]
    ind49_mcap = erk.get_ind_market_caps(49, weights=True)["2013":]
    inds = ['Hlth', 'Fin', 'Whlsl', 'Rtail', 'Food']
    rho_ = ind49_rets[inds].corr()
    vols_ = (ind49_rets[inds].std() * np.sqrt(12))
    sigma_prior_ = (vols_.T).dot(vols_) * rho_

    w = ind49_mcap[inds].iloc[0] / ind49_mcap[inds].iloc[0].sum()
    mu_i = erk.implied_returns(2.5, sigma_prior_, w)
    print("Cap weights: {}".format(w))
    print("Implied returns: {}".format(mu_i))

    # View Health will outperform Rtail and Whlsl by 3%
    q = pd.Series([0.05])
    denom = w.loc[['Rtail', 'Whlsl']].sum()
    p = pd.DataFrame([{'Hlth': 1,
                       'Fin': 0,
                       'Whlsl': -w.loc['Whlsl'] / denom,
                       'Rtail': -w.loc['Rtail'] / denom,
                       'Food': 0}])
    print("p_vec")
    print(p)
    mu_bl, sig_bl = erk.bl(w, sigma_prior_, p, q, delta=2.5, tau=0.025)
    print('mu_bl')
    print(mu_bl)
    msr_bl = erk.w_msr(sig_bl, mu_bl)
    print('msr_bl')
    print(msr_bl)
