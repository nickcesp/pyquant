import pandas as pd
import matplotlib.pyplot as plt

import courses.edhec.edhec_risk_kit as erk

if __name__ == '__main__':
    ind_rets = erk.get_ind_returns(weighting="vw", n_inds=49)["2014":]
    ind_mcap = erk.get_ind_market_caps(49, weights=True)["1974":]

    s_cov = erk.sample_cov(ind_rets)
    cw_port = erk.weight_cw(ind_rets, ind_mcap)
    cw_port_rc = erk.risk_contribution(cw_port, s_cov).sort_values()
    print("Industry risk contributions of Cap-Weighted:")
    print(cw_port_rc)

    ew_port = erk.weight_ew(ind_rets)
    ew_port_rc = erk.risk_contribution(ew_port, s_cov).sort_values()
    print("Industry risk contributions of Equally-Weighted:")
    print(ew_port_rc)

    erc_port = pd.Series(erk.weight_erc(ind_rets), index=ind_rets.columns)
    print("Sector weights of ERC Portfolio:")
    print(erc_port.sort_values())

    print("Difference between highest-lowest CW {}".format(cw_port_rc.iloc[-1] - cw_port_rc.iloc[0]))
    print("Difference between highest-lowest EW {}".format(ew_port_rc.iloc[-1] - ew_port_rc.iloc[0]))