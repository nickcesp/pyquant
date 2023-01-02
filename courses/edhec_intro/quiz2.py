import edhec_risk_kit as erk
from courses.edhec_intro.edhec_data import get_ind_returns, get_hfi_data

if __name__ == "__main__":
    df_hfi = get_hfi_data()
    r_dist = df_hfi.loc["2000":]['Distressed Securities'] / 100
    print(r_dist)

    # Question 1: Gaussian parametric VaR of Distressed Securities at 1% level
    print("Question 1: {}".format(erk.var_gaussian(r_dist, level=1)))
    # Question 2: With Corner-Fisher adjustment
    print("Question 2: {}".format(erk.var_gaussian(r_dist, level=1, modified=True)))
    # Question 3: Historic Var
    print("Question 3: {}".format(erk.var_historic(r_dist, level=1)))

    ind = get_ind_returns()
    l = ["Books", "Steel", "Oil", "Mines"]
    ind_13_17 = ind.loc["2013":"2017"][l]
    cov = ind_13_17.cov()
    rf = 0.1

    # Question 4: Weight of Steel in EW port? 0.25
    # Question 5-7: Weight of largest component in MSR port
    er = erk.annualize_rets(ind_13_17, 12)
    msr_wts = erk.msr(rf, er, cov)
    print("MSR Weights: {}".format(msr_wts))

    #Question 8-10 GMV:
    gmv_wts = erk.gmv(cov)
    print("GMV Weights: {}".format(gmv_wts))

    # Questions 11/12
    cov_2018 = ind.loc["2018"][l].cov()
    print("MSR Vol 2018: {}".format(erk.port_vol(msr_wts, cov_2018) * (12 ** (1 / 2))))
    print("GMV Vol 2018: {}".format(erk.port_vol(gmv_wts, cov_2018) * (12 ** (1 / 2))))
