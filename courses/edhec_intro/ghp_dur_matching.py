import numpy as np
import pandas as pd
import edhec_risk_kit as erk

if __name__ == '__main__':

    # Compute Maculay duration
    cf = erk.bond_cash_flows(maturity=3, principal=1000, coupon_rate=0.06, coupons_per_year=2)
    discounts = erk.discount(cf.index, 0.06 / 2)
    dcf = discounts * cf