import edhec_risk_kit as erk
import pandas as pd

if __name__ == '__main__':

    # 3 Bonds
    # B1 - 15Y - Face 1000 - 5% semi-annual cpn
    # B2 - 5Y - Face 1000 - 6% quarterly cpn
    # B3 - 10Y - Face 1000 - ZC
    # Yield curve flat 5%

    p1 = erk.bond_price(15, 1000, coupon_rate=0.05, coupons_per_year=2, discount_rate=0.05)
    p2 = erk.bond_price(5, 1000, coupon_rate=0.06, coupons_per_year=4, discount_rate=0.05)
    p3 = erk.bond_price(10, 1000, coupon_rate=0, coupons_per_year=1, discount_rate=0.05)
    print("Prices ({}, {}, {})".format(p1, p2, p3))

    md1 = erk.macaulay_duration(erk.bond_cash_flows(15, 1000, 0.05, 2, index_by_year=True), discount_rate=0.05)
    md2 = erk.macaulay_duration(erk.bond_cash_flows(5, 1000, 0.06, 4, index_by_year=True), discount_rate=0.05)
    md3 = erk.macaulay_duration(erk.bond_cash_flows(10, 1000, 0, index_by_year=True), discount_rate=0.05)
    print("Mac Dur ({}, {}, {})".format(md1, md2, md3))

    liab = pd.Series([1, 2, 3], index=[3, 5, 10]) * 100000
    ld = erk.macaulay_duration(liab, discount_rate=0.05)
    print(ld)

    print("w2: {}".format((ld - md1) / (md2 - md1)))
    print("w2b: {}".format((ld - md3) / (md2 - md3)))