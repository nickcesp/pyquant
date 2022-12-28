from datetime import date

class Cds:

    @staticmethod
    def upf_pymt(notional : float, trd_price: float) -> float:
        """ Computes the upfront payment of a CDS trade

            :return: Upfront payment value.
                     Positive value indicates that the protection buyer pays amount
                     Negative value indicates that the protection buyer receives amount
        """
        return notional * (100 - trd_price) / 100

    @staticmethod
    def accr_int(asof: date, accr_start_dt: date, notional: float, cpn: float) -> float:
        """ Computes the accured interest of the CDS contract on any given day

            :param asof: Analysis date
            :param accr_start_dt: Previous accrual date
            :param notional: Notional face of CDS contract
            :param cpn: Coupon

            :return: Value of accured interest on analysis date
        """
        return notional * cpn * (asof - accr_start_dt).days / 360

