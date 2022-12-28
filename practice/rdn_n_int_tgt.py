"""
9.10:  Given a target number generate a random sample of n integers that sum to that target
that are also within sigma standard deviations of the mean
"""

def generate_numbers(tgt, n, sigma):
    """ Solution """
    mean = tgt / n
    sd = int(sigma * mean)

    max_val, min_val = mean + sd, mean - sd

