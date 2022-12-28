import math


def mean(arr):
    return sum(arr) / len(arr)


def corr(x, y):

    if len(x) != len(y):
        raise ValueError("Must be same length!")

    x_mu = mean(x)
    y_mu = mean(y)

    x_diff = [v - x_mu for v in x]
    y_diff = [v - y_mu for v in y]

    cov_x_y = sum([v_x * v_y for v_x, v_y in zip(x_diff, y_diff)]) / (len(x) - 1)

    s_dev_x = math.sqrt(sum([x_d ** 2 for x_d in x_diff]) / len(x))
    s_dev_y = math.sqrt(sum([y_d ** 2 for y_d in y_diff]) / len(y))

    return cov_x_y / (s_dev_x * s_dev_y)


def sdev(arr):
    m = mean(arr)
    return math.sqrt(sum([(v - m) ** 2 for v in arr]) / len(arr))

