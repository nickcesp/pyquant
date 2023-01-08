from scipy.stats import norm
import numpy as np

if __name__ == '__main__':

    rand = np.random.default_rng()
    theta_sampl = rand.beta(5, 3, 100000)
    theta_bar = np.mean(theta_sampl)

    print('Q5: Odds: {}'.format(np.mean(theta_sampl / (1 - theta_sampl))))
    #print('Q5: Given: {}'.format(0.625 / (1-0.625)))

    print("Q6: Better than 50/50 odds: {}".format(np.sum(theta_sampl > 0.5) / len(theta_sampl)))

    norm_sample = np.sort(rand.normal(0, 1, 10000))
    print('Q7: 0.3 quantile of norm dist: {}'.format(norm_sample[2999]))
    print('Q7 check: {}'.format(norm.ppf(0.3, 0, 1)))

    print('Q8: {}'.format(np.sqrt(5.2 / 5000)))

    print("honors quiz")
    arr = np.array([[0, 1], [0.3, 0.7]])
    for i in range(40):
        arr = np.matmul(arr, arr)

    print(arr)