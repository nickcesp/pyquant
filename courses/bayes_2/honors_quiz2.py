from courses.bayes_2.rnd_walk_met_hastings import rw_met_hastings
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    y = (-0.2, -1.5, -5.3, 0.3, -0.8, -2.2)
    n = len(y)
    y_bar = np.mean(y)
    rdn = np.random.default_rng()
    fig, axes = plt.subplots(4, 1)
    i = 0
    for sd in (0.5, 1.5, 3, 4):
        post, accpt_rate = rw_met_hastings(n=n,
                                           y_bar=y_bar,
                                           n_iter=10000,
                                           theta_0=0,
                                           cand_sampling_func=rdn.normal,
                                           cand_sampling_kwargs={'scale': sd})
        axes[i].plot(post)
        i += 1
        print("Acceptance rate for {}: {}".format(sd, accpt_rate))
        print("mean {}".format(np.mean(post)))

    plt.show()