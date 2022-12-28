import numpy as np
import seaborn
import matplotlib.pyplot as plt



def linregfit(X: np.matrix, Y: np.matrix ) -> np.matrix:
    """ Fit a linear regression """
    X_sqrd = np.matmul(np.transpose(X), X)
    print(X_sqrd)
    X_sqrd_inv = np.linalg.inv(X_sqrd)
    X_dot_Y = np.matmul(np.transpose(X), Y)
    return np.matmul(X_sqrd_inv, X_dot_Y)

if __name__ == '__main__':

    X = np.transpose(np.array([np.ones(10), np.random.uniform(0, 10, size=10), np.random.uniform(0, 20, size=10)]))
    real_B = np.array([100, 2, 3]).transpose()
    noise = np.random.normal(loc=0, scale=10, size=X.shape[0])

    Y = np.matmul(X, real_B) + noise

    print(X)
    print(Y)
    est_beta = linregfit(X, Y)

    print(est_beta)
    ax = plt.axes(projection='3d')
    sim_x = np.linspace(0, 10, 250)
    sim_y = np.linspace(0, 20, 250)
    ax.plot3D(sim_x, sim_y, sim_x * est_beta[1] + sim_y * est_beta[2] + est_beta[0])
    ax.scatter3D(X[:, 1], X[:, 2], Y)
    plt.show()