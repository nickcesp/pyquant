from matplotlib.pyplot import plot, show
import numpy as np


if __name__ == '__main__':

    # Create some test data
    dx = .01
    X = np.arange(-2, 2, dx)
    Y = np.exp(-X ** 2)  # 1 / e^(x^2)?

    # Normalize the data to a proper PDF
    Y /= (dx * Y).sum()

    # Compute the CDF
    CY = np.cumsum(Y * dx)

    # Plot both
    plot(X, Y)
    plot(X, CY, 'r--')

    show()

