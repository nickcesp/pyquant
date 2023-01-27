import numpy as np
import pandas as pd
from numpy.linalg import inv

def as_colvec(x):
    if (x.ndim == 2):
        return x
    else:
        return np.expand_dims(x, axis=1)

def implied_returns(delta, sigma, w):
    """
        Obtain the implied expected returns by reverse engineering the weights
        param delta: Risk Aversion Coefficient (scalar)
        param sigma: Variance-Covariance Matrix (N x N) as DataFrame
        param w: Portfolio weights (N x 1) as Series
        Returns an N x 1 vector of Returns as Series
    """
    ir = delta * sigma.dot(w).squeeze() # to get a series from a 1-column dataframe
    ir.name = 'Implied Returns'
    return ir


# Assumes that Omega is proportional to the variance of the prior
def proportional_prior(sigma, tau, p):
    """
    Returns the He-Litterman simplified Omega
    Inputs:
    sigma: N x N Covariance Matrix as DataFrame
    tau: a scalar
    p: a K x N DataFrame linking Q and Assets
    returns a P x P DataFrame, a Matrix representing Prior Uncertainties
    """
    helit_omega = p.dot(tau * sigma).dot(p.T)
    # Make a diag matrix from the diag elements of Omega
    return pd.DataFrame(np.diag(np.diag(helit_omega.values)),index=p.index, columns=p.index)


def bl(w_prior, sigma_prior, p, q,
                omega=None,
                delta=2.5, tau=.02):
    """
        # Computes the posterior expected returns based on
        # the original black litterman reference model
        #
        # W.prior must be an N x 1 vector of weights, a Series
        # Sigma.prior is an N x N covariance matrix, a DataFrame
        # P must be a K x N matrix linking Q and the Assets, a DataFrame
        # Q must be an K x 1 vector of views, a Series
        # Omega must be a K x K matrix a DataFrame, or None
        # if Omega is None, we assume it is
        #    proportional to variance of the prior
        # delta and tau are scalars
    """
    if omega is None:
        omega = proportional_prior(sigma_prior, tau, p)
    # Force w.prior and Q to be column vectors
    # How many assets do we have?
    N = w_prior.shape[0]
    # And how many views?
    K = q.shape[0]
    # First, reverse-engineer the weights to get pi
    pi = implied_returns(delta, sigma_prior,  w_prior)
    # Adjust (scale) Sigma by the uncertainty scaling factor
    sigma_prior_scaled = tau * sigma_prior
    # posterior estimate of the mean, use the "Master Formula"
    # we use the versions that do not require
    # Omega to be inverted (see previous section)
    # this is easier to read if we use '@' for matrixmult instead of .dot()
    #     mu_bl = pi + sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ (q - p @ pi)
    mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values))
    # posterior estimate of uncertainty of mu.bl
#     sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ p @ sigma_prior_scaled
    sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega)).dot(p).dot(sigma_prior_scaled)
    return (mu_bl, sigma_bl)


# for convenience and readability, define the inverse of a dataframe
def inverse(d):
    """
    Invert the dataframe by inverting the underlying matrix
    """
    return pd.DataFrame(inv(d.values), index=d.columns, columns=d.index)

def w_msr(sigma, mu, scale=True):
    """
    Optimal (Tangent/Max Sharpe Ratio) Portfolio weights
    by using the Markowitz Optimization Procedure
    Mu is the vector of Excess expected Returns
    Sigma must be an N x N matrix as a DataFrame and Mu a column vector as a Series
    This implements page 188 Equation 5.2.28 of
    "The econometrics of financial markets" Campbell, Lo and Mackinlay.
    """
    w = inverse(sigma).dot(mu)
    if scale:
        w = w/sum(w) # fix: this assumes all w is +ve
    return w

def ex1():
    # Consider the portfolio consisting of just two stocks: Intel (INTC) and Pfizer (PFE).
    # Assume that Intel has a market capitalization of approximately USD 80B and that of Pfizer is approximately
    # USD 100B (this is not quite accurate, but works just fine as an example!). Thus, if you held a market-cap weighted
    # portfolio you would hold INTC and PFE with the following weights: 𝑊𝐼𝑁𝑇𝐶=80/180=44%,𝑊𝑃𝐹𝐸=100/180=56%
    # . These appear to be reasonable weights without an extreme allocation to either stock, even though Pfizer is slightly overweighted.
    # We can compute the equilibrium implied returns 𝜋 as follows:
    tickers = ['INTC', 'PFE']
    s = pd.DataFrame([[46.0, 1.06], [1.06, 5.33]], index=tickers, columns=tickers) *  10E-4
    pi = implied_returns(delta=2.5, sigma=s, w=pd.Series([.44, .56], index=tickers))
    # Thus the equilibrium implied returns for INTC are a bit more than 5% and a bit less than 1% for PFE.
    # Assume that the investor thinks that Intel will return 2% and that Pfizer is poised to rebounce, and will return
    # 4% . We can now examine the optimal weights according to the Markowitz procedure.
    # What would happen if we used these expected returns to compute the Optimal Max Sharpe Ratio portfolio?

    # The Max Sharpe Ratio (MSR) Portfolio weights are easily computed in explicit form if there are no constraints
    # on the weights. The weights are given by the expression:
    # 𝑊𝑀𝑆𝑅= (Σ_inv * mu_e) / (one_T * Σ_inv * mu_e)
    # where mu_e is the vector of expected excess returns and Σ is the variance-covariance matrix.

    # Recall that the investor expects that Intel will return 2% and Pfizer will return 4% .
    # We can now examine the optimal weights obtained by naively implementing the Markowitz procedure with these expected returns.
    mu_exp = pd.Series([.02, .04], index=tickers)  # INTC and PFE
    np.round(w_msr(s, mu_exp) * 100, 2)

    # Absolute view 1: INTC will return 2%
    # Absolute view 2: PFE will return 4%
    q = pd.Series({'INTC': 0.02, 'PFE': 0.04})
    # The Pick Matrix
    # For View 2, it is for PFE
    p = pd.DataFrame([
        # For View 1, this is for INTC
        {'INTC': 1, 'PFE': 0},
        # For View 2, it is for PFE
        {'INTC': 0, 'PFE': 1}
    ])

    # Find the Black Litterman Expected Returns
    bl_mu, bl_sigma = bl(w_prior=pd.Series({'INTC': .44, 'PFE': .56}), sigma_prior=s, p=p, q=q)
    # Use the Black Litterman expected returns to get the Optimal Markowitz weights
    opt_markov_wts = w_msr(bl_sigma, bl_mu)
    print(opt_markov_wts)


def ex2():
    tickers = ['INTC', 'PFE']
    s = pd.DataFrame([[46.0, 1.06], [1.06, 5.33]], index=tickers, columns=tickers) * 10E-4
    pi = implied_returns(delta=2.5, sigma=s, w=pd.Series([.44, .56], index=tickers))

    q = pd.Series([
        # Relative View 1: INTC will outperform PFE by 2%
        0.02
    ]
    )
    # The Pick Matrix
    p = pd.DataFrame([
        # For View 1, this is for INTC outperforming PFE
        {'INTC': +1, 'PFE': -1}
    ])

    # Find the Black Litterman Expected Returns
    bl_mu, bl_sigma = bl(w_prior=pd.Series({'INTC': .44, 'PFE': .56}), sigma_prior=s, p=p, q=q)
    opt_markov_wts = w_msr(bl_sigma, bl_mu)
    print(opt_markov_wts)

if __name__ == '__main__':
    ex1()
    ex2()

