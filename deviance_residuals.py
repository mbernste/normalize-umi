"""
Normalization via binomial deviance residuals as described in Townes et al.
https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1861-6
"""

import numpy as np

def normalize(X):
    """
    Parameters
    ----------
    X
        MxG matrix of raw counts where M is the number of samples
        and G is the number of genes

    Returns
    -------
    A MxG matrix of normalized expression values

    """
    # Total counts. This is an MxG matrix where each row repeats
    # the total counts for the given row's sample.
    N = np.full(
        (X.shape[1], X.shape[0]), 
        np.sum(X, axis=1)
    ).T

    # Means. This is a G-length vector of gene means.
    M = np.mean(X, axis=0)

    # Compute residual deviances
    R = np.sign(X - M) * np.sqrt(
        np.nan_to_num(2*X*np.log(X/M), copy=True, nan=0.0) \
        + 2*(N-X)*np.log((N-X)/(N-M))
    )

    return R
