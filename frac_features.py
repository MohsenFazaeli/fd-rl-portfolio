import numpy as np
import pandas as pd


def get_weights(d, lags):
    """
    Return the weights from the series expansion of the differencing operator
    for real orders d and up to lags coefficients.
    """
    weights = [1]
    for k in range(1, lags):
        weights.append(-weights[-1] * ((d - k + 1)) / k)
    return np.array(weights).reshape(-1, 1)

def ts_differencing(series, order, lag_cutoff):
    """
    Return the time series resulting from (fractional) differencing
    for real orders order up to lag_cutoff coefficients.
    """
    weights = get_weights(order, lag_cutoff)
    res = 0
    for k in range(lag_cutoff):
        res += weights[k] * series.shift(k).fillna(0)
    return res[lag_cutoff:]

def cutoff_find(order, cutoff, start_lags):
    """
    Find the number of lags required until the weight of the last term in the series expansion
    of the differencing operator falls below a certain cutoff value.

    Parameters:
    - order: Real order of differencing (d).
    - cutoff: Threshold value below which the weight is considered insignificant.
    - start_lags: Initial amount of lags to start the search from.

    Returns:
    - lags: Number of lags required until the weight falls below the cutoff.
    """
    val = np.inf
    lags = start_lags
    while abs(val) > cutoff:
        w = get_weights(order, lags)
        val = w[-1][-1]  # Retrieve the last element of the last array in w
        lags += 1
    return lags

def ts_differencing_tau(series, order, tau):
    """
    Return the time series resulting from (fractional) differencing
    with a specified cutoff threshold (tau).

    Parameters:
    - series: Input time series.
    - order: Real order of differencing (d).
    - tau: Cutoff threshold value.

    Returns:
    - res: Resulting time series after differencing.
    """
    lag_cutoff = cutoff_find(order, tau, 1)  # Finding lag cutoff with tau
    weights = get_weights(order, lag_cutoff)
    res = 0
    for k in range(lag_cutoff):
        res += weights[k] * series.shift(k).fillna(0)
    return res[lag_cutoff:]
