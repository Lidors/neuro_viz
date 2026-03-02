"""
analysis.py  –  Neuroscience data analysis functions.

This module collects analysis routines that operate on PSTH data returned by
SessionData.  Import from here and call from data_loader.py or psth_panel.py.
"""

import numpy as np
from scipy.signal import correlate

# Maximum lag window used by block_best_lag (seconds)
_MAX_LAG_SEC = 1.0


def block_amplitude(
    block_mat: np.ndarray,
    mean_psth: np.ndarray,
) -> np.ndarray:
    """Compute a normalised per-block amplitude proxy.

    Mimics the MATLAB pattern::

        Mat = Ma - mean(Ma);          % zero-mean the mean PSTH
        At  = P1 - mean(P1, 2);       % zero-mean each block (subtract its own mean)
        v   = At * Mat' / (Mat * Mat');  % normalised dot product

    Steps
    -----
    1. Subtract the temporal mean from the mean PSTH  →  ``Mat``
    2. Subtract each block's own temporal mean         →  ``At``
    3. Dot-product each zero-meaned block with ``Mat``
    4. Divide by ``||Mat||²`` (sum of squares of Mat)  →  normalised amplitude

    Dividing by ``||Mat||²`` makes the result dimensionless:
    a block identical to the mean returns 1.0, a block with twice the
    modulation depth returns ~2.0, a silent block returns ~0.

    Parameters
    ----------
    block_mat : (n_time, n_blocks)
        Block-averaged PSTHs (already smoothed if desired).
    mean_psth : (n_time,)
        Mean PSTH across all trials on the same time axis.

    Returns
    -------
    amplitude : (n_blocks,)
        Normalised amplitude proxy.
        1.0 = block modulation matches the mean; >1 = enhanced; <1 = reduced.
    """
    # Step 1 – zero-mean the mean PSTH
    Mat = mean_psth - np.mean(mean_psth)                          # (n_time,)

    # Step 2 – subtract each block's own temporal mean
    At = block_mat - np.mean(block_mat, axis=0, keepdims=True)   # (n_time, n_blocks)

    # Step 3 – dot product then normalise by ||Mat||²
    raw = At.T @ Mat                                              # (n_blocks,)
    norm = float(np.dot(Mat, Mat))                                # ||Mat||²
    if norm > 0:
        raw = raw / norm

    return raw


def block_best_lag(
    block_mat: np.ndarray,
    mean_psth: np.ndarray,
    time_ax: np.ndarray,
) -> np.ndarray:
    """Find the temporal lag at which each block PSTH best matches the mean.

    Uses cross-correlation of the zero-meaned signals (same DC removal as
    ``block_amplitude``) to isolate the temporal structure.  The search is
    constrained to ±1 s (or the full PSTH window if shorter).

    Parameters
    ----------
    block_mat : (n_time, n_blocks)
        Block-averaged PSTHs (already smoothed if desired).
    mean_psth : (n_time,)
        Mean PSTH across all trials on the same time axis.
    time_ax : (n_time,)
        Time axis in seconds.

    Returns
    -------
    best_lag_ms : (n_blocks,)
        Lag in milliseconds at which cross-correlation is maximised.
        Positive → block response is delayed relative to the mean.
        Negative → block response is earlier than the mean.
    """
    n_time, n_blocks = block_mat.shape
    dt = float(time_ax[1] - time_ax[0])                          # s / bin

    # Lag window: ±1 s clipped to what the data support
    max_lag_samples = int(round(min(_MAX_LAG_SEC, (n_time - 1) * dt) / dt))
    lags = np.arange(-(n_time - 1), n_time)                      # (2*n_time-1,)
    window_mask = np.abs(lags) <= max_lag_samples

    # Zero-mean both signals (same DC removal as block_amplitude)
    Mat = mean_psth - np.mean(mean_psth)                          # (n_time,)
    At  = block_mat - np.mean(block_mat, axis=0, keepdims=True)  # (n_time, n_blocks)

    best_lag_ms = np.empty(n_blocks)
    for j in range(n_blocks):
        xc     = correlate(At[:, j], Mat, mode='full')
        xc_w   = xc[window_mask]
        lags_w = lags[window_mask]
        idx    = int(np.argmax(xc_w))
        best_lag_ms[j] = float(lags_w[idx]) * dt * 1000.0        # samples → ms

    return best_lag_ms
