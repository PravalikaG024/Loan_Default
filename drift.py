import numpy as np
from scipy.stats import ks_2samp

# -----------------------------
# PSI (Population Stability Index)
# -----------------------------
def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.linspace(0, 1, buckets + 1)

    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi = np.sum(
        (expected_percents - actual_percents) *
        np.log((expected_percents + 1e-6) / (actual_percents + 1e-6))
    )
    return psi


# -----------------------------
# KS (Kolmogorov-Smirnov)
# -----------------------------
def calculate_ks(expected, actual):
    ks_stat, _ = ks_2samp(expected, actual)
    return ks_stat


# -----------------------------
# CSI (Characteristic Stability Index)
# -----------------------------
def calculate_csi(expected, actual):
    return np.mean(np.abs(expected - actual))