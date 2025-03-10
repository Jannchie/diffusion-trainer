from typing import overload

import numpy as np
from scipy.special import logit


@overload
def log_normal_pdf(t: float, m: float, s: float) -> float: ...


@overload
def log_normal_pdf(t: np.ndarray, m: float, s: float) -> np.ndarray: ...


def log_normal_pdf(t: float | np.ndarray, m: float, s: float) -> float | np.ndarray:
    """
    Compute the log-normal probability density function as specified.

    Parameters:
    t : float or array-like
        The input value(s), must be between 0 and 1
    m : float
        Location parameter
    s : float
        Scale parameter (standard deviation)

    Returns:
    float or array-like
        The probability density at t
    """
    # Ensure t is within (0, 1)
    if np.any((t <= 0) | (t >= 1)):
        msg = "t must be strictly between 0 and 1"
        raise ValueError(msg)

    # Calculate the logit of t
    logit_t = logit(t)

    # Calculate the density
    numerator = np.exp(-((logit_t - m) ** 2) / (2 * s**2))
    denominator = s * np.sqrt(2 * np.pi) * t * (1 - t)

    return numerator / denominator


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create a range of t values between 0 and 1 (exclusive)
    t_values = np.linspace(0.001, 0.999, 1000)

    # Calculate PDF for different parameters
    pdf_1 = log_normal_pdf(t_values, m=0, s=0.5)
    pdf_2 = log_normal_pdf(t_values, m=0, s=1)
    pdf_3 = log_normal_pdf(t_values, m=0.5, s=1)
    pdf_4 = log_normal_pdf(t_values, m=-0.5, s=1.0)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, pdf_1, label="m=0, s=0.5")
    plt.plot(t_values, pdf_2, label="m=0, s=1")
    plt.plot(t_values, pdf_3, label="m=0.5, s=1")
    plt.plot(t_values, pdf_4, label="m=-0.5, s=1")
    plt.plot()
    plt.title("Log-normal Probability Density Function")
    plt.xlabel("t")
    plt.ylabel("π_ln(t; m, s)")
    plt.xlim(0, 1)
    plt.ylim(0, 3.5)
    plt.legend()
    plt.grid(True)
    plt.savefig("log_normal_pdf_plot.png")
