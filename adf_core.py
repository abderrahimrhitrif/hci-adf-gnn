"""Assumed Density Filtering (ADF) Core Mathematical Operations.

This module provides primitives for propagating Gaussian probability distributions
represented by their mean (mu) and variance (var) through neural network layers.
Moment matching is used to approximate output distributions after non-linear transformations
as Gaussian distributions.

References:
    - Boyen, X., & Koller, D. (1998). Tractable inference for complex stochastic processes.
    - Gast, J., & Roth, S. (2018). Lightweight Probabilistic Deep Networks. CVPR.
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np


def phi(x: torch.Tensor) -> torch.Tensor:
    """Calculates the probability density function (PDF) of a standard normal distribution.

    Args:
        x (torch.Tensor): Input tensor of standardized values.

    Returns:
        torch.Tensor: Evaluated PDF values matching the shape of x.
    """
    return torch.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def Phi(x: torch.Tensor) -> torch.Tensor:
    """Calculates the cumulative distribution function (CDF) of a standard normal distribution.

    Args:
        x (torch.Tensor): Input tensor of standardized values.

    Returns:
        torch.Tensor: Evaluated CDF values matching the shape of x.
    """
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def adf_linear(
    mu: torch.Tensor,
    var: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Propagates Gaussian mean and variance through an affine linear transformation.

    Given input random variable X ~ N(mu, diag(var)), the output Y = W * X + b has:
        E[Y] = W * mu + b
        Var[Y] = (W^2) * var

    Args:
        mu (torch.Tensor): Input mean tensor of shape (..., in_features).
        var (torch.Tensor): Input variance tensor of shape (..., in_features).
        weight (torch.Tensor): Linear transformation weight matrix of shape (out_features, in_features).
        bias (Optional[torch.Tensor]): Optional bias vector of shape (out_features,).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - mu_out (torch.Tensor): Transformed mean tensor of shape (..., out_features).
            - var_out (torch.Tensor): Transformed variance tensor of shape (..., out_features).
    """
    mu_out = F.linear(mu, weight, bias)
    # Variance propagation uses squared weights (assumes zero cross-covariance between inputs)
    var_out = F.linear(var, weight**2, None)
    return mu_out, var_out


def adf_relu(
    mu: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Propagates mean and variance through a ReLU non-linearity via Rectified Gaussian moment matching.

    For a random variable X ~ N(mu, var), computes the exact mean and variance of Y = max(0, X):
        gamma = mu / sigma
        E[max(0, X)] = mu * Phi(gamma) + sigma * phi(gamma)
        E[max(0, X)^2] = (mu^2 + var) * Phi(gamma) + mu * sigma * phi(gamma)
        Var[max(0, X)] = E[max(0, X)^2] - (E[max(0, X)])^2

    Args:
        mu (torch.Tensor): Input mean tensor.
        var (torch.Tensor): Input variance tensor (must be non-negative).
        eps (float): Numerical stability threshold for standard deviation and variance clamping.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - mu_new (torch.Tensor): Propagated mean tensor.
            - var_new (torch.Tensor): Propagated variance tensor (clamped to min value of eps).
    """
    # Standard deviation with eps clamp for stability
    sigma = torch.sqrt(torch.clamp(var, min=eps))
    
    # Standardized mean ratio
    gamma = mu / sigma
    
    pdf_gamma = phi(gamma)
    cdf_gamma = Phi(gamma)
    
    # First moment (expected value of rectified Gaussian)
    mu_new = mu * cdf_gamma + sigma * pdf_gamma
    
    # Second raw moment E[Y^2]
    e_x2_rect = (mu**2 + var) * cdf_gamma + mu * sigma * pdf_gamma
    
    # Second central moment (Variance) = E[Y^2] - E[Y]^2
    var_new = e_x2_rect - mu_new**2
    
    # Numerical stability clamp to maintain non-negative variance
    var_new = torch.clamp(var_new, min=eps)
    
    return mu_new, var_new
