"""Signal-to-Noise Ratio (SNR) and Gaussian Noise Injection Utilities.

This module provides helper functions for estimating feature signal power across power grid datasets
and corrupting dataset measurements with controlled additive Gaussian noise for robustness testing.
"""

from typing import List
import copy
import numpy as np
import torch
from torch_geometric.data import HeteroData


def calculate_signal_power(
    dataset: List[HeteroData],
    node_type: str = 'factor',
    feature_idx: int = 0
) -> float:
    """Calculates the empirical variance (signal power) of a target feature across a graph dataset.

    Args:
        dataset (List[HeteroData]): List or dataset object containing PyG HeteroData graphs.
        node_type (str): Target node type ('factor' or 'variable').
        feature_idx (int): Feature index within the node feature matrix.

    Returns:
        float: Empirical variance of the specified feature.
    """
    all_features = [data[node_type].x[:, feature_idx] for data in dataset]
    concatenated = torch.cat(all_features)
    return torch.var(concatenated).item()


def add_noise_to_dataset(
    dataset: List[HeteroData],
    snr_db: float,
    signal_power: float,
    node_type: str = 'factor',
    feature_idx: int = 0
) -> List[HeteroData]:
    """Injects zero-mean Gaussian noise into a target node feature based on the desired SNR in dB.

    Noise standard deviation formula:
        sigma_noise = sqrt(signal_power / 10^(SNR_dB / 10))

    Args:
        dataset (List[HeteroData]): Input dataset containing PyG HeteroData graphs.
        snr_db (float): Desired Signal-to-Noise Ratio in decibels (dB).
        signal_power (float): Signal variance power of the target feature.
        node_type (str): Target node type ('factor' or 'variable').
        feature_idx (int): Target feature index to corrupt.

    Returns:
        List[HeteroData]: New list of HeteroData graphs with corrupted target feature.
    """
    sigma_noise = np.sqrt(signal_power / (10 ** (snr_db / 10.0)))

    noisy_dataset = []
    for data in dataset:
        # Clone feature matrix being modified
        noisy_x = data[node_type].x.clone()
        noisy_x[:, feature_idx] += torch.randn(noisy_x.shape[0], device=noisy_x.device) * sigma_noise

        # Create shallow copy of graph container and assign modified feature matrix
        noisy_data = copy.copy(data)
        noisy_data[node_type].x = noisy_x
        noisy_dataset.append(noisy_data)

    return noisy_dataset
