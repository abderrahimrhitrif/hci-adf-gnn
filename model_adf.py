"""Assumed Density Filtering Heterogeneous Graph Neural Network (ADF-GNN).

This module implements an ADF-aware Graph Neural Network for power grid state estimation.
It propagates both Gaussian mean (mu) and variance (var) analytically through graph convolutions,
linear transformations, and non-linear activations.

Theoretical Basis:
    - Linear layers: Var[W * X] = (W^2) * Var[X]
    - Graph Attention: Attention weights alpha scale mean as alpha * mu, and variance as alpha^2 * var.
    - ReLU activations: Rectified Gaussian analytical moment matching.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from adf_core import adf_linear, adf_relu

# =============================================================================
# 1. ADF BASE LAYERS
# =============================================================================


class ADFLinear(nn.Module):
    """Linear layer operating on Gaussian mean and variance inputs."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)

    def forward(
        self, mu: torch.Tensor, var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagates mean and variance through the linear weight matrix.

        Args:
            mu (torch.Tensor): Input mean tensor.
            var (torch.Tensor): Input variance tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Propagated mean and variance tensors.
        """
        return adf_linear(mu, var, self.lin.weight, self.lin.bias)


class ADFReLU(nn.Module):
    """ReLU activation operating on Gaussian mean and variance via moment matching."""

    def forward(
        self, mu: torch.Tensor, var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagates mean and variance through Rectified Gaussian moment matching.

        Args:
            mu (torch.Tensor): Input mean tensor.
            var (torch.Tensor): Input variance tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rectified Gaussian mean and variance tensors.
        """
        return adf_relu(mu, var)


# =============================================================================
# 2. ADF GRAPH ATTENTION CONVOLUTION (ADFGATConv)
# =============================================================================


class ADFGATConv(MessagePassing):
    """ADF-aware Graph Attention Convolutional Layer.

    Propagates mean and variance through attention-weighted graph edges:
        - Mean message: m_mu = alpha * mu_j
        - Variance message: m_var = (alpha^2) * var_j

    Args:
        in_channels (int): Input feature dimension.
        out_channels (int): Output feature dimension per head.
        heads (int): Number of attention heads (default: 1).
        negative_slope (float): LeakyReLU negative slope for attention score computation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__(aggr="sum", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope

        self.lin_mu = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes linear weights and attention vector with Xavier uniform distribution."""
        nn.init.xavier_uniform_(self.lin_mu.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(
        self,
        mu: Tuple[torch.Tensor, torch.Tensor],
        var: Tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs ADF graph convolution forward pass.

        Args:
            mu (Tuple[torch.Tensor, torch.Tensor]): Tuple of (mu_source, mu_target).
            var (Tuple[torch.Tensor, torch.Tensor]): Tuple of (var_source, var_target).
            edge_index (torch.Tensor): Edge indices of shape (2, num_edges).
            size (Optional[Tuple[int, int]]): Tuple of (num_source_nodes, num_target_nodes).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Aggregated (mu_out, var_out) for target nodes.
        """
        mu_src, mu_tgt = mu
        var_src, var_tgt = var

        return self.propagate(
            edge_index, mu=(mu_src, mu_tgt), var=(var_src, var_tgt), size=size
        )

    def message(
        self,
        mu_i: torch.Tensor,
        mu_j: torch.Tensor,
        var_j: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes message payloads for graph edges.

        Args:
            mu_i (torch.Tensor): Target node mean tensor.
            mu_j (torch.Tensor): Source node mean tensor.
            var_j (torch.Tensor): Source node variance tensor.
            index (torch.Tensor): Edge target indices.
            ptr (Optional[torch.Tensor]): CSR graph pointer.
            size_i (Optional[int]): Number of target nodes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Weighted mean message and squared-weight variance message.
        """
        # 1. Compute Softmax Attention Coefficients (alpha)
        x_cat = torch.cat([mu_i, mu_j], dim=-1)
        alpha = (x_cat * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)

        # Saving alpha for the dashboard
        self._alpha = alpha.detach()

        # 2. Weighted Propagation (Mean scales with alpha, Variance scales with alpha^2)
        weighted_mu = mu_j * alpha.unsqueeze(-1)
        weighted_var = var_j * (alpha**2).unsqueeze(-1)

        return weighted_mu, weighted_var

    def aggregate(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregates messages using sum aggregation.

        Args:
            inputs (Tuple[torch.Tensor, torch.Tensor]): Tuple of (weighted_mu, weighted_var).
            index (torch.Tensor): Target node indices.
            ptr (Optional[torch.Tensor]): CSR graph pointer.
            dim_size (Optional[int]): Number of target nodes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Summed mean and variance aggregations.
        """
        mu_j, var_j = inputs
        mu_agg = super().aggregate(mu_j, index, ptr, dim_size)
        var_agg = super().aggregate(var_j, index, ptr, dim_size)
        return mu_agg, var_agg

    def update(
        self, aggr_out: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Averages multi-head attention outputs.

        Args:
            aggr_out (Tuple[torch.Tensor, torch.Tensor]): Tuple of aggregated (mu_agg, var_agg).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Head-averaged (mu_out, var_out).
        """
        mu_agg, var_agg = aggr_out
        return mu_agg.mean(dim=1), var_agg.mean(dim=1)


# =============================================================================
# 3. ADF HETEROGENEOUS POWER GRID GNN
# =============================================================================


class ADFPowerGridGNN(nn.Module):
    """Assumed Density Filtering Heterogeneous Power Grid GNN.

    Propagates Gaussian means and variances across factor and variable nodes
    in power system factor graphs for probabilistic state estimation.

    Args:
        var_in_dim (int): Feature dimension for variable nodes.
        factor_in_dim (int): Feature dimension for factor nodes.
        hidden_dim (int): Dimensionality of hidden node embeddings (default: 32).
        readout_units (int): Dimensionality of hidden readout layer (default: 16).
        output_units (int): Output prediction dimension (default: 1).
        num_iterations (int): Number of bipartite message passing iterations (default: 3).
        dropout_rate (float): Dropout probability for epistemic uncertainty estimation (default: 0.15).
    """

    def __init__(
        self,
        var_in_dim: int,
        factor_in_dim: int,
        hidden_dim: int = 32,
        readout_units: int = 16,
        output_units: int = 1,
        num_iterations: int = 3,
        dropout_rate: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_iterations = num_iterations

        # State Encoders
        self.var_encoder = ADFLinear(var_in_dim, hidden_dim)
        self.factor_encoder = ADFLinear(factor_in_dim, hidden_dim)

        # Message Passing Convolutions
        self.stage1_convs = nn.ModuleList(
            [ADFGATConv(hidden_dim, hidden_dim) for _ in range(num_iterations)]
        )
        self.stage2_convs = nn.ModuleList(
            [ADFGATConv(hidden_dim, hidden_dim) for _ in range(num_iterations)]
        )

        # Node Update Neural Networks
        self.update_nns = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "variable": nn.Sequential(
                            ADFLinear(hidden_dim, hidden_dim), ADFReLU()
                        ),
                        "factor": nn.Sequential(
                            ADFLinear(hidden_dim, hidden_dim), ADFReLU()
                        ),
                    }
                )
                for _ in range(num_iterations)
            ]
        )

        # Readout Model
        self.readout_model = nn.Sequential(
            ADFLinear(hidden_dim, readout_units),
            ADFReLU(),
            ADFLinear(readout_units, output_units),
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        mu_dict: Dict[str, torch.Tensor],
        var_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes ADF forward propagation to compute predicted mean and variance.

        Args:
            mu_dict (Dict[str, torch.Tensor]): Dict of node mean feature tensors ('variable', 'factor').
            var_dict (Dict[str, torch.Tensor]): Dict of node variance feature tensors ('variable', 'factor').
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): Dict of edge indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - final_mu (torch.Tensor): Predicted mean state vector.
                - final_var (torch.Tensor): Propagated aleatoric variance vector.
        """
        # Step 1: Initial encoding of mean and variance
        mu_v, var_v = self.var_encoder(mu_dict["variable"], var_dict["variable"])
        mu_f, var_f = self.factor_encoder(mu_dict["factor"], var_dict["factor"])

        num_vars = mu_v.size(0)
        num_factors = mu_f.size(0)

        # Step 2: Multi-stage bipartite message passing
        for i in range(self.num_iterations):
            # STAGE 1: Factor -> Variable propagation
            edge_fv = edge_index_dict[("factor", "to", "variable")]
            m_f = (
                self.stage1_convs[i]
                .lin_mu(mu_f)
                .view(-1, self.stage1_convs[i].heads, self.stage1_convs[i].out_channels)
            )
            v_f = F.linear(var_f, self.stage1_convs[i].lin_mu.weight ** 2).view(
                -1, self.stage1_convs[i].heads, self.stage1_convs[i].out_channels
            )
            m_v_target = (
                self.stage1_convs[i]
                .lin_mu(mu_v)
                .view(-1, self.stage1_convs[i].heads, self.stage1_convs[i].out_channels)
            )
            v_v_target = F.linear(var_v, self.stage1_convs[i].lin_mu.weight ** 2).view(
                -1, self.stage1_convs[i].heads, self.stage1_convs[i].out_channels
            )

            m_mu, m_var = self.stage1_convs[i](
                (m_f, m_v_target),
                (v_f, v_v_target),
                edge_fv,
                size=(num_factors, num_vars),
            )
            up_mu, up_var = self.update_nns[i]["variable"][0](m_mu, m_var)
            up_mu, up_var = self.update_nns[i]["variable"][1](up_mu, up_var)
            mu_v = mu_v + up_mu
            var_v = var_v + up_var

            # STAGE 2: Variable -> Factor propagation
            edge_vf = edge_index_dict[("variable", "to", "factor")]
            m_v = (
                self.stage2_convs[i]
                .lin_mu(mu_v)
                .view(-1, self.stage2_convs[i].heads, self.stage2_convs[i].out_channels)
            )
            v_v = F.linear(var_v, self.stage2_convs[i].lin_mu.weight ** 2).view(
                -1, self.stage2_convs[i].heads, self.stage2_convs[i].out_channels
            )
            m_f_target = (
                self.stage2_convs[i]
                .lin_mu(mu_f)
                .view(-1, self.stage2_convs[i].heads, self.stage2_convs[i].out_channels)
            )
            v_f_target = F.linear(var_f, self.stage2_convs[i].lin_mu.weight ** 2).view(
                -1, self.stage2_convs[i].heads, self.stage2_convs[i].out_channels
            )

            m_mu, m_var = self.stage2_convs[i](
                (m_v, m_f_target),
                (v_v, v_f_target),
                edge_vf,
                size=(num_vars, num_factors),
            )
            up_mu, up_var = self.update_nns[i]["factor"][0](m_mu, m_var)
            up_mu, up_var = self.update_nns[i]["factor"][1](up_mu, up_var)
            mu_f = mu_f + up_mu
            var_f = var_f + up_var

        # Step 3: Readout layer propagation
        final_mu = self.dropout(mu_v)
        final_var = var_v
        for layer in self.readout_model:
            if isinstance(layer, (ADFLinear, ADFReLU)):
                final_mu, final_var = layer(final_mu, final_var)

        return final_mu, final_var

    def get_attention_weights(self) -> list:
        attentions = []
        for conv in self.stage1_convs + self.stage2_convs:
            if hasattr(conv, "_alpha"):
                attentions.append(conv._alpha)
        return attentions
