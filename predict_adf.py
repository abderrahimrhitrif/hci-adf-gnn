"""Inference and Uncertainty Quantification Script for ADF-GNN.

This script implements dual Uncertainty Quantification (UQ) for power system state estimation:
    1. Aleatoric Uncertainty (Data Noise): Propagated analytically through ADF moment-matching.
    2. Epistemic Uncertainty (Model Uncertainty): Estimated via T Monte Carlo Dropout forward passes.
    3. Total Predictive Uncertainty: var_total = var_aleatoric + var_epistemic.

Usage:
    python predict_adf.py --model_path best_model_adf.pth --raw_dir IEEE30 --num_mc_samples 30
"""

import argparse
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from create_graph import PowerGridDataset
from model_adf import ADFPowerGridGNN


def predict_with_adf(
    model: ADFPowerGridGNN,
    batch: torch.nn.Module,
    num_mc_samples: int = 30,
    v_init: float = 1e-5,
    var_factor_override: torch.Tensor = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Performs post-hoc ADF prediction with Monte Carlo Dropout uncertainty fusion.

    Args:
        model (ADFPowerGridGNN): Trained ADF model instance.
        batch (HeteroData): Single input graph batch.
        num_mc_samples (int): Number of Monte Carlo Dropout forward passes (T).
        v_init (float): Baseline measurement noise variance (aleatoric input).
        device (torch.device): PyTorch compute device.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing:
            - 'mu': Mean prediction tensor.
            - 'var_aleatoric': Propagated aleatoric variance.
            - 'var_epistemic': Model epistemic variance.
            - 'var_total': Total predictive variance.
            - 'target': Ground truth target state.
    """
    model.eval()
    # Keep dropout active during inference for epistemic uncertainty sampling
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    batch = batch.to(device)

    # Construct input Gaussian distributions
    mu_dict = {"variable": batch["variable"].x, "factor": batch["factor"].x}

    if var_factor_override is not None:
        var_dict = {
            "variable": torch.full_like(batch["variable"].x, v_init),
            "factor": var_factor_override,
        }
    else:
        var_dict = {
            "variable": torch.full_like(batch["variable"].x, v_init),
            "factor": torch.full_like(batch["factor"].x, v_init),
        }

    means_list = []
    vars_list = []

    with torch.no_grad():
        for _ in range(num_mc_samples):
            mu_out, var_out = model(mu_dict, var_dict, batch.edge_index_dict)
            means_list.append(mu_out)
            vars_list.append(var_out)

    means_t = torch.stack(means_list)  # (T, num_nodes, 1)
    vars_t = torch.stack(vars_list)  # (T, num_nodes, 1)

    # 1. Average propagated aleatoric variance
    avg_aleatoric = vars_t.mean(dim=0)

    # 2. Epistemic variance (variance of Monte Carlo mean predictions)
    total_mu = means_t.mean(dim=0)
    epistemic = means_t.var(dim=0)

    # 3. Total predictive variance
    total_var = avg_aleatoric + epistemic

    return {
        "mu": total_mu,
        "var_aleatoric": avg_aleatoric,
        "var_epistemic": epistemic,
        "var_total": total_var,
        "target": batch["variable"].y.view(-1, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADF-GNN Inference and Uncertainty Quantification."
    )
    parser.add_argument("--root", type=str, default="./", help="Root directory.")
    parser.add_argument(
        "--raw_dir", type=str, default="IEEE30", help="Raw dataset directory."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model_adf.pth",
        help="Path to ADF weights.",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=32, help="Hidden embedding dimension."
    )
    parser.add_argument(
        "--num_mc_samples", type=int, default=30, help="Monte Carlo Dropout samples."
    )
    parser.add_argument(
        "--v_init", type=float, default=1e-5, help="Baseline input variance."
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="data/test/adf_uncertainty_sample.png",
        help="Output plot path.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ADF-GNN] Running inference & UQ on device: {device}")

    # 1. Load Dataset
    print(f"Loading '{args.raw_dir}' test dataset...")
    dataset = PowerGridDataset(
        root=args.root, dataset_type="test", raw_data_dir=args.raw_dir
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    sample = dataset[0]
    var_in_dim = sample["variable"].x.shape[1]
    factor_in_dim = sample["factor"].x.shape[1]

    # 2. Initialize Model
    model = ADFPowerGridGNN(
        var_in_dim,
        factor_in_dim,
        hidden_dim=args.hidden_dim,
        readout_units=16,
        output_units=1,
    ).to(device)

    # 3. Load Trained Weights
    if os.path.exists(args.model_path):
        print(f"Loading trained weights from {args.model_path}...")
        weights = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(weights, strict=False)
    else:
        print(
            f"Warning: Model checkpoint '{args.model_path}' not found. Using initialized weights."
        )

    # 4. Evaluate First Few Test Samples
    results = []
    print("\n--- ADF-GNN Uncertainty Quantification Results ---")
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        res = predict_with_adf(
            model,
            batch,
            num_mc_samples=args.num_mc_samples,
            v_init=args.v_init,
            device=device,
        )
        results.append(res)

        err = torch.abs(res["mu"] - res["target"]).mean().item()
        ale = res["var_aleatoric"].mean().item()
        epi = res["var_epistemic"].mean().item()
        tot = res["var_total"].mean().item()
        print(
            f"Sample {i:02d} | Abs Error: {err:.6f} | Aleatoric Var: {ale:.2e} | Epistemic Var: {epi:.2e} | Total Var: {tot:.2e}"
        )

    # 5. Plot Sample 0 Prediction with 95% Confidence Bounds
    res0 = results[0]
    mu_arr = res0["mu"].cpu().numpy().flatten()
    target_arr = res0["target"].cpu().numpy().flatten()
    std_arr = torch.sqrt(res0["var_total"]).cpu().numpy().flatten()

    indices = np.arange(len(mu_arr))
    os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(
        indices, target_arr, "k-", label="Ground Truth State", alpha=0.7, linewidth=1.5
    )
    plt.plot(indices, mu_arr, "g--", label="ADF Mean Prediction", linewidth=1.5)
    plt.fill_between(
        indices,
        mu_arr - 2 * std_arr,
        mu_arr + 2 * std_arr,
        color="green",
        alpha=0.2,
        label="95% Confidence Interval (ADF Total)",
    )
    plt.title(
        f"ADF-GNN: Post-hoc Uncertainty Quantification (Sample 0, MC Samples={args.num_mc_samples})"
    )
    plt.xlabel("Node Index")
    plt.ylabel("Voltage State")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(args.output_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved uncertainty plot to: {args.output_plot}")


if __name__ == "__main__":
    main()
