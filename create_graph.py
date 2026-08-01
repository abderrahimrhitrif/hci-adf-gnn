"""PyTorch Geometric Dataset Loader for Power Grid State Estimation.

This module reads Weighted Least Squares State Estimation (WLS-SE) CSV output files
and constructs PyTorch Geometric `HeteroData` objects representing power grid factor graphs.

Graph Architecture:
    - Variable Nodes: Represent bus voltage states (real/imaginary parts or magnitude/angle).
    - Factor Nodes: Represent physical measurements (line power flows, bus injections, voltage angles).
    - Edges:
        1. Bipartite: ('factor', 'to', 'variable') & ('variable', 'to', 'factor') based on non-zero Jacobian entries.
        2. Variable-to-Variable: ('variable', 'to', 'variable') based on shared measurement factors (H^T * H).
"""

from typing import Tuple, List, Optional
import os
import argparse
from csv import reader
import pandas as pd
import numpy as np
import category_encoders as ce
import torch
from torch_geometric.data import InMemoryDataset, HeteroData
from tqdm import tqdm


def encode_variable_node_indices(num_variable_nodes: int) -> np.ndarray:
    """Encodes variable node integer indices into binary positional feature vectors.

    Args:
        num_variable_nodes (int): Number of variable nodes in the graph.

    Returns:
        np.ndarray: Binary encoded matrix of shape (num_variable_nodes, num_bits).
    """
    df = pd.DataFrame({'nodeIdx': list(range(num_variable_nodes))})
    encoder = ce.BaseNEncoder(cols=['nodeIdx'], return_df=False, base=2)
    encoded_matrix = encoder.fit_transform(df)
    return encoded_matrix


def read_wls_se_files(
    dataset_type: str,
    data_dir: str
) -> Tuple[int, int, int, List[List[str]], List[List[int]], List[List[str]], List[List[str]], List[List[str]]]:
    """Reads WLS-SE raw data files for a specific dataset split (train/validation/test).

    Args:
        dataset_type (str): Dataset split ('train', 'validation', or 'test').
        data_dir (str): Directory path containing the raw CSV and TXT files.

    Returns:
        Tuple containing:
            - num_graphs (int)
            - num_variable_nodes (int)
            - num_measurements (int)
            - estimate_rows (List[List[str]])
            - jacobian_rows (List[List[int]])
            - measurement_rows (List[List[str]])
            - variance_rows (List[List[str]])
            - covariance_rows (List[List[str]])
    """
    prefix_map = {'test': 'Test', 'validation': 'Validation', 'train': 'Training'}
    prefix = prefix_map.get(dataset_type, 'Training')

    # Read Estimates (Ground truth / initial state)
    est_path = os.path.join(data_dir, f"{prefix}_Estimate.csv")
    with open(est_path, 'r') as f:
        estimate_rows = list(reader(f))

    # Read Jacobian binary topology matrix
    jac_path = os.path.join(data_dir, f"{prefix}_Jacobian.csv")
    with open(jac_path, 'r') as f:
        jacobian_rows = [
            [1 if abs(float(val)) > 1e-4 else 0 for val in line]
            for line in reader(f)
        ]

    # Read Measurement values
    meas_path = os.path.join(data_dir, f"{prefix}_Measurement.csv")
    with open(meas_path, 'r') as f:
        measurement_rows = list(reader(f))

    # Read Variance values
    var_path = os.path.join(data_dir, f"{prefix}_Variance.csv")
    with open(var_path, 'r') as f:
        variance_rows = list(reader(f))

    # Read Covariance values
    cov_path = os.path.join(data_dir, f"{prefix}_Covariance.csv")
    with open(cov_path, 'r') as f:
        covariance_rows = list(reader(f))

    # Read Network dimensions metadata
    num_meas_path = os.path.join(data_dir, f"{prefix}_NumMeasurements.txt")
    with open(num_meas_path, 'r') as f:
        lines = f.readlines()
        num_line_p = int(lines[0])
        num_p_inj = int(lines[1])
        num_theta = int(lines[2])
        num_graphs = int(lines[3])
        num_buses = int(lines[4])
        num_variable_nodes = num_buses * 2  # Vreal and Vimag (or Vmag and Vangle)
        num_measurements = num_line_p + num_p_inj + num_theta

    return (
        num_graphs,
        num_variable_nodes,
        num_measurements,
        estimate_rows,
        jacobian_rows,
        measurement_rows,
        variance_rows,
        covariance_rows
    )


class PowerGridDataset(InMemoryDataset):
    """PyTorch Geometric InMemoryDataset for Power Grid State Estimation graphs.

    Args:
        root (str): Root directory where processed graph binary files are stored.
        dataset_type (str): Split type ('train', 'validation', 'test').
        raw_data_dir (str): Subdirectory containing the raw power grid CSV files.
        transform (Optional[callable]): Dynamic graph transformation callback.
        pre_transform (Optional[callable]): Pre-processing graph transformation callback.
    """

    def __init__(
        self,
        root: str = './',
        dataset_type: str = 'train',
        raw_data_dir: str = 'IEEE30',
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None
    ) -> None:
        self.dataset_type = dataset_type
        self._raw_data_dir = os.path.join(root, raw_data_dir)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self) -> str:
        return self._raw_data_dir

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.dataset_type}_data.pt']

    def process(self) -> None:
        """Processes raw CSV data files into PyG HeteroData graph structures."""
        print(f"Loading {self.dataset_type} CSV data from {self.raw_dir}...")
        (
            num_graphs,
            num_variable_nodes,
            num_measurements,
            estimate_rows,
            jacobian_rows,
            measurement_rows,
            variance_rows,
            covariance_rows
        ) = read_wls_se_files(self.dataset_type, self.raw_dir)

        flat_start = torch.ones((num_variable_nodes, 1), dtype=torch.float)
        encoded_indices = torch.tensor(
            encode_variable_node_indices(num_variable_nodes),
            dtype=torch.float
        )

        data_list = []
        print(f"Vectorizing {num_graphs} power grid graphs ({self.dataset_type})...")

        for idx in tqdm(range(num_graphs), desc=f"Processing {self.dataset_type}"):
            data = HeteroData()

            # 1. Variable Node Features & Targets
            est_row = [float(x) for x in estimate_rows[idx]]
            x_vars = torch.tensor(est_row, dtype=torch.float).view(-1, 1)
            data['variable'].x = torch.cat([flat_start, encoded_indices], dim=-1)
            data['variable'].y = x_vars

            # 2. Factor Node Features (Measurement, Log-Variance, Scaled Covariance)
            meas = torch.tensor([float(x) for x in measurement_rows[idx]], dtype=torch.float)
            var = torch.log10(torch.tensor([float(x) for x in variance_rows[idx]], dtype=torch.float)) / 10.0
            cov_temp = torch.tensor([float(x) for x in covariance_rows[idx]], dtype=torch.float)
            cov = torch.repeat_interleave(cov_temp, 2) * 10.0
            data['factor'].x = torch.stack([meas, var, cov], dim=-1)

            # 3. Bipartite Edges (Factor <-> Variable) derived from non-zero Jacobian elements
            start_idx = idx * num_measurements
            end_idx = start_idx + num_measurements
            jacob_block = torch.tensor(jacobian_rows[start_idx:end_idx], dtype=torch.float)
            edge_index_fv = torch.nonzero(torch.abs(jacob_block) > 1e-4).t().contiguous()
            data['factor', 'to', 'variable'].edge_index = edge_index_fv
            data['variable', 'to', 'factor'].edge_index = edge_index_fv.flip(0)

            # 4. Augmented Variable-to-Variable Edges (Shared Measurement Factor Adjacency)
            binary_jacob = (torch.abs(jacob_block) > 1e-4).float()
            adj_vv = torch.matmul(binary_jacob.t(), binary_jacob)
            edge_index_vv = torch.nonzero(adj_vv > 0).t().contiguous()
            data['variable', 'to', 'variable'].edge_index = edge_index_vv

            data_list.append(data)

        print(f"Saving compiled PyG graph dataset to {self.processed_paths[0]}...")
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save(self.collate(data_list), self.processed_paths[0])


def main():
    parser = argparse.ArgumentParser(description="Build PyTorch Geometric Power Grid Dataset.")
    parser.add_argument('--root', type=str, default='./', help='Root directory for dataset.')
    parser.add_argument('--raw_dir', type=str, default='IEEE30', help='Raw data directory name (e.g., IEEE30).')
    args = parser.parse_args()

    print("Generating PyG HeteroData graph datasets...")
    train_ds = PowerGridDataset(root=args.root, dataset_type='train', raw_data_dir=args.raw_dir)
    val_ds = PowerGridDataset(root=args.root, dataset_type='validation', raw_data_dir=args.raw_dir)
    test_ds = PowerGridDataset(root=args.root, dataset_type='test', raw_data_dir=args.raw_dir)

    print(f"Dataset summary for {args.raw_dir}:")
    print(f"  - Train graphs: {len(train_ds)}")
    print(f"  - Validation graphs: {len(val_ds)}")
    print(f"  - Test graphs: {len(test_ds)}")


if __name__ == "__main__":
    main()