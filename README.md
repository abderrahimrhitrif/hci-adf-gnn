# ⚡ HCI ADF-GNN: Human-in-the-Loop Decision Support for Power Grid State Estimation

> **Note**: This repository is an extension of [**BGNN-PSSE**](https://github.com/abderrahimrhitrif/BGNN-PSSE) (*Bipartite Graph Neural Network for Power System State Estimation*). It extends the original ADF-GNN architecture into an interactive, human-centered decision-support system tailored for power system operators.

---

## 💡 What is this Project?

In modern electrical power grids, system operators must continuously estimate voltage magnitudes and phase angles across all substations (buses) based on readings collected from physical sensors (factor nodes). 

While deep learning models can estimate grid states rapidly, operators often hesitate to trust "black-box" predictions—especially during sensor faults, bad weather, or cyber-attacks (False Data Injection).

**HCI ADF-GNN** solves this by putting the human operator in the loop:

1. **🎯 Human Noise Steering**: Allows operators to adjust sensor noise variance ($\sigma^2$) or override suspicious sensors in real-time, observing how uncertainty updates across the grid.
2. **📊 Dual Uncertainty Quantification (UQ)**: Separates total prediction variance into:
   * **Aleatoric Uncertainty** (*Data Noise*): Propagated analytically through graph edges using Assumed Density Filtering (ADF moment matching).
   * **Epistemic Uncertainty** (*Model Noise*): Calculated via Monte Carlo Dropout forward passes ($T$ passes).
3. **🔍 Explainable AI (XAI)**: Visualizes Graph Attention Network (GAT) edge weights ($\alpha$) to show *which* physical sensors the AI relies on most to estimate a specific bus voltage.
4. **🖥️ Interactive Visual Analytics Dashboard**: A single-page Streamlit dashboard featuring network topology graphs, dynamic confidence intervals ($\mu \pm 2\sigma$), and attention heatmaps.

---

## 🌟 Key Features

* **Real-time Re-estimation**: Instantly re-calculates state estimates when sensor parameters are steered.
* **Network Topology Graph**: Interactive 2D graph visualizer displaying factor measurement nodes, variable state nodes, and active selections using NetworkX & Plotly.
* **XAI Sensor Influence Inspector**: Displays exact percentage/weight contributions of nearby sensors for any selected electrical bus state.

---

## 📁 Repository Structure

```
├── app.py              # Main Streamlit Interactive Visual Dashboard
├── model_adf.py        # Assumed Density Filtering GAT (ADF-GAT) Model
├── predict_adf.py      # Dual UQ Inference & Monte Carlo Dropout Engine
├── adf_core.py         # Analytical Gaussian Moment Matching (Linear & ReLU)
├── create_graph.py     # PyTorch Geometric Dataset Loader for WLS-SE Grid Data
├── cli_demo.py         # Lightweight CLI demonstration of noise steering & XAI
├── requirements.txt    # Python dependencies
└── processed/          # Compiled PyG graph datasets (IEEE 30-bus)
```

---

## 🚀 Quickstart & Usage

### 1. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/abderrahimrhitrif/hci-adf-gnn.git
cd hci-adf-gnn
pip install -r requirements.txt
```

### 2. Run the Interactive Dashboard
Launch the decision-support web dashboard:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to explore:
* **Tab 1: ⚡ Dual UQ State Estimation** (Interactive confidence bounds and baseline comparisons)
* **Tab 2: 🕸️ Power Grid Graph Topology** (Interactive 2D heterogeneous network map)
* **Tab 3: 🔍 XAI Attention Inspector** (Sensor influence bar charts and UQ breakdown)

### 3. Run the CLI Steering Demo
To test human steering in a terminal interface:

```bash
python cli_demo.py
```

---

## 📜 Citation & Credits
This project builds upon the foundational research in [BGNN-PSSE](https://github.com/abderrahimrhitrif/BGNN-PSSE). If you use this codebase or dashboard in your research, please reference the original repository.
