# HCI ADF-GNN: Power Grid State Estimation & Decision Support Dashboard

## Overview
An interactive decision-support dashboard built on top of an Analytic Derivative Free Graph Neural Network (ADF-GNN) for power system state estimation on the IEEE 30-bus system. It combines graph neural networks, dual uncertainty quantification, explainable AI (XAI), and human-in-the-loop noise steering.

## Architecture & Pipeline
1. **ADF-GNN Model (`model_adf.py`)**: Graph attention network incorporating analytic derivative-free moment matching for variance propagation across network edges.
2. **Dual Uncertainty Quantification**: Quantifies aleatoric uncertainty (sensor noise propagated analytically) and epistemic uncertainty (model ignorance via Monte Carlo dropout).
3. **Human-in-the-loop Steering (`noise_utils.py`)**: Operators can override sensor noise variance or inject noise into specific state variables to observe real-time uncertainty response.
4. **Interactive Dashboard (`app.py`)**: Streamlit web interface rendering network topology, confidence intervals ($\mu \pm 2\sigma$), attention heatmaps, and XAI sensor influence inspectors.

## Data
- **Dataset**: IEEE 30-bus power system simulation dataset (`IEEE30/`).
- **Features**: 60 variable state nodes and 100 factor sensors.

## Method & Algorithms
- **Graph Attention Networks (GAT)**: Message passing over power grid graph structures.
- **Uncertainty Propagation**: Analytic Derivative-Free (ADF) moment matching combined with MC dropout.

## Verified Results & Metrics
- Implemented interactive Streamlit dashboard supporting real-time sensor noise steering across 60 variable state nodes and 100 factor sensors.
- Visualized $\mu \pm 2\sigma$ confidence bounds and graph-attention edge weights.

## Limitations & Caveats
- Research prototype evaluated on simulated IEEE 30-bus power grid datasets.

## Run Instructions
```bash
pip install -r requirements.txt
streamlit run app.py
```
