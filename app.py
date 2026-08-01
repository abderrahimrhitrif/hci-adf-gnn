import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import torch

from create_graph import PowerGridDataset
from model_adf import ADFPowerGridGNN
from predict_adf import predict_with_adf

st.set_page_config(
    page_title="HCI ADF-GNN Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("HCI ADF-GNN: Decision Support Dashboard")
st.markdown("""
*An interactive decision-support system featuring **Dual Uncertainty Quantification (UQ)**,
**Human Noise Steering**, **Graph Topology Visualization**, and **Explainable AI (XAI)** attention maps.*
""")


@st.cache_resource
def load_assests():
    dataset = PowerGridDataset(root="./", dataset_type="test", raw_data_dir="IEEE30")
    model = ADFPowerGridGNN(
        var_in_dim=7, factor_in_dim=3, hidden_dim=32, readout_units=16, output_units=1
    )
    model.load_state_dict(
        torch.load("best_model_adf.pth", weights_only=False), strict=False
    )
    model.eval()
    return dataset, model


dataset, model = load_assests()

st.sidebar.header("Human Steering Controls")

snapshot_idx = st.sidebar.slider(
    "Test Snapshot Index", min_value=0, max_value=len(dataset) - 1, value=0
)

sample = dataset[snapshot_idx]
num_sensor = sample["factor"].x.size(0)
num_buses = sample["variable"].x.size(0)

selected_sensor = st.sidebar.number_input(
    "Select Factor/Sensor ID", min_value=0, max_value=num_sensor - 1, value=0, step=1
)

noise_variance = st.sidebar.select_slider(
    "Sensor Noise Variance (σ²)",
    options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    value=1e-5,
    format_func=lambda x: f"{x:.1e}",
)

mc_samples = st.sidebar.slider(
    "Monte Carlo Samples (T)", min_value=5, max_value=30, value=15, step=2
)

# Construct custom variance vector
custom_var = torch.full_like(sample["factor"].x, 1e-5)
custom_var[int(selected_sensor)] = noise_variance

with st.spinner("Running ADF-GNN Inference & Dual UQ..."):
    base_res = predict_with_adf(model, sample, num_mc_samples=mc_samples, v_init=1e-5)
    steered_res = predict_with_adf(
        model, sample, num_mc_samples=mc_samples, var_factor_override=custom_var
    )

tab_uq, tab_graph, tab_xai = st.tabs(
    [
        "Dual UQ State Estimation",
        "Power Grid Graph Topology",
        "XAI Attention Inspector",
    ]
)

with tab_uq:
    st.subheader("Power Grid State Estimation & Confidence Bounds")

    buses = list(range(num_buses))
    base_mu = base_res["mu"].squeeze().cpu().numpy()

    steered_mu = steered_res["mu"].squeeze().cpu().numpy()
    steered_std = torch.sqrt(steered_res["var_total"]).squeeze().cpu().numpy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=buses + buses[::-1],
            y=list(steered_mu + 2 * steered_std)
            + list(steered_mu - 2 * steered_std)[::-1],
            fill="toself",
            fillcolor="rgba(255, 75, 75, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Steered 95% CI (±2σ)",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=buses,
            y=base_mu,
            mode="lines+markers",
            name="Baseline Voltage (p.u.)",
            line=dict(color="gray", dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=buses,
            y=steered_mu,
            mode="lines+markers",
            name="Steered Voltage (p.u.)",
            line=dict(color="#FF4B4B", width=2),
        )
    )

    fig.update_layout(
        xaxis_title="Variable State Node Index (0 to 59)",
        yaxis_title="Voltage State (p.u.)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
    )

    st.plotly_chart(fig, width="stretch")

with tab_graph:
    st.subheader(f"🕸️ Heterogeneous Graph Layout for Snapshot #{snapshot_idx}")
    st.markdown("""
    Visualizing the bipartite graph structure: **Green markers** represent Factor Sensors (measurements),
    **Blue markers** represent Bus State Variables, and **Red/Orange markers** highlight active selections.
    """)

    @st.cache_data
    def build_graph_layout(_sample_data):
        edge_index_fv = (
            _sample_data[("factor", "to", "variable")].edge_index.cpu().numpy()
        )
        n_factors = _sample_data["factor"].x.size(0)
        n_vars = _sample_data["variable"].x.size(0)

        G = nx.Graph()
        for f in range(n_factors):
            G.add_node(f"F_{f}", type="factor", index=f)
        for v in range(n_vars):
            G.add_node(f"V_{v}", type="variable", index=v)

        for f_idx, v_idx in zip(edge_index_fv[0], edge_index_fv[1]):
            G.add_edge(f"F_{f_idx}", f"V_{v_idx}")

        pos = nx.spring_layout(G, k=0.20, seed=42)
        return G, pos

    G, pos = build_graph_layout(sample)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.6, color="rgba(180, 180, 180, 0.4)"),
        hoverinfo="none",
        mode="lines",
    )

    factor_x, factor_y, factor_text, factor_color, factor_size = [], [], [], [], []
    for f in range(num_sensor):
        node_id = f"F_{f}"
        x, y = pos[node_id]
        factor_x.append(x)
        factor_y.append(y)
        factor_text.append(f"Factor Sensor {f}")
        if f == int(selected_sensor):
            factor_color.append("#FF4B4B")  # Active Steered Sensor (Red)
            factor_size.append(14)
        else:
            factor_color.append("#00CC96")  # Factor Sensors (Green)
            factor_size.append(8)

    factor_trace = go.Scatter(
        x=factor_x,
        y=factor_y,
        mode="markers",
        hoverinfo="text",
        text=factor_text,
        name="Factor Sensors (Measurements)",
        marker=dict(color=factor_color, size=factor_size, line_width=1),
    )

    var_x, var_y, var_text, var_color, var_size = [], [], [], [], []
    for v in range(num_buses):
        node_id = f"V_{v}"
        x, y = pos[node_id]
        var_x.append(x)
        var_y.append(y)
        var_text.append(f"Variable Node {v}")
        var_color.append("#636EFA")  # Variable Nodes (Blue)
        var_size.append(8)

    var_trace = go.Scatter(
        x=var_x,
        y=var_y,
        mode="markers",
        hoverinfo="text",
        text=var_text,
        name="Variable Nodes (Bus States)",
        marker=dict(color=var_color, size=var_size, line_width=1),
    )

    fig_graph = go.Figure(data=[edge_trace, factor_trace, var_trace])
    fig_graph.update_layout(
        title=f"Power Grid Heterogeneous Topology (100 Factor Sensors ↔ 60 Variable States)",
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    st.plotly_chart(fig_graph, width="stretch")

with tab_xai:
    st.subheader("🔍 Explainable AI (XAI): GAT Attention Weight Inspector")
    st.markdown(
        "Inspect which physical measurement sensors the GNN relies on most for any selected state node."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        target_node = st.number_input(
            "Select State Node ID to Explain",
            min_value=0,
            max_value=num_buses - 1,
            value=0,
            step=1,
        )

        target_aleatoric = steered_res["var_aleatoric"].squeeze()[target_node].item()
        target_epistemic = steered_res["var_epistemic"].squeeze()[target_node].item()
        target_total = steered_res["var_total"].squeeze()[target_node].item()

        st.metric("Total Uncertainty (σ²)", f"{target_total:.3e}")
        st.metric("Aleatoric (Data Noise)", f"{target_aleatoric:.3e}")
        st.metric("Epistemic (Model Noise)", f"{target_epistemic:.3e}")

    with col2:
        attentions = model.get_attention_weights()

        if len(attentions) > 0:
            alpha = attentions[0].squeeze()
            edge_fv = sample[("factor", "to", "variable")].edge_index

            mask = edge_fv[1] == target_node
            connecting_factors = edge_fv[0][mask].cpu().numpy()
            attention_scores = alpha[mask].cpu().numpy()

            if len(connecting_factors) > 0:
                fig_xai = go.Figure(
                    go.Bar(
                        x=[f"Sensor {f_id}" for f_id in connecting_factors],
                        y=attention_scores,
                        marker_color="#00CC96",
                    )
                )

                fig_xai.update_layout(
                    xaxis_title="Connected Factor Sensor",
                    yaxis_title="GAT Attention Weight (α)",
                    title=f"Sensor Influence on Variable State Node {target_node}",
                    margin=dict(l=20, r=20, t=40, b=20),
                )

                st.plotly_chart(fig_xai, width="stretch")
            else:
                st.info(
                    f"No direct factor sensor connections found for Node {target_node}."
                )
        else:
            st.warning("No attention weights logged.")
