import torch

from model_adf import ADFPowerGridGNN
from predict_adf import predict_with_adf


def main():
    print("CLI DEMO XAI")

    sample, _ = torch.load("processed/test_data.pt", weights_only=False)
    print("Data loaded")
    model = ADFPowerGridGNN(
        var_in_dim=7, factor_in_dim=3, hidden_dim=32, readout_units=16, output_units=1
    )
    model.load_state_dict(
        torch.load("best_model_adf.pth", weights_only=False), strict=False
    )
    model.eval()
    base_res = predict_with_adf(model, sample, num_mc_samples=10, v_init=1e-5)

    bus0_mu = base_res["mu"][0].item()
    bus0_var = base_res["var_aleatoric"][0].item()

    print(f"Bus 0 Estimated Voltage : {bus0_mu:.5f} p.u.")
    print(f"Bus 0 Aleatoric Variance: {bus0_var:.6f}")

    num_sensors = sample["factor"].x.size(0)
    print("Human in the loop part")
    sensor_id = int(input(f"Enter Sensor ID to adjust (0 to {num_sensors - 1}) :"))
    noise_val = float(input("Enter new noise variance : "))

    costum_var = torch.full_like(sample["factor"].x, 1e-5)
    costum_var[sensor_id] = noise_val

    steered_res = predict_with_adf(
        model, sample, num_mc_samples=10, var_factor_override=costum_var
    )

    bus0_mu_steered = steered_res["mu"][0].item()
    bus0_var_steered = steered_res["var_aleatoric"][0].item()

    print("\n" + "=" * 50)
    print(f"  STEERING RESULT COMPARISON (Bus 0)")
    print("=" * 50)
    print(f"Baseline Voltage : {bus0_mu:.5f} p.u. | Baseline Var: {bus0_var:.6f}")
    print(
        f"Steered Voltage  : {bus0_mu_steered:.5f} p.u. | Steered Var : {bus0_var_steered:.6f}"
    )

    atts = model.get_attention_weights()
    print(f"✓ XAI: Extracted attention weights across {len(atts)} GAT layers!")


if __name__ == "__main__":
    main()
