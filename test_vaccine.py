import numpy as np
from models import run_vaccine_strategy_model, compute_vaccine_metrics

contact_matrix = np.array([
    [2.0, 0.5, 0.4, 0.2],
    [0.5, 3.0, 0.8, 0.3],
    [0.4, 0.8, 2.5, 0.6],
    [0.2, 0.3, 0.6, 1.5],
])
age_props = np.array([0.20, 0.35, 0.30, 0.15])
params = {"beta": 0.08, "gamma": 0.1, "sigma": 0.3, "omega": 0.01}
N = 1000000
I0 = 100
R0_init = 0
total_vaccines = 300000
vacc_days = 60
efficacy = 0.8
custom_props = [0.15, 0.30, 0.35, 0.20]

print("Running vaccine strategy model...")
result = run_vaccine_strategy_model(
    "SIR", params, contact_matrix, age_props,
    N, I0, R0_init, total_vaccines, vacc_days, efficacy,
    custom_props=custom_props, sim_days=300
)
print("Model run completed successfully.")
print("Time eval shape:", result["t_eval"].shape)
print("Baseline shape:", result["baseline"].shape)
print("Strategies:", list(result["strategies"].keys()))
for k, v in result["strategies"].items():
    print("  ", k, ": shape=", v.shape)
print("Actual doses used:", result["strategy_actual_doses"])
print("Immune conversions:", {k: v.sum() for k, v in result["strategy_immune_conversions"].items()})

print()
print("Computing metrics...")
metrics = compute_vaccine_metrics(result, N, total_vaccines)
print("Baseline cumulative:", f"{metrics['baseline_cum']:,.0f}", f"({metrics['baseline_cum']/N*100:.2f}%)")
print("Baseline peak:", f"{metrics['baseline_peak']:,.0f}", "at day", f"{metrics['baseline_peak_day']:.0f}")
print()
for strat_name, strat_metrics in metrics["strategies"].items():
    print("  ", strat_name, ":")
    print("    Cumulative:", f"{strat_metrics['cum_infections']:,.0f}", f"({strat_metrics['cum_infections']/N*100:.2f}%)")
    print("    Reduction:", f"{strat_metrics['cum_reduction_pct']:.2f}%")
    print("    Peak delay:", f"{strat_metrics['peak_delay']:.1f}", "days")
    print("    Peak reduction:", f"{strat_metrics['peak_reduction_pct']:.2f}%")
    print("    Vaccine efficiency:", f"{strat_metrics['vacc_efficiency']:.4f}", "infections avoided per dose")
    print("    Actual doses:", f"{strat_metrics['actual_doses']:,.0f}", "/", f"{total_vaccines:,}")
    print("    Fairness:", f"{strat_metrics['fairness']:.4f}")
    print("    Attack rates:", [f"{x*100:.2f}%" for x in strat_metrics["attack_rates"]])
    print()

print("=== VALIDATION CHECKS ===")
print("1. All cumulative infections <= N:", all(
    strat_metrics["cum_infections"] <= N * 1.001
    for strat_metrics in metrics["strategies"].values()
))
print("2. All attack rates between 0 and 1:", all(
    all(0 <= x <= 1.001 for x in strat_metrics["attack_rates"])
    for strat_metrics in metrics["strategies"].values()
))
print("3. Vaccine strategies have <= cumulative infections than baseline:", all(
    strat_metrics["cum_infections"] <= metrics["baseline_cum"] * 1.001
    for strat_metrics in metrics["strategies"].values()
))
print("4. Fairness index between 0 and 1:", all(
    0 <= strat_metrics["fairness"] <= 1.0
    for strat_metrics in metrics["strategies"].values()
))
