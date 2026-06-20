import numpy as np
from models import run_vaccine_strategy_model, compute_vaccine_metrics
from visualization import plot_vaccine_cumulative_comparison, plot_vaccine_radar
import matplotlib
matplotlib.use("Agg")

AGE_GROUPS = ["0-17", "18-44", "45-64", "65+"]

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

print("Running model...")
result = run_vaccine_strategy_model(
    "SIR", params, contact_matrix, age_props,
    N, I0, R0_init, total_vaccines, vacc_days, efficacy,
    custom_props=custom_props, sim_days=300
)

print("Computing metrics...")
metrics = compute_vaccine_metrics(result, N, total_vaccines)

print("\n=== Radar Chart Dimensions Validation ===")
for strat_name, strat_metrics in metrics["strategies"].items():
    print(f"\n{strat_name}:")
    print(f"  1. 累计感染减少率 (Infection Reduction %): {strat_metrics['cum_reduction_pct']:.2f}%")
    print(f"  2. 峰值降低率 (Peak Reduction %): {strat_metrics['peak_reduction_pct']:.2f}%")
    print(f"  3. 达峰延迟 (Peak Delay): {max(0, strat_metrics['peak_delay']):.1f} days")
    print(f"  4. 疫苗利用效率 (Vaccine Efficiency): {strat_metrics['vacc_efficiency']:.4f} infections/dose")
    print(f"  5. 公平性指数 (Fairness Index): {strat_metrics['fairness']:.4f}")

print("\nGenerating plots...")
fig_cum = plot_vaccine_cumulative_comparison(result, metrics, AGE_GROUPS)
print("Cumulative comparison plot generated successfully.")
print(f"  Figure size: {fig_cum.get_size_inches()}")

fig_radar = plot_vaccine_radar(metrics)
print("Radar plot generated successfully.")
print(f"  Figure size: {fig_radar.get_size_inches()}")

print("\nAll visualization tests passed!")
