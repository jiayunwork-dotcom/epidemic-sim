import sys
import numpy as np

sys.path.insert(0, '.')
from models import run_model, get_compartment_names
from analysis import (
    estimate_rt, compute_daily_new_infections,
    run_monte_carlo_ci,
)

def test_rt_estimation():
    print("="*60)
    print("TEST 1: R_t Estimation (peak→should not drop to 0)")
    print("="*60)
    N = 1_000_000
    params = {'beta': 0.3, 'gamma': 0.1, 'N': N}
    I0 = 100
    R0_init = 0
    t_span = (0, 300)
    t_eval = np.linspace(0, 300, 301)

    sol = run_model('SIR', params, N, I0, R0_init, t_span, t_eval=t_eval)
    comp_names = get_compartment_names('SIR')
    print(f"SIR success={sol.success}, shape={sol.y.shape}")

    daily_new = compute_daily_new_infections(sol.y, comp_names)
    I_vals = sol.y[comp_names.index("I")]
    R_vals = sol.y[comp_names.index("R")]
    peak_idx = np.argmax(I_vals)
    peak_day = sol.t[peak_idx]
    print(f"Peak Day: {peak_day:.0f}, Peak I: {I_vals[peak_idx]:,.0f}")
    print(f"daily_new range: [{np.min(daily_new):.2f}, {np.max(daily_new):.2f}]")
    print(f"daily_new @ peak: {daily_new[peak_idx]:.2f}")
    print(f"daily_new [after peak] day {int(peak_day)+30}: {daily_new[int(peak_day)+30]:.2f}")

    rt_times, rt_vals = estimate_rt(daily_new)
    print(f"Rt length={len(rt_times)}, range=[{np.min(rt_vals):.3f}, {np.max(rt_vals):.3f}]")
    print(f"Rt after peak (Day {int(peak_day)+10}): ", end="")
    mask_after_peak = rt_times >= peak_day + 10
    if np.any(mask_after_peak):
        idx_first = np.where(mask_after_peak)[0][0]
        print(f"{rt_vals[idx_first]:.3f}")
    else:
        print("NO DATA after peak!")

    mask_last_100 = rt_times > 200
    if np.any(mask_last_100):
        rt_last = rt_vals[mask_last_100]
        print(f"Rt at Day>200: min={np.min(rt_last):.3f}, max={np.max(rt_last):.3f}")
        print(f"Any zeros? {np.any(rt_last == 0.0)}")

    rt_below_1 = np.where(rt_vals < 0.995)[0]
    if len(rt_below_1) > 0:
        print(f"✅ Rt first drops below 1 on Day {int(rt_times[rt_below_1[0]])}")
    else:
        print("❌ Rt NEVER drops below 1!")

    return True


def test_monte_carlo_ci():
    print("\n" + "="*60)
    print("TEST 2: Monte Carlo Confidence Intervals")
    print("="*60)
    N = 1_000_000
    params = {'beta': 0.3, 'gamma': 0.1, 'N': N}
    I0 = 100
    R0_init = 0
    t_span = (0, 200)

    param_ranges = {
        'beta': (0.01, 2.0),
        'gamma': (0.01, 1.0),
    }
    result = run_monte_carlo_ci('SIR', params, N, I0, R0_init, t_span,
                                 param_ranges, n_runs=20, confidence=0.95)
    lower = result["lower"]
    upper = result["upper"]
    median = result["median"]
    I_idx = get_compartment_names('SIR').index("I")

    lower_I = lower[I_idx]
    upper_I = upper[I_idx]
    med_I = median[I_idx]

    print(f"Median I peak: {np.max(med_I):,.0f}")
    print(f"Lower I min={np.min(lower_I):.2f} (should be >= 0)")
    print(f"Upper I peak={np.max(upper_I):,.0f}")

    has_negative = np.any(lower < 0)
    print(f"Any negative lower bounds? {has_negative}")
    if has_negative:
        print("❌ Negative values detected!")
    else:
        print("✅ No negative values")

    ci_width = float(np.mean(upper_I - lower_I) / np.maximum(np.mean(med_I), 1))
    print(f"Avg CI width (relative to median): {ci_width:.4f}")
    if ci_width < 0.01:
        print("❌ CI too narrow (sampling likely incorrect)")
    elif ci_width > 10:
        print("❌ CI unreasonably wide")
    else:
        print(f"✅ CI width looks reasonable: ~{ci_width*100:.1f}% of median")

    return True


def test_scenario_metrics():
    print("\n" + "="*60)
    print("TEST 3: Scenario Metrics (beta=0.2 vs beta=0.5)")
    print("="*60)
    N = 1_000_000
    I0 = 100
    R0_init = 0
    t_span = (0, 300)
    t_eval = np.linspace(0, 300, 301)

    results = {}
    for beta_val in [0.2, 0.5]:
        params = {'beta': beta_val, 'gamma': 0.1, 'N': N}
        sol = run_model('SIR', params, N, I0, R0_init, t_span, t_eval=t_eval)
        comp_names = get_compartment_names('SIR')

        I_vals = sol.y[comp_names.index("I")]
        peak_infections = np.max(I_vals)
        peak_time = sol.t[np.argmax(I_vals)]
        cumulative = sol.y[comp_names.index("R")][-1]

        daily_new = compute_daily_new_infections(sol.y, comp_names)
        rt_times, rt_values = estimate_rt(daily_new)
        rt_below_1_day = None
        if len(rt_values) > 0:
            smoothed_rt = rt_values
            if len(smoothed_rt) > 7:
                smoothed_rt = np.convolve(smoothed_rt, np.ones(7)/7, mode='same')
            rt_below_1 = np.where(smoothed_rt < 0.995)[0]
            if len(rt_below_1) > 0:
                rt_below_1_day = int(rt_times[rt_below_1[0]])

        results[beta_val] = {
            "cumulative": cumulative,
            "peak": peak_infections,
            "peak_time": peak_time,
            "rt_below_1": rt_below_1_day,
        }
        print(f"beta={beta_val}: peak={peak_infections:,.0f} @ Day {peak_time:.0f}, "
              f"cumul={cumulative:,.0f}, Rt<1 Day={rt_below_1_day}")

    all_good = True
    for beta_val, res in results.items():
        if res["rt_below_1"] is None:
            print(f"❌ beta={beta_val}: rt_below_1 is None!")
            all_good = False
        else:
            print(f"✅ beta={beta_val}: rt_below_1 = Day {res['rt_below_1']}")
    return all_good


if __name__ == "__main__":
    try:
        test_rt_estimation()
        test_monte_carlo_ci()
        test_scenario_metrics()
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)
    except Exception as e:
        import traceback
        traceback.print_exc()
