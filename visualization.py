import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import io

COMPARTMENT_COLORS = {
    "S": "#3498db",
    "E": "#f39c12",
    "I": "#e74c3c",
    "R": "#2ecc71",
    "Q": "#9b59b6",
}

SCENARIO_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]


def _get_comp_names(model_type):
    if model_type == "SIR":
        return ["S", "I", "R"]
    elif model_type in ("SEIR", "SEIRS"):
        return ["S", "E", "I", "R"]
    elif model_type == "SIS":
        return ["S", "I"]
    return ["S", "I", "R"]


def plot_epidemic_curve(sol, model_type, chart_type="line", interventions=None, title=""):
    comp_names = _get_comp_names(model_type)
    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "stacked":
        bottoms = np.zeros(len(sol.t))
        for i, name in enumerate(comp_names):
            vals = sol.y[i]
            ax.fill_between(sol.t, bottoms, bottoms + vals, alpha=0.7,
                           label=name, color=COMPARTMENT_COLORS.get(name, "#95a5a6"),
                           edgecolor='white', linewidth=0.5)
            bottoms += vals
        ax.set_ylabel("Population (stacked)")
    else:
        for i, name in enumerate(comp_names):
            ax.plot(sol.t, sol.y[i], label=name,
                   color=COMPARTMENT_COLORS.get(name, "#95a5a6"), linewidth=2.5)

    if interventions:
        _add_intervention_bars(ax, interventions, sol.t[-1])

    ax.set_xlabel("Days")
    ax.set_ylabel("Population")
    ax.set_title(title or f"{model_type} Model - Epidemic Curve", fontweight="bold", fontsize=14)
    ax.legend(loc="upper right", frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_alpha(0.3)
    ax.spines["right"].set_alpha(0.3)
    fig.tight_layout()
    return fig


def plot_epidemic_curve_with_ci(sol, model_type, ci_result, chart_type="line", title=""):
    comp_names = _get_comp_names(model_type)
    fig, ax = plt.subplots(figsize=(10, 6))

    t_eval = ci_result["t_eval"]
    lower = ci_result["lower"]
    upper = ci_result["upper"]
    median = ci_result["median"]

    for i, name in enumerate(comp_names):
        color = COMPARTMENT_COLORS.get(name, "#95a5a6")
        ax.plot(sol.t, sol.y[i], label=name, color=color, linewidth=2.5)
        ax.fill_between(t_eval, lower[i], upper[i], alpha=0.2, color=color, edgecolor=None)

    ax.set_xlabel("Days")
    ax.set_ylabel("Population")
    ax.set_title(title or f"{model_type} Model with 95% Confidence Interval", fontweight="bold", fontsize=14)
    ax.legend(loc="upper right", frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_alpha(0.3)
    ax.spines["right"].set_alpha(0.3)
    fig.tight_layout()
    return fig


def plot_rt_curve(rt_times, rt_values, title="Effective Reproduction Number (R_t)"):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(rt_times, rt_values, color="#3498db", linewidth=2.5, label="R_t")
    ax.axhline(y=1, color="#e74c3c", linestyle="--", linewidth=2, label="R_t = 1 (Critical)")

    ax.fill_between(rt_times, 1, rt_values,
                    where=rt_values >= 1, alpha=0.15, color="#e74c3c", interpolate=True)
    ax.fill_between(rt_times, rt_values, 1,
                    where=rt_values < 1, alpha=0.15, color="#2ecc71", interpolate=True)

    ax.set_xlabel("Days")
    ax.set_ylabel("R_t")
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.legend(frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_alpha(0.3)
    ax.spines["right"].set_alpha(0.3)
    fig.tight_layout()
    return fig


def plot_intervention_timeline(interventions, total_days, title="Intervention Timeline"):
    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(interventions) + 1.5)))

    interv_labels = {
        "quarantine": "Quarantine",
        "vaccination": "Vaccination",
        "social_distance": "Social Distancing",
        "regional_lockdown": "Regional Lockdown",
    }
    interv_colors = {
        "quarantine": "#9b59b6",
        "vaccination": "#2ecc71",
        "social_distance": "#f39c12",
        "regional_lockdown": "#e74c3c",
    }

    y_positions = list(range(len(interventions)))

    for i, interv in enumerate(interventions):
        start = interv.get("start_day", 0)
        duration = interv.get("duration", 30)
        itype = interv.get("type", "")
        label = interv_labels.get(itype, itype)
        color = interv_colors.get(itype, "#95a5a6")

        ax.barh(i, duration, left=start, height=0.5, color=color, alpha=0.85,
                edgecolor="white", linewidth=1.5)
        mid_point = start + duration / 2
        if mid_point < total_days:
            ax.text(mid_point, i, label, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

    ax.set_xlabel("Days")
    ax.set_xlim(0, total_days)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([interv_labels.get(iv.get("type", ""), "") for iv in interventions])
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")
    ax.spines["top"].set_alpha(0.3)
    ax.spines["right"].set_alpha(0.3)
    fig.tight_layout()
    return fig


def plot_spatial_heatmap(t_eval, region_curves, region_names=None, title="Spatial Infection Heatmap"):
    n_regions = len(region_curves)
    n_days = len(t_eval)

    data = np.zeros((n_days, n_regions))
    for r in range(n_regions):
        data[:, r] = region_curves[r]

    fig, ax = plt.subplots(figsize=(max(10, n_regions * 2 + 4), max(6, n_days / 15)))
    im = ax.imshow(data.T, aspect="auto", cmap="YlOrRd",
                   origin="lower", interpolation="bilinear")

    if region_names is None:
        region_names = [f"Region {i+1}" for i in range(n_regions)]

    ax.set_yticks(range(n_regions))
    ax.set_yticklabels(region_names, fontsize=10)
    ax.set_xlabel("Days", fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=14, pad=15)

    n_ticks = min(10, n_days)
    tick_positions = np.linspace(0, n_days - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(t_eval[t])) for t in tick_positions], rotation=45, ha="right")

    cbar = plt.colorbar(im, ax=ax, label="Infected Population", pad=0.02)
    cbar.ax.tick_params(labelsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_spatial_curves(t_eval, region_curves, region_names=None, title="Multi-Region Infection Curves"):
    fig, ax = plt.subplots(figsize=(10, 6))

    if region_names is None:
        region_names = [f"Region {i+1}" for i in range(len(region_curves))]

    cmap = plt.cm.tab10
    for r, (curve, name) in enumerate(zip(region_curves, region_names)):
        ax.plot(t_eval, curve, label=name, color=cmap(r % 10), linewidth=2.5, alpha=0.9)

    ax.set_xlabel("Days")
    ax.set_ylabel("Infected Population")
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.legend(loc="upper right", frameon=True, fancybox=True, ncol=min(4, len(region_curves)))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_alpha(0.3)
    ax.spines["right"].set_alpha(0.3)
    fig.tight_layout()
    return fig


def plot_scenario_comparison(t_eval, scenarios_data, comp_idx=1, comp_name="I", title="Scenario Comparison"):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, sol) in enumerate(scenarios_data):
        color = SCENARIO_COLORS[i % len(SCENARIO_COLORS)]
        if comp_idx < sol.y.shape[0]:
            ax.plot(sol.t, sol.y[comp_idx], label=name, color=color, linewidth=2.5, alpha=0.9)

    ax.set_xlabel("Days")
    ax.set_ylabel(f"{comp_name} Population")
    ax.set_title(title + f" - {comp_name} Compartment", fontweight="bold", fontsize=14)
    ax.legend(loc="upper right", frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_alpha(0.3)
    ax.spines["right"].set_alpha(0.3)
    fig.tight_layout()
    return fig


def plot_scenario_metrics_bar(metrics_dict, title="Scenario Metrics Comparison"):
    scenario_names = list(metrics_dict.keys())
    n_scenarios = len(scenario_names)
    metric_info = [
        ("cumulative", "Cumulative Infections"),
        ("peak", "Peak Infections"),
        ("peak_time", "Peak Time (days)"),
    ]

    valid_metrics = [(k, label) for k, label in metric_info if k in metrics_dict[scenario_names[0]]]

    fig, axes = plt.subplots(1, len(valid_metrics), figsize=(5 * len(valid_metrics) + 2, 6))
    if len(valid_metrics) == 1:
        axes = [axes]

    for m, (metric_key, metric_label) in enumerate(valid_metrics):
        values = [metrics_dict[s].get(metric_key, 0) for s in scenario_names]
        colors = [SCENARIO_COLORS[i % len(SCENARIO_COLORS)] for i in range(n_scenarios)]

        bars = axes[m].bar(range(n_scenarios), values, color=colors, alpha=0.85,
                          edgecolor="white", linewidth=1.5)

        for bar, val in zip(bars, values):
            axes[m].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:,.0f}" if abs(val) >= 100 else f"{val:.2f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

        axes[m].set_xticks(range(n_scenarios))
        axes[m].set_xticklabels(scenario_names, rotation=30, ha="right", fontsize=10)
        axes[m].set_title(metric_label, fontsize=12, fontweight="bold")
        axes[m].grid(True, alpha=0.3, axis="y", linestyle="--")
        axes[m].spines["top"].set_alpha(0.3)
        axes[m].spines["right"].set_alpha(0.3)
        axes[m].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    fig.suptitle(title, fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_sensitivity_scatter(samples, param_names, output_values, prcc_values, output_label="Cumulative Infections"):
    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5.5))

    if n_params == 1:
        axes = [axes]

    for i, name in enumerate(param_names):
        ax = axes[i]
        valid_mask = ~np.isnan(output_values) & ~np.isnan(samples[:, i])

        x_vals = samples[valid_mask, i]
        y_vals = output_values[valid_mask]

        ax.scatter(x_vals, y_vals, alpha=0.6, s=35, color="#3498db",
                   edgecolors="white", linewidth=0.5, zorder=2)

        prcc_val = prcc_values[i]
        is_significant = abs(prcc_val) > 0.5
        title_color = "#e74c3c" if is_significant else "#2ecc71"
        sig_marker = "★" if is_significant else ""

        ax.set_title(f"{name}\nPRCC = {prcc_val:.3f} {sig_marker}",
                    color=title_color, fontweight="bold", fontsize=11)
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel(output_label, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--", zorder=1)
        ax.spines["top"].set_alpha(0.3)
        ax.spines["right"].set_alpha(0.3)

        if valid_mask.sum() > 1:
            try:
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(x_line, p(x_line), color="#e74c3c", linewidth=2,
                       linestyle="--", alpha=0.8, zorder=3, label="Trend")
            except Exception:
                pass

    fig.suptitle(f"Sensitivity Analysis - {output_label}", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_optimal_timing(result, title="Optimal Intervention Timing Analysis"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    start_days = result["start_days"]
    cum_inf = result["cumulative_infections"]
    valid_mask = ~np.isnan(cum_inf)

    ax1.plot(start_days[valid_mask], cum_inf[valid_mask], color="#3498db", linewidth=2.5, marker="o",
            markersize=4, alpha=0.8)
    ax1.axvline(x=result["best_start_day"], color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Best start: Day {result['best_start_day']}")

    if len(result["optimal_window"]) > 0:
        win_start = result["optimal_window"][0]
        win_end = result["optimal_window"][-1]
        if win_end > win_start:
            ax1.axvspan(win_start, win_end, alpha=0.2, color="#2ecc71",
                       label=f"Optimal window: Days {win_start}-{win_end}")

    ax1.set_xlabel("Intervention Start Day", fontsize=11)
    ax1.set_ylabel("Cumulative Infections", fontsize=11)
    ax1.set_title("Start Time vs Cumulative Infections", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, fancybox=True, fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.spines["top"].set_alpha(0.3)
    ax1.spines["right"].set_alpha(0.3)
    ax1.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    t_eval = result["t_eval"]
    all_curves = result["all_I_curves"]

    if len(all_curves) >= 3:
        early_idx = 0
        optimal_idx = min(len(all_curves) - 1, max(0, result["best_start_day"] - 1))
        late_idx = len(all_curves) - 1

        early_curve = all_curves[early_idx]
        optimal_curve = all_curves[optimal_idx]
        late_curve = all_curves[late_idx]

        min_len = min(len(early_curve), len(optimal_curve), len(late_curve), len(t_eval))

        ax2.plot(t_eval[:min_len], early_curve[:min_len],
                color="#f39c12", linewidth=2.5, label=f"Early (Day {early_idx + 1})", alpha=0.9)
        ax2.plot(t_eval[:min_len], optimal_curve[:min_len],
                color="#2ecc71", linewidth=2.5, label=f"Optimal (Day {result['best_start_day']})", alpha=0.9)
        ax2.plot(t_eval[:min_len], late_curve[:min_len],
                color="#e74c3c", linewidth=2.5, label=f"Late (Day {late_idx + 1})", alpha=0.9)

    ax2.set_xlabel("Days", fontsize=11)
    ax2.set_ylabel("Infected Population", fontsize=11)
    ax2.set_title("Early vs Optimal vs Late Intervention", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fancybox=True, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.spines["top"].set_alpha(0.3)
    ax2.spines["right"].set_alpha(0.3)
    ax2.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    fig.suptitle(title, fontweight="bold", fontsize=15, y=1.02)
    fig.tight_layout()
    return fig


def plot_R0_indicator(r0_value):
    fig, ax = plt.subplots(figsize=(5, 2.5))

    if r0_value < 1:
        color = "#2ecc71"
        status = "Below 1 - Epidemic Declining ✓"
        bg_color = "#eafaf1"
    elif r0_value < 2:
        color = "#f1c40f"
        status = "1-2 - Moderate Growth ⚠"
        bg_color = "#fef9e7"
    else:
        color = "#e74c3c"
        status = "Above 2 - Rapid Growth ✗"
        bg_color = "#fdedec"

    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)

    ax.text(0.5, 0.65, f"R₀ = {r0_value:.3f}", transform=ax.transAxes,
            fontsize=28, fontweight="bold", color=color,
            ha="center", va="center")
    ax.text(0.5, 0.15, status, transform=ax.transAxes,
            fontsize=12, color=color,
            ha="center", va="center", fontweight="500")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _add_intervention_bars(ax, interventions, max_time):
    interv_colors = {
        "quarantine": "#9b59b6",
        "vaccination": "#2ecc71",
        "social_distance": "#f39c12",
        "regional_lockdown": "#e74c3c",
    }
    for interv in interventions:
        start = interv.get("start_day", 0)
        duration = interv.get("duration", 30)
        itype = interv.get("type", "")
        color = interv_colors.get(itype, "#95a5a6")
        end = min(start + duration, max_time)
        ax.axvspan(start, end, alpha=0.1, color=color, zorder=0)


def plot_vaccine_cumulative_comparison(result, metrics, AGE_GROUPS):
    comp_names = result["comp_names"]
    n_comp = result["n_comp"]
    n_groups = result["n_groups"]
    t_eval = result["t_eval"]
    strategy_immune_ts = result.get("strategy_immune_ts", {})
    baseline_immune_ts = result.get("baseline_immune_ts", np.zeros((n_groups, len(t_eval))))

    strategy_colors = {
        "uniform": "#3498db",
        "elderly_priority": "#e74c3c",
        "high_contact": "#2ecc71",
        "custom": "#f39c12",
    }
    strategy_labels = {
        "uniform": "Uniform",
        "elderly_priority": "Elderly Priority",
        "high_contact": "High-Contact Priority",
        "custom": "Custom",
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    if "R" in comp_names:
        R_idx = comp_names.index("R")
        baseline_cum = np.zeros(len(t_eval))
        for g in range(n_groups):
            r_vals = result["baseline"][R_idx * n_groups + g]
            immune_vals = baseline_immune_ts[g] if baseline_immune_ts is not None else np.zeros(len(t_eval))
            baseline_cum += np.maximum(0, r_vals - immune_vals)
        ax.plot(t_eval, baseline_cum, label="No Vaccine Baseline",
                color="#95a5a6", linewidth=2.5, linestyle="--", alpha=0.7)
    else:
        I_idx = comp_names.index("I")
        baseline_cum = np.zeros(len(t_eval))
        for g in range(n_groups):
            I_vals = result["baseline"][I_idx * n_groups + g]
            baseline_cum += np.cumsum(np.maximum(np.diff(I_vals, prepend=I_vals[0]), 0))
        ax.plot(t_eval, baseline_cum, label="No Vaccine Baseline",
                color="#95a5a6", linewidth=2.5, linestyle="--", alpha=0.7)

    for strat_name, strat_data in result["strategies"].items():
        color = strategy_colors.get(strat_name, "#95a5a6")
        label = strategy_labels.get(strat_name, strat_name)
        immune_ts = strategy_immune_ts.get(strat_name, np.zeros((n_groups, len(t_eval))))

        if "R" in comp_names:
            R_idx = comp_names.index("R")
            cum_vals = np.zeros(len(t_eval))
            for g in range(n_groups):
                r_vals = strat_data[R_idx * n_groups + g]
                immune_vals = immune_ts[g]
                cum_vals += np.maximum(0, r_vals - immune_vals)
        else:
            I_idx = comp_names.index("I")
            cum_vals = np.zeros(len(t_eval))
            for g in range(n_groups):
                I_vals = strat_data[I_idx * n_groups + g]
                cum_vals += np.cumsum(np.maximum(np.diff(I_vals, prepend=I_vals[0]), 0))

        ax.plot(t_eval, cum_vals, label=label, color=color, linewidth=2.5, alpha=0.9)

    ax.set_xlabel("Days", fontsize=12)
    ax.set_ylabel("Cumulative Infections", fontsize=12)
    ax.set_title("Vaccine Allocation Strategy Comparison \u2014 Cumulative Infections",
                 fontweight="bold", fontsize=14)
    ax.legend(loc="lower right", frameon=True, fancybox=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_alpha(0.3)
    ax.spines["right"].set_alpha(0.3)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    fig.tight_layout()
    return fig


def plot_vaccine_radar(metrics):
    strategy_colors = {
        "uniform": "#3498db",
        "elderly_priority": "#e74c3c",
        "high_contact": "#2ecc71",
        "custom": "#f39c12",
    }
    strategy_labels = {
        "uniform": "Uniform",
        "elderly_priority": "Elderly Priority",
        "high_contact": "High-Contact Priority",
        "custom": "Custom",
    }

    categories = [
        "Infection\nReduction %",
        "Peak\nReduction %",
        "Peak Delay\n(days)",
        "Vaccine\nEfficiency",
        "Fairness\nIndex",
    ]
    n_cats = len(categories)

    all_raw_values = {}
    for strat_name, strat_metrics in metrics["strategies"].items():
        raw = [
            strat_metrics["cum_reduction_pct"],
            strat_metrics["peak_reduction_pct"],
            max(strat_metrics["peak_delay"], 0),
            strat_metrics["vacc_efficiency"],
            strat_metrics["fairness"] * 100,
        ]
        all_raw_values[strat_name] = raw

    max_vals = [0.01] * n_cats
    for strat_name, raw in all_raw_values.items():
        for i in range(n_cats):
            if raw[i] > max_vals[i]:
                max_vals[i] = raw[i]

    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for strat_name, raw in all_raw_values.items():
        normalized = [r / m if m > 0 else 0 for r, m in zip(raw, max_vals)]
        values = normalized + normalized[:1]
        color = strategy_colors.get(strat_name, "#95a5a6")
        label = strategy_labels.get(strat_name, strat_name)

        ax.plot(angles, values, "o-", linewidth=2.5, label=label, color=color, alpha=0.9)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8, alpha=0.6)
    ax.set_title("Vaccine Strategy Radar Comparison\n(Normalized to Best Strategy = 100%)",
                 fontweight="bold", fontsize=13, y=1.12)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), frameon=True, fancybox=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf


HEALTHCARE_COLORS = {
    "bed": "#3498db",
    "icu": "#e74c3c",
    "ventilator": "#9b59b6",
}

ALERT_COLORS = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e67e22",
    3: "#c0392b",
}

ALERT_LABELS = {
    0: "Green - Safe",
    1: "Yellow - Watch",
    2: "Orange - Warning",
    3: "Red - Critical",
}


def plot_healthcare_occupancy(healthcare_result, title="Healthcare Resource Occupancy"):
    flow = healthcare_result["flow"]
    config = healthcare_result["config"]
    days = flow["days"]

    fig, ax = plt.subplots(figsize=(12, 7))

    bed_occ = flow["bed"]["occupied"]
    icu_occ = flow["icu"]["occupied"]
    vent_occ = flow["ventilator"]["occupied"]

    ax.fill_between(days, 0, bed_occ, alpha=0.7,
                    label=f"General Beds ({flow['bed']['capacity']:,})",
                    color=HEALTHCARE_COLORS["bed"], edgecolor='white', linewidth=0.5)
    ax.fill_between(days, bed_occ, bed_occ + icu_occ, alpha=0.7,
                    label=f"ICU Beds ({flow['icu']['capacity']:,})",
                    color=HEALTHCARE_COLORS["icu"], edgecolor='white', linewidth=0.5)
    ax.fill_between(days, bed_occ + icu_occ, bed_occ + icu_occ + vent_occ, alpha=0.7,
                    label=f"Ventilators ({flow['ventilator']['capacity']:,})",
                    color=HEALTHCARE_COLORS["ventilator"], edgecolor='white', linewidth=0.5)

    total_capacity = (flow["bed"]["capacity"] +
                      flow["icu"]["capacity"] +
                      flow["ventilator"]["capacity"])

    bed_cap_line = flow["bed"]["capacity"]
    icu_cap_line = flow["bed"]["capacity"] + flow["icu"]["capacity"]
    vent_cap_line = total_capacity

    ax.axhline(y=bed_cap_line, color=HEALTHCARE_COLORS["bed"],
               linestyle="--", linewidth=2, alpha=0.8, label="Bed Capacity Line")
    ax.axhline(y=icu_cap_line, color=HEALTHCARE_COLORS["icu"],
               linestyle="--", linewidth=2, alpha=0.8, label="ICU Capacity Line")
    ax.axhline(y=vent_cap_line, color=HEALTHCARE_COLORS["ventilator"],
               linestyle="--", linewidth=2, alpha=0.8, label="Ventilator Capacity Line")

    ax.set_xlabel("Days", fontsize=12)
    ax.set_ylabel("Occupied Resources", fontsize=12)
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.legend(loc="upper left", frameon=True, fancybox=True, fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_alpha(0.3)
    ax.spines["right"].set_alpha(0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def plot_alert_timeline(healthcare_result, title="Alert Level Timeline"):
    alerts = healthcare_result["alerts"]
    days = healthcare_result["flow"]["days"]
    n_days = len(days)

    layers = ["ventilator", "icu", "bed"]
    layer_labels = ["Ventilator", "ICU", "General Bed"]
    y_positions = list(range(len(layers)))

    fig, ax = plt.subplots(figsize=(12, max(3, len(layers) * 0.8 + 1.5)))

    for i, layer in enumerate(layers):
        levels = alerts[layer]["levels"]
        y = y_positions[i]

        start_day = 0
        current_level = levels[0] if n_days > 0 else 0

        for day in range(1, n_days):
            if levels[day] != current_level:
                ax.barh(y, day - start_day, left=start_day,
                        height=0.6, color=ALERT_COLORS[current_level],
                        alpha=0.85, edgecolor="white", linewidth=1)
                start_day = day
                current_level = levels[day]

        if start_day < n_days:
            ax.barh(y, n_days - start_day, left=start_day,
                    height=0.6, color=ALERT_COLORS[current_level],
                    alpha=0.85, edgecolor="white", linewidth=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(layer_labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Days", fontsize=12)
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.set_xlim(0, n_days)
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")
    ax.spines["top"].set_alpha(0.3)
    ax.spines["right"].set_alpha(0.3)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ALERT_COLORS[0], alpha=0.85, label=ALERT_LABELS[0]),
        plt.Rectangle((0, 0), 1, 1, facecolor=ALERT_COLORS[1], alpha=0.85, label=ALERT_LABELS[1]),
        plt.Rectangle((0, 0), 1, 1, facecolor=ALERT_COLORS[2], alpha=0.85, label=ALERT_LABELS[2]),
        plt.Rectangle((0, 0), 1, 1, facecolor=ALERT_COLORS[3], alpha=0.85, label=ALERT_LABELS[3]),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              frameon=True, fancybox=True, fontsize=9, ncol=2)

    fig.tight_layout()
    return fig


def plot_mortality_comparison(healthcare_result, title="Mortality Comparison: Ideal vs Actual"):
    mortality = healthcare_result["mortality"]
    days = healthcare_result["flow"]["days"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(days, mortality["ideal_cumulative"],
            label="Ideal (Unlimited Resources)",
            color="#2ecc71", linewidth=2.5, linestyle="--", alpha=0.9)
    ax.plot(days, mortality["actual_cumulative"],
            label="Actual (Resource Constrained)",
            color="#e74c3c", linewidth=2.5, alpha=0.9)

    ax.fill_between(days, mortality["ideal_cumulative"], mortality["actual_cumulative"],
                    where=mortality["actual_cumulative"] >= mortality["ideal_cumulative"],
                    alpha=0.2, color="#e74c3c", interpolate=True, label="Excess Deaths")

    ax.set_xlabel("Days", fontsize=12)
    ax.set_ylabel("Cumulative Deaths", fontsize=12)
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.legend(loc="upper left", frameon=True, fancybox=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_alpha(0.3)
    ax.spines["right"].set_alpha(0.3)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    fig.tight_layout()
    return fig


def plot_healthcare_summary_cards(healthcare_result):
    summary = healthcare_result["summary"]
    alerts = healthcare_result["alerts"]
    flow = healthcare_result["flow"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    card_data = [
        ("Excess Deaths", f"{summary['excess_deaths']:,.0f}",
         f"+{summary['excess_pct']:.1f}% vs ideal", "#e74c3c"),
        ("Resource Gap Peak Days", f"{int(summary['resource_gap_peak_days'])} days",
         "Max duration of capacity overflow", "#f39c12"),
        ("First Red Alert",
         f"Day {int(summary['first_red_alert_day'])}" if summary['first_red_alert_day'] is not None else "Never",
         "Earliest red alert across all layers", "#c0392b"),
        ("Total Capacity",
         f"{flow['bed']['capacity'] + flow['icu']['capacity'] + flow['ventilator']['capacity']:,}",
         "Beds + ICU + Ventilators", "#3498db"),
    ]

    for i, (title, value, subtitle, color) in enumerate(card_data):
        ax = axes[i]
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("white")

        ax.text(0.5, 0.70, title, transform=ax.transAxes,
                fontsize=14, fontweight="bold", color=color,
                ha="center", va="center")
        ax.text(0.5, 0.45, value, transform=ax.transAxes,
                fontsize=22, fontweight="bold", color="#2c3e50",
                ha="center", va="center")
        ax.text(0.5, 0.20, subtitle, transform=ax.transAxes,
                fontsize=10, color="#7f8c8d",
                ha="center", va="center")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        rect = FancyBboxPatch((0.02, 0.05), 0.96, 0.90,
                              boxstyle="round,pad=0.02",
                              linewidth=2, edgecolor=color, facecolor="white",
                              transform=ax.transAxes, alpha=0.9)
        ax.add_patch(rect)

    fig.suptitle("Healthcare System Stress Summary",
                 fontweight="bold", fontsize=16, y=0.98)
    fig.tight_layout()
    return fig
