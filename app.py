import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import (
    run_model, compute_R0_basic, compute_R0_age_stratified,
    get_compartment_names, run_age_stratified_model,
    run_vaccine_strategy_model, compute_vaccine_metrics,
)
from interventions import build_intervention_ode, build_age_intervention_ode
from spatial import run_metapopulation_model, create_default_commuting_matrix
from analysis import (
    estimate_rt, compute_daily_new_infections,
    run_sensitivity_analysis, run_monte_carlo_ci,
    run_optimal_timing_analysis,
)
from visualization import (
    plot_epidemic_curve, plot_epidemic_curve_with_ci,
    plot_rt_curve, plot_intervention_timeline,
    plot_spatial_heatmap, plot_spatial_curves,
    plot_scenario_comparison, plot_scenario_metrics_bar,
    plot_sensitivity_scatter, plot_optimal_timing,
    plot_R0_indicator, fig_to_bytes,
    plot_vaccine_cumulative_comparison, plot_vaccine_radar,
    plot_healthcare_occupancy, plot_alert_timeline,
    plot_mortality_comparison, plot_healthcare_summary_cards,
)
from healthcare import (
    run_healthcare_simulation, compute_daily_new_infections_from_I,
    compute_admission_rates, BASELINE_MORTALITY_RATE,
)

st.set_page_config(
    page_title="Epidemic Dynamics Modeling & Intervention Evaluation",
    page_icon="🦠",
    layout="wide",
)

DEFAULT_CONTACT_MATRIX = np.array([
    [8.0, 2.0, 1.5, 1.0],
    [2.0, 10.0, 3.0, 1.5],
    [1.5, 3.0, 7.0, 2.5],
    [1.0, 1.5, 2.5, 5.0],
])

AGE_GROUPS = ["0-17", "18-44", "45-64", "65+"]
DEFAULT_AGE_PROPS = [0.20, 0.35, 0.30, 0.15]


def init_session_state():
    defaults = {
        "scenarios": [],
        "scenario_results": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def sidebar_model_selection():
    st.sidebar.header("🎯 Model Selection")
    model_type = st.sidebar.selectbox(
        "Compartment Model", ["SIR", "SEIR", "SEIRS", "SIS"],
        help="Choose the epidemic compartment model:\n\n"
             "SIR: Susceptible → Infected → Recovered\n"
             "SEIR: Susceptible → Exposed → Infected → Recovered\n"
             "SEIRS: S→E→I→R with waning immunity back to S\n"
             "SIS: Susceptible → Infected → Susceptible (no lasting immunity)"
    )
    model_descriptions = {
        "SIR": "Standard epidemic model with permanent recovery immunity.",
        "SEIR": "Adds latent (Exposed) period for diseases with incubation time.",
        "SEIRS": "SEIR with waning immunity - recovered individuals become susceptible again.",
        "SIS": "Diseases without lasting immunity - recovered return to susceptible pool.",
    }
    st.sidebar.info(f"📝 {model_type}: {model_descriptions[model_type]}")
    return model_type


def sidebar_parameters(model_type):
    st.sidebar.header("⚙️ Core Parameters")

    beta = _dual_control(st.sidebar, "Infection Rate (β)", 0.3, 0.01, 2.0, 0.01, "beta",
                         help="Daily probability of transmission per contact")
    gamma = _dual_control(st.sidebar, "Recovery Rate (γ)", 0.1, 0.01, 1.0, 0.01, "gamma",
                          help="Daily probability of recovering (1/γ = avg infectious period in days)")

    sigma = None
    omega = None

    if model_type in ("SEIR", "SEIRS"):
        sigma = _dual_control(st.sidebar, "Incubation Rate (σ)", 0.3, 0.05, 1.0, 0.01, "sigma",
                              help="Daily rate from Exposed to Infected (1/σ = avg incubation period)")
    if model_type in ("SEIRS", "SIS"):
        omega = _dual_control(st.sidebar, "Waning Immunity Rate (ω)", 0.05, 0.001, 0.5, 0.001, "omega",
                              help="Daily rate of immunity loss (1/ω = avg immunity duration)")

    st.sidebar.header("👥 Initial Conditions")
    N = _dual_control(st.sidebar, "Total Population (N)", 1000000, 10000, 100000000, 10000, "N", fmt="%d",
                      help="Total population size")
    I0 = _dual_control(st.sidebar, "Initial Infected (I₀)", 100, 1, 1000, 1, "I0", fmt="%d",
                       help="Number of initially infected individuals")
    R0_init = _dual_control(st.sidebar, "Initial Recovered (R₀)", 0, 0, max(0, int(N * 0.1)), 1, "R0_init", fmt="%d",
                            help=f"Number of initially immune individuals (max {int(N * 0.1):,} = 10% of N)")

    st.sidebar.header("⏱️ Simulation Settings")
    sim_days = st.sidebar.slider("Simulation Days", 60, 730, 300, 10,
                                 help="Total number of days to simulate")

    params = {"beta": beta, "gamma": gamma, "N": N}
    if sigma is not None:
        params["sigma"] = sigma
    if omega is not None:
        params["omega"] = omega

    return params, N, I0, R0_init, sim_days


def _dual_control(container, label, default, min_val, max_val, step, key, fmt="%.3f", help=None):
    min_val = type(default)(min_val)
    max_val = type(default)(max_val)
    step = type(default)(step)

    col1, col2 = container.columns([3, 2])
    with col1:
        slider_val = col1.slider(label, min_value=min_val, max_value=max_val,
                                 value=default, step=step, key=f"{key}_slider", format=fmt,
                                 help=help, label_visibility="collapsed")
    with col2:
        if not hasattr(st.session_state, f"_input_{key}"):
            st.session_state[f"_input_{key}"] = default
        input_val = col2.number_input("", min_value=min_val, max_value=max_val,
                                      value=slider_val, step=step, key=f"{key}_input", format=fmt,
                                      label_visibility="collapsed")
    return input_val


def section_age_stratification(model_type, params, N, I0, R0_init):
    st.header("👥 Age-Stratified Extension")
    st.markdown("Divide population into age groups with differential contact patterns.")

    enable_age = st.checkbox("✅ Enable Age Stratification", value=False, key="enable_age")

    if not enable_age:
        return None, None, None

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Age Group Proportions")
        age_props = []
        cols = st.columns(4)
        for i, age in enumerate(AGE_GROUPS):
            with cols[i]:
                val = st.number_input(
                    f"**{age}**\n(%)",
                    min_value=0.0, max_value=100.0,
                    value=DEFAULT_AGE_PROPS[i] * 100,
                    key=f"age_prop_{i}",
                    step=0.5,
                ) / 100.0
                age_props.append(val)
        total_prop = sum(age_props)
        if abs(total_prop - 1.0) > 0.01:
            st.warning(f"⚠️ Proportions sum to {total_prop:.2%}. Will be normalized automatically.")
            age_props = [p / total_prop for p in age_props]
        else:
            st.success(f"✅ Proportions sum to {total_prop:.2%}")

    with col2:
        st.subheader("🔗 Contact Matrix (4×4)")
        st.caption("Element (i,j) = average daily contacts between person in group i and group j")
        contact_df = pd.DataFrame(
            DEFAULT_CONTACT_MATRIX,
            index=AGE_GROUPS, columns=AGE_GROUPS
        )
        edited_df = st.data_editor(
            contact_df, key="contact_matrix_editor",
            use_container_width=True,
            column_config={
                col: st.column_config.NumberColumn(col, min_value=0.0, step=0.1)
                for col in AGE_GROUPS
            },
        )
        contact_matrix = edited_df.values.astype(float)

    r0_age = compute_R0_age_stratified(
        params["beta"], params["gamma"], contact_matrix, age_props,
        sigma=params.get("sigma"), model_type=model_type
    )

    st.markdown("---")
    col_r0, col_info = st.columns([1, 2])
    with col_r0:
        r0_color = "green" if r0_age < 1 else ("orange" if r0_age < 2 else "red")
        st.markdown(
            f"### R₀ (Age-Stratified) = "
            f"<span style='color:{r0_color};font-size:1.8em;font-weight:bold'>{r0_age:.3f}</span>",
            unsafe_allow_html=True
        )
        status = "✅ Epidemic will decline" if r0_age < 1 else (
            "⚠️ Moderate growth expected" if r0_age < 2 else "🚨 Rapid growth likely"
        )
        st.markdown(f"**{status}**")
    with col_info:
        fig_r0 = plot_R0_indicator(r0_age)
        st.pyplot(fig_r0)

    return contact_matrix, np.array(age_props), r0_age


def section_spatial(model_type, params, sim_days):
    st.header("🗺️ Spatial Metapopulation Module")
    st.markdown("Multi-region model with population commuting between regions.")

    enable_spatial = st.checkbox("✅ Enable Spatial Model", value=False, key="enable_spatial")

    if not enable_spatial:
        return None

    n_regions = st.slider("Number of Regions", 2, 10, 3, 1, key="n_regions",
                          help="Configure 2 to 10 interconnected regions")

    region_params = []
    region_names = []

    st.subheader("📍 Region Configurations")
    cols = st.columns(min(n_regions, 4))
    for i in range(n_regions):
        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"**🏙️ Region {i+1}**")
                name = st.text_input(f"Name", value=f"Region {i+1}", key=f"region_name_{i}",
                                     label_visibility="collapsed")
                pop = st.number_input(f"Population", value=500000, min_value=1000,
                                     key=f"region_pop_{i}", step=1000)
                i0 = st.number_input(f"Initial Infected", value=max(1, 10 + i * 5), min_value=0,
                                     key=f"region_i0_{i}", step=1)
                region_names.append(name)
                region_params.append({
                    "N": pop, "I0": i0, "R0": 0,
                    "beta": params["beta"], "gamma": params["gamma"],
                    "sigma": params.get("sigma", 0.0), "omega": params.get("omega", 0.0),
                })

    st.subheader("🚗 Commuting Matrix")
    st.caption("Element (i,j) = daily proportion of people traveling from region i to region j.\n"
               "Diagonal (i,i) = proportion staying in region. Each row sums to 1.")
    default_commuting = create_default_commuting_matrix(n_regions)
    commuting_df = pd.DataFrame(
        default_commuting,
        index=region_names, columns=region_names
    )
    edited_commuting = st.data_editor(
        commuting_df, key="commuting_matrix_editor",
        use_container_width=True,
        column_config={
            col: st.column_config.NumberColumn(col, min_value=0.0, max_value=1.0, step=0.01, format="%.3f")
            for col in region_names
        },
    )
    commuting_matrix = edited_commuting.values.astype(float)

    row_sums = commuting_matrix.sum(axis=1)
    row_warnings = []
    for i in range(n_regions):
        if abs(row_sums[i] - 1.0) > 0.05:
            row_warnings.append(f"Row {i+1} ({region_names[i]}) sums to {row_sums[i]:.3f}")
        if abs(row_sums[i] - 1.0) > 0.001:
            commuting_matrix[i] = commuting_matrix[i] / max(row_sums[i], 1e-10)

    if row_warnings:
        st.warning("⚠️ " + "; ".join(row_warnings) + ". Rows auto-normalized to sum to 1.")
    else:
        st.success("✅ All rows sum to 1.0")

    return {
        "n_regions": n_regions,
        "region_params": region_params,
        "commuting_matrix": commuting_matrix,
        "region_names": region_names,
    }


def section_interventions(model_type, params, N):
    st.header("💊 Intervention Strategies")
    st.markdown("Configure and combine multiple public health interventions.")

    enable_interventions = st.checkbox("✅ Enable Interventions", value=False, key="enable_interventions")

    if not enable_interventions:
        return []

    interventions = []

    intervention_types = {
        "quarantine": ("🛡️ Case Isolation/Quarantine",
                        "Remove detected infected individuals from transmission pool"),
        "vaccination": ("💉 Vaccination Campaign",
                         "Vaccinate susceptible population to grant immunity"),
        "social_distance": ("👋 Social Distancing",
                             "Reduce contact rates across all groups"),
        "regional_lockdown": ("🚧 Regional Lockdown",
                               "Restrict inter-regional population movement"),
    }

    for itype, (ilabel, idesc) in intervention_types.items():
        enabled = st.checkbox(f"Enable {ilabel}", value=False, key=f"interv_enable_{itype}")
        if not enabled:
            continue

        with st.container(border=True):
            st.markdown(f"**{ilabel}** — *{idesc}*")
            col1, col2 = st.columns(2)
            with col1:
                start_day = st.number_input(
                    f"🚀 Start Day", min_value=0, max_value=730,
                    value=30, key=f"interv_start_{itype}", step=1,
                    help="Day number to begin the intervention"
                )
                duration = st.number_input(
                    f"📅 Duration (days)", min_value=1, max_value=730,
                    value=30, key=f"interv_duration_{itype}", step=1,
                    help="How many days the intervention lasts"
                )
            with col2:
                if itype == "quarantine":
                    param_val = st.slider("🔄 Daily Quarantine Rate", 0.01, 1.0, 0.1, 0.01,
                                          key=f"interv_qrate",
                                          help="Daily proportion of detected infected isolated")
                    interventions.append({
                        "type": "quarantine", "start_day": start_day,
                        "duration": duration, "q_rate": param_val,
                    })
                elif itype == "vaccination":
                    vacc_rate = st.slider("💉 Daily Vaccination Rate", 0.001, 0.1, 0.01, 0.001,
                                          key=f"interv_vrate",
                                          help="Proportion of susceptible vaccinated daily")
                    efficacy = st.slider("✨ Vaccine Efficacy", 0.0, 1.0, 0.8, 0.05,
                                         key=f"interv_efficacy",
                                         help="Probability vaccination grants immunity")
                    interventions.append({
                        "type": "vaccination", "start_day": start_day,
                        "duration": duration, "vacc_rate": vacc_rate,
                        "efficacy": efficacy,
                    })
                elif itype == "social_distance":
                    reduction = st.slider("📉 Contact Reduction Factor", 0.3, 1.0, 0.5, 0.05,
                                          key=f"interv_sd",
                                          help="Multiply all contacts by this (0.5 = 50% reduction)")
                    interventions.append({
                        "type": "social_distance", "start_day": start_day,
                        "duration": duration, "reduction": reduction,
                    })
                elif itype == "regional_lockdown":
                    lockdown_factor = st.slider("🔒 Movement Restriction (0=full, 1=none)", 0.0, 1.0, 0.1, 0.05,
                                                 key=f"interv_lf",
                                                 help="Multiply off-diagonal commuting by this factor")
                    interventions.append({
                        "type": "regional_lockdown", "start_day": start_day,
                        "duration": duration, "lockdown_factor": lockdown_factor,
                    })

    return interventions


def section_scenario_comparison(model_type, params, N, I0, R0_init, sim_days):
    st.header("🔄 Scenario Comparison Panel")
    st.markdown("Run multiple parameter/strategy scenarios side-by-side for comparison.")

    n_scenarios = st.slider("Number of Scenarios", 2, 4, 2, 1, key="n_scenarios")

    scenario_configs = []
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for i in range(n_scenarios):
        with st.expander(f"🎨 Scenario {i+1}", expanded=(i == 0)):
            col_name, col_color = st.columns([3, 1])
            with col_name:
                name = st.text_input("Scenario Name", value=f"Scenario {i+1}", key=f"scenario_name_{i}")
            with col_color:
                st.color_picker("Color", colors[i], key=f"sc_color_{i}", disabled=True)

            s_beta = st.slider("β Infection Rate", 0.01, 2.0, params["beta"], 0.01, key=f"sc_beta_{i}")
            s_gamma = st.slider("γ Recovery Rate", 0.01, 1.0, params["gamma"], 0.01, key=f"sc_gamma_{i}")
            s_params = {"beta": s_beta, "gamma": s_gamma, "N": N}

            if model_type in ("SEIR", "SEIRS"):
                s_sigma = st.slider("σ Incubation Rate", 0.05, 1.0, params.get("sigma", 0.3), 0.01, key=f"sc_sigma_{i}")
                s_params["sigma"] = s_sigma
            if model_type in ("SEIRS", "SIS"):
                s_omega = st.slider("ω Waning Immunity", 0.001, 0.5, params.get("omega", 0.05), 0.001, key=f"sc_omega_{i}")
                s_params["omega"] = s_omega

            has_interv = st.checkbox("Include Intervention", value=False, key=f"sc_interv_{i}")
            interventions = []
            if has_interv:
                interv_type = st.selectbox(
                    "Intervention Type",
                    ["quarantine", "vaccination", "social_distance", "regional_lockdown"],
                    format_func=lambda x: {
                        "quarantine": "🛡️ Quarantine",
                        "vaccination": "💉 Vaccination",
                        "social_distance": "👋 Social Distancing",
                        "regional_lockdown": "🚧 Regional Lockdown",
                    }.get(x, x),
                    key=f"sc_interv_type_{i}"
                )
                c1, c2 = st.columns(2)
                with c1:
                    start = st.number_input("Start Day", value=30, key=f"sc_interv_start_{i}", step=1)
                    dur = st.number_input("Duration", value=30, key=f"sc_interv_dur_{i}", step=1)
                with c2:
                    interv = {"type": interv_type, "start_day": start, "duration": dur}
                    if interv_type == "quarantine":
                        interv["q_rate"] = st.slider("Q Rate", 0.01, 1.0, 0.1, key=f"sc_qr_{i}")
                    elif interv_type == "vaccination":
                        interv["vacc_rate"] = st.slider("Vacc Rate", 0.001, 0.1, 0.01, key=f"sc_vr_{i}")
                        interv["efficacy"] = st.slider("Efficacy", 0.0, 1.0, 0.8, key=f"sc_eff_{i}")
                    elif interv_type == "social_distance":
                        interv["reduction"] = st.slider("Reduction", 0.3, 1.0, 0.5, key=f"sc_red_{i}")
                    elif interv_type == "regional_lockdown":
                        interv["lockdown_factor"] = st.slider("Lockdown Factor", 0.0, 1.0, 0.1, key=f"sc_lf_{i}")

                interventions = [interv]

            scenario_configs.append({
                "name": name, "params": s_params,
                "interventions": interventions,
            })

    return scenario_configs


def section_sensitivity(model_type, params, N, I0, R0_init, t_span):
    st.header("🔬 Sensitivity & Uncertainty Analysis")
    st.markdown("Explore how parameter uncertainty affects model outputs using LHS and Monte Carlo.")

    param_options = list(params.keys())
    if "N" in param_options:
        param_options.remove("N")

    if len(param_options) < 2:
        st.warning("⚠️ Need at least 2 variable parameters for sensitivity analysis.")
        return

    col1, col2 = st.columns(2)
    with col1:
        p1_name = st.selectbox("📌 Parameter 1", param_options, index=0, key="sens_p1")
    with col2:
        remaining = [p for p in param_options if p != p1_name]
        p2_name = st.selectbox("📌 Parameter 2", remaining, index=0 if remaining else 0, key="sens_p2")

    n_samples = st.slider("🔢 LHS Sample Size", 50, 500, 100, 10, key="lhs_samples",
                          help="Number of Latin Hypercube samples (more = more accurate, slower)")

    param_ranges = {}
    for pname in [p1_name, p2_name]:
        if pname == "beta":
            param_ranges["beta"] = (0.01, 2.0)
        elif pname == "gamma":
            param_ranges["gamma"] = (0.01, 1.0)
        elif pname == "sigma":
            param_ranges["sigma"] = (0.05, 1.0)
        elif pname == "omega":
            param_ranges["omega"] = (0.001, 0.5)

    if st.button("🚀 Run Sensitivity Analysis", type="primary", key="run_sens"):
        with st.spinner(f"Running {n_samples} LHS simulations... This may take a moment."):
            result = run_sensitivity_analysis(
                model_type, param_ranges, params, N, I0, R0_init, t_span,
                n_samples=n_samples
            )

            valid = ~np.isnan(result["cumulative_infections"])
            valid_count = valid.sum()
            st.info(f"✅ {valid_count}/{n_samples} valid simulations completed")

            if valid_count > 1:
                col_a, col_b = st.columns(2)

                with col_a:
                    fig_cum = plot_sensitivity_scatter(
                        result["samples"][valid], result["param_names"],
                        result["cumulative_infections"][valid],
                        result["prcc_cumulative"],
                        "Cumulative Infections"
                    )
                    st.pyplot(fig_cum)
                    _download_button(fig_cum, "sensitivity_cumulative.png")

                with col_b:
                    fig_peak = plot_sensitivity_scatter(
                        result["samples"][valid], result["param_names"],
                        result["peak_times"][valid],
                        result["prcc_peak_time"],
                        "Peak Time (days)"
                    )
                    st.pyplot(fig_peak)
                    _download_button(fig_peak, "sensitivity_peak_time.png")

                st.subheader("📋 PRCC Results Table")
                prcc_df = pd.DataFrame({
                    "Parameter": result["param_names"],
                    "PRCC (Cumulative Infections)": result["prcc_cumulative"],
                    "PRCC (Peak Time)": result["prcc_peak_time"],
                    "Significant (|PRCC| > 0.5)": [
                        "✅ Yes" if abs(v) > 0.5 else "❌ No"
                        for v in result["prcc_cumulative"]
                    ],
                    "Interpretation": [
                        "Strong positive" if v > 0.5 else (
                            "Strong negative" if v < -0.5 else (
                                "Moderate" if abs(v) > 0.2 else "Weak"
                            )
                        )
                        for v in result["prcc_cumulative"]
                    ],
                })
                st.dataframe(prcc_df, use_container_width=True, hide_index=True)
            else:
                st.error("❌ Too few valid simulations. Please check parameter ranges.")

    st.markdown("---")
    st.subheader("🎲 Monte Carlo Uncertainty (95% CI)")
    st.markdown("Generate 95% confidence intervals around the epidemic curve using parameter perturbation.")

    mc_ranges = {}
    for pname in [p1_name, p2_name]:
        base_val = params.get(pname, 0.1)
        mc_ranges[pname] = (base_val * 0.8, base_val * 1.2)

    n_mc = st.slider("🔢 Monte Carlo Runs", 20, 200, 50, 10, key="mc_runs",
                     help="Number of Monte Carlo samples")

    if st.button("🚀 Run Monte Carlo CI", type="primary", key="run_mc"):
        with st.spinner(f"Running {n_mc} Monte Carlo simulations..."):
            ci_result = run_monte_carlo_ci(
                model_type, params, N, I0, R0_init, t_span,
                mc_ranges, n_runs=n_mc
            )
            sol_base = run_model(model_type, params, N, I0, R0_init, t_span)
            fig = plot_epidemic_curve_with_ci(sol_base, model_type, ci_result)
            st.pyplot(fig)
            _download_button(fig, "monte_carlo_ci.png")
            st.caption("Shaded regions represent 95% confidence intervals from parameter uncertainty.")


def section_vaccine_allocation(model_type, params, N, I0, R0_init, contact_matrix, age_props):
    st.header("💉 Vaccine Allocation Strategy Optimization")
    st.markdown(
        "Compare the impact of different vaccine allocation strategies under limited supply "
        "using the age-stratified model. **Age Stratification must be enabled** in the "
        "Age Stratification tab with a configured contact matrix."
    )

    if contact_matrix is None or age_props is None:
        st.warning(
            "⚠️ **Age Stratification is not enabled.** Please go to the "
            "**👥 Age Stratification** tab, enable it, and configure the contact matrix "
            "before using this module."
        )
        return

    st.subheader("💉 Vaccine Supply Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        max_vacc = int(N * 0.5)
        total_vaccines = st.number_input(
            "💉 Total Vaccine Doses",
            min_value=1000, max_value=max_vacc,
            value=min(100000, max_vacc), step=1000,
            key="vacc_total_doses",
            help=f"Total available vaccine doses (1000 to {max_vacc:,} = 50% of N)"
        )
    with col2:
        vacc_days = st.number_input(
            "📅 Vaccination Period (days)",
            min_value=7, max_value=180,
            value=30, step=1,
            key="vacc_period_days",
            help="Number of days over which vaccines are administered (7-180)"
        )
    with col3:
        efficacy = st.slider(
            "✨ Vaccine Efficacy",
            min_value=0.0, max_value=1.0,
            value=0.8, step=0.05,
            key="vacc_efficacy",
            help="Proportion of vaccinated individuals who gain immunity"
        )

    daily_max = total_vaccines / vacc_days
    st.info(
        f"📊 **Daily vaccination capacity:** {daily_max:,.1f} doses/day  |  "
        f"**Effective immune conversions:** {daily_max * efficacy:,.1f} people/day"
    )

    st.markdown("---")
    st.subheader("🎛️ Custom Allocation Proportions")
    st.markdown("Specify the percentage of vaccines allocated to each age group (must sum to 100%).")

    custom_sliders = []
    custom_cols = st.columns(4)
    default_custom = [25, 25, 25, 25]
    for i, age in enumerate(AGE_GROUPS):
        with custom_cols[i]:
            val = st.slider(
                f"**{age}** (%)",
                min_value=0, max_value=100,
                value=default_custom[i], step=1,
                key=f"vacc_custom_{i}"
            )
            custom_sliders.append(val)

    custom_sum = sum(custom_sliders)
    custom_props = [v / 100.0 for v in custom_sliders]
    if abs(custom_sum - 100) > 1:
        st.warning(f"⚠️ Custom proportions sum to {custom_sum}%. Will be normalized automatically.")
        custom_props = [p / (custom_sum / 100.0) for p in custom_props]
    else:
        st.success(f"✅ Custom proportions sum to {custom_sum}%")

    st.markdown("---")

    strategy_labels = {
        "uniform": "🔵 Uniform (by population proportion)",
        "elderly_priority": "🔴 Elderly Priority (65+ first)",
        "high_contact": "🟢 High-Contact Priority (by contact rate)",
        "custom": "🟠 Custom Proportions (user-defined)",
    }
    st.subheader("📋 Strategy Descriptions")
    for strat, desc in strategy_labels.items():
        st.markdown(f"- {desc}")

    if st.button("🚀 Run Vaccine Strategy Comparison", type="primary", key="run_vaccine_strat"):
        with st.spinner("Running 4 vaccine allocation strategies + baseline (300 days each)... This may take a minute."):
            result = run_vaccine_strategy_model(
                model_type, params, contact_matrix, age_props,
                N, I0, R0_init, total_vaccines, vacc_days, efficacy,
                custom_props=custom_props, sim_days=300
            )

            metrics = compute_vaccine_metrics(result, N, total_vaccines)

            st.subheader("📈 Cumulative Infection Curves Comparison")
            fig_cum = plot_vaccine_cumulative_comparison(result, metrics, AGE_GROUPS)
            st.pyplot(fig_cum, use_container_width=True)
            _download_button(fig_cum, "vaccine_cumulative_comparison.png")

            st.subheader("📋 Strategy Metrics Table")
            strat_display = {
                "uniform": "Uniform",
                "elderly_priority": "Elderly Priority",
                "high_contact": "High-Contact Priority",
                "custom": "Custom",
            }

            table_data = []
            for strat_name, strat_metrics in metrics["strategies"].items():
                actual_doses = strat_metrics.get("actual_doses", total_vaccines)
                table_data.append({
                    "Strategy": strat_display.get(strat_name, strat_name),
                    "Final Cumulative Infections": f"{strat_metrics['cum_infections']:,.0f}",
                    "Infection Reduction %": f"{strat_metrics['cum_reduction_pct']:.2f}%",
                    "Peak Delay (days)": f"{max(0, strat_metrics['peak_delay']):.1f}",
                    "Peak Reduction %": f"{strat_metrics['peak_reduction_pct']:.2f}%",
                    "Vaccine Efficiency\n(infections avoided/dose)": f"{strat_metrics['vacc_efficiency']:.4f}",
                    "Fairness Index": f"{strat_metrics['fairness']:.4f}",
                    "Actual Doses Used": f"{actual_doses:,.0f}",
                })

            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

            st.caption(
                f"Baseline (no vaccine) cumulative infections: {metrics['baseline_cum']:,.0f}  |  "
                f"Baseline peak: {metrics['baseline_peak']:,.0f} at Day {metrics['baseline_peak_day']:.0f}"
            )

            st.subheader("🕸️ Strategy Radar Comparison")
            fig_radar = plot_vaccine_radar(metrics)
            st.pyplot(fig_radar, use_container_width=True)
            _download_button(fig_radar, "vaccine_radar.png")

            with st.expander("📊 Detailed Age-Group Attack Rates by Strategy"):
                for strat_name, strat_metrics in metrics["strategies"].items():
                    st.markdown(f"**{strat_display.get(strat_name, strat_name)}**")
                    attack_data = {}
                    for g in range(len(AGE_GROUPS)):
                        attack_data[AGE_GROUPS[g]] = {
                            "Attack Rate": f"{strat_metrics['attack_rates'][g]:.4f}",
                            "Attack Rate %": f"{strat_metrics['attack_rates'][g] * 100:.2f}%",
                        }
                    st.dataframe(pd.DataFrame(attack_data), use_container_width=True)
                    st.markdown("")


def section_optimal_timing(model_type, params, N, I0, R0_init, t_span):
    st.header("⏱️ Optimal Intervention Timing Analysis")
    st.markdown("Find the best day to start an intervention to minimize total infections.")

    col1, col2 = st.columns(2)
    with col1:
        interv_type = st.selectbox(
            "🎯 Intervention Type",
            ["social_distance", "quarantine", "vaccination", "regional_lockdown"],
            format_func=lambda x: {
                "quarantine": "🛡️ Quarantine",
                "vaccination": "💉 Vaccination",
                "social_distance": "👋 Social Distancing",
                "regional_lockdown": "🚧 Regional Lockdown",
            }.get(x, x),
            key="opt_interv_type"
        )
        duration = st.number_input("📅 Intervention Duration (days)",
                                    value=30, min_value=7, max_value=365, step=1, key="opt_dur")
    with col2:
        if interv_type == "social_distance":
            interv_param = st.slider("📉 Contact Reduction", 0.3, 1.0, 0.5, 0.05, key="opt_sd")
            interv_config = {"type": "social_distance", "duration": duration, "reduction": interv_param}
        elif interv_type == "quarantine":
            interv_param = st.slider("🔄 Quarantine Rate", 0.01, 1.0, 0.1, key="opt_qr")
            interv_config = {"type": "quarantine", "duration": duration, "q_rate": interv_param}
        elif interv_type == "vaccination":
            vr = st.slider("💉 Vacc Rate", 0.001, 0.1, 0.01, key="opt_vr")
            eff = st.slider("✨ Efficacy", 0.0, 1.0, 0.8, key="opt_eff")
            interv_config = {"type": "vaccination", "duration": duration, "vacc_rate": vr, "efficacy": eff}
        elif interv_type == "regional_lockdown":
            interv_param = st.slider("🔒 Lockdown Factor", 0.0, 1.0, 0.1, key="opt_lf")
            interv_config = {"type": "regional_lockdown", "duration": duration, "lockdown_factor": interv_param}

    max_start = st.slider("⏳ Max Start Day to Test", 1, 120, 60, 5, key="opt_max_start",
                          help="Test intervention start from Day 1 to this day")

    if st.button("🚀 Run Timing Analysis", type="primary", key="run_timing"):
        with st.spinner(f"Testing {max_start} intervention start days..."):
            result = run_optimal_timing_analysis(
                model_type, params, N, I0, R0_init, t_span,
                interv_config, max_start_day=max_start
            )
            fig = plot_optimal_timing(result)
            st.pyplot(fig)
            _download_button(fig, "optimal_timing.png")

            col_best, col_window = st.columns(2)
            with col_best:
                st.success(f"**🎯 Best Start Day:** Day {result['best_start_day']}")
            with col_window:
                if len(result["optimal_window"]) > 0:
                    win_start = result["optimal_window"][0]
                    win_end = result["optimal_window"][-1]
                    if win_end > win_start:
                        st.info(f"**🪟 Optimal Window:** Day {win_start} to Day {win_end}")
                    else:
                        st.info(f"**🪟 Optimal Window:** Around Day {win_start}")

            with st.expander("💡 Interpretation Guide", expanded=True):
                st.markdown("""
                **Key Insights:**
                - **Early intervention** (Day 1-10): May suppress epidemic initially, but lifting too early can cause
                  a large rebound if the susceptible pool is still large. Useful when case counts must stay low.
                - **Optimal intervention**: Balances between delaying the epidemic long enough to build capacity
                  while minimizing total infections. Typically starts just before or at the early exponential phase.
                - **Late intervention** (after Day 40): Limited effect since most infections have already occurred.
                  May still reduce peak but cumulative impact is minimal.
                - The **optimal window** indicates the range of start days where cumulative infections are within
                  5% of the absolute minimum.
                """)


def section_rt_estimation(sol, model_type):
    st.header("📈 R_t Real-time Effective Reproduction Number")
    st.markdown("Estimate the time-varying reproduction number from simulated incidence data.")

    comp_names = get_compartment_names(model_type)

    daily_new = compute_daily_new_infections(sol.y, comp_names)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        mean_serial = st.slider("📊 Mean Serial Interval (days)", 2.0, 10.0, 5.0, 0.5, key="rt_mean")
    with col_b:
        std_serial = st.slider("📉 Std Serial Interval (days)", 0.5, 5.0, 2.0, 0.25, key="rt_std")
    with col_c:
        st.caption("Gamma-distributed generation interval")

    rt_times, rt_values = estimate_rt(daily_new, mean_serial=mean_serial, std_serial=std_serial)

    if len(rt_times) > 0:
        fig = plot_rt_curve(rt_times, rt_values)
        st.pyplot(fig)
        _download_button(fig, "rt_curve.png")

        smoothed_rt = rt_values
        if len(smoothed_rt) > 7:
            smoothed_rt = np.convolve(smoothed_rt, np.ones(7)/7, mode='same')
        rt_below_1 = np.where(smoothed_rt < 0.995)[0]
        if len(rt_below_1) > 0:
            day_below_1 = rt_times[rt_below_1[0]]
            st.success(f"**✅ R_t drops below 1 on Day {int(day_below_1)}**")

            after_day = day_below_1 + 14
            if after_day < sol.t[-1]:
                st.info(f"💡 If R_t remains below 1, epidemic effectively ends ~Day {int(after_day)}")
        else:
            st.error("**⚠️ R_t never drops below 1** in this simulation. Consider stronger interventions.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Peak R_t", f"{np.max(rt_values):.2f}")
        with col2:
            st.metric("📉 Final R_t", f"{rt_values[-1]:.2f}")
        with col3:
            rt_mean = np.mean(rt_values[-min(14, len(rt_values)):])
            st.metric("📊 14-Day Avg R_t", f"{rt_mean:.2f}")
    else:
        st.warning("⚠️ Insufficient data for R_t estimation. Try longer simulation.")


def _download_button(fig, filename):
    buf = fig_to_bytes(fig)
    st.download_button(
        label=f"📥 Download PNG",
        data=buf, file_name=filename, mime="image/png",
        key=f"dl_{filename}_{id(fig)}",
        use_container_width=False,
    )


def compute_scenario_metrics(sol, model_type):
    comp_names = get_compartment_names(model_type)
    I_idx = comp_names.index("I")

    I_vals = sol.y[I_idx]
    peak_infections = np.max(I_vals)
    peak_time = sol.t[np.argmax(I_vals)]

    if "R" in comp_names:
        R_idx = comp_names.index("R")
        cumulative = sol.y[R_idx][-1]
    else:
        cumulative = np.sum(np.diff(I_vals, prepend=0))

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

    return {
        "cumulative": cumulative,
        "peak": peak_infections,
        "peak_time": peak_time,
        "rt_below_1": rt_below_1_day,
    }


def main():
    init_session_state()

    st.title("🦠 Epidemic Dynamics Modeling & Intervention Evaluation Platform")
    st.markdown(
        "A comprehensive tool for **public health researchers** to build compartment models, "
        "simulate epidemic trajectories, and quantitatively evaluate intervention strategies."
    )

    model_type = sidebar_model_selection()
    params, N, I0, R0_init, sim_days = sidebar_parameters(model_type)

    r0_basic = compute_R0_basic(
        model_type, params["beta"], params["gamma"],
        sigma=params.get("sigma"), omega=params.get("omega")
    )

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        fig_r0 = plot_R0_indicator(r0_basic)
        st.pyplot(fig_r0, use_container_width=False)
    with col2:
        st.metric("👥 Total Population", f"{N:,}")
    with col3:
        st.metric("🦠 Initial Infected", f"{I0:,}")
    with col4:
        dur_infect = 1 / params["gamma"] if params["gamma"] > 0 else float("inf")
        st.metric("⏰ Infectious Period", f"{dur_infect:.1f} days")

    tabs = st.tabs([
        "📊 Epidemic Curve",
        "👥 Age Stratification",
        "🗺️ Spatial Model",
        "💊 Interventions",
        "🔄 Scenario Comparison",
        "📈 R_t Estimation",
        "🔬 Sensitivity Analysis",
        "⏱️ Optimal Timing",
        "💉 Vaccine Allocation",
        "🏥 Healthcare Stress",
    ])

    t_span = (0, sim_days)
    t_eval = np.linspace(0, sim_days, sim_days + 1)

    vaccine_contact_matrix = None
    vaccine_age_props = None
    if st.session_state.get("enable_age", False):
        vaccine_age_props = []
        for i in range(4):
            val = st.session_state.get(f"age_prop_{i}", DEFAULT_AGE_PROPS[i] * 100) / 100.0
            vaccine_age_props.append(val)
        total_prop = sum(vaccine_age_props)
        if abs(total_prop - 1.0) > 0.01:
            vaccine_age_props = [p / total_prop for p in vaccine_age_props]
        vaccine_age_props = np.array(vaccine_age_props)

        contact_df = pd.DataFrame(
            DEFAULT_CONTACT_MATRIX,
            index=AGE_GROUPS, columns=AGE_GROUPS
        )
        if "contact_matrix_editor" in st.session_state:
            edited = st.session_state["contact_matrix_editor"]
            if isinstance(edited, pd.DataFrame):
                vaccine_contact_matrix = edited.values.astype(float)
            else:
                vaccine_contact_matrix = DEFAULT_CONTACT_MATRIX.copy()
        else:
            vaccine_contact_matrix = DEFAULT_CONTACT_MATRIX.copy()

    with tabs[0]:
        st.header("📊 Baseline Epidemic Curve")

        chart_type = st.radio("📊 Chart Display Mode",
                              ["📈 Line Chart", "📊 Stacked Area"],
                              horizontal=True, key="chart_type")

        sol = run_model(model_type, params, N, I0, R0_init, t_span, t_eval=t_eval)
        ct = "stacked" if "Stacked" in chart_type else "line"
        fig = plot_epidemic_curve(sol, model_type, chart_type=ct)
        st.pyplot(fig, use_container_width=True)
        _download_button(fig, "epidemic_curve.png")

        st.subheader("📋 Compartment Summary")
        comp_names = get_compartment_names(model_type)
        summary_data = {}
        for i, name in enumerate(comp_names):
            final_val = sol.y[i][-1]
            pct_final = final_val / N * 100 if N > 0 else 0
            summary_data[name] = {
                "Initial": f"{sol.y[i][0]:,.0f}",
                "Final": f"{final_val:,.0f} ({pct_final:.1f}%)",
                "Peak": f"{np.max(sol.y[i]):,.0f}",
                "Peak Day": f"{sol.t[np.argmax(sol.y[i])]:.0f}",
            }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        attack_rate = 0
        if "R" in comp_names:
            attack_rate = sol.y[comp_names.index("R")][-1] / N * 100 if N > 0 else 0
        elif len(comp_names) == 2:
            daily_cases = compute_daily_new_infections(sol.y, comp_names)
            attack_rate = np.sum(daily_cases) / N * 100 if N > 0 else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🎯 Final Attack Rate", f"{attack_rate:.2f}%")
        with c2:
            I_idx = comp_names.index("I")
            peak_val = np.max(sol.y[I_idx])
            st.metric("📈 Peak Infected", f"{peak_val:,.0f} ({peak_val/N*100:.2f}%)")
        with c3:
            st.metric("📅 Day of Peak", f"Day {sol.t[np.argmax(sol.y[I_idx])]:.0f}")

    with tabs[1]:
        contact_matrix, age_props, r0_age = section_age_stratification(model_type, params, N, I0, R0_init)

        if contact_matrix is not None:
            if st.button("🚀 Run Age-Stratified Model", type="primary", key="run_age_model"):
                with st.spinner("Running age-stratified simulation..."):
                    sol_age = run_age_stratified_model(
                        model_type, params, contact_matrix, age_props,
                        N, I0, R0_init, t_span, t_eval=t_eval
                    )

                    comp_names = get_compartment_names(model_type)
                    n_comp = len(comp_names)
                    n_groups = len(age_props)

                    st.subheader("📊 Age Group Dynamics")
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    axes = axes.flatten()

                    for g in range(min(4, n_groups)):
                        ax = axes[g]
                        for c in range(n_comp):
                            idx = c * n_groups + g
                            if idx < sol_age.y.shape[0]:
                                vals = sol_age.y[idx]
                                ax.plot(sol_age.t, vals, label=comp_names[c], linewidth=2)
                        ax.set_title(f"Age Group: {AGE_GROUPS[g]} ({age_props[g]*100:.0f}%)", fontweight="bold")
                        ax.set_xlabel("Days")
                        ax.set_ylabel("Population")
                        ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.3, linestyle="--")
                        ax.spines["top"].set_alpha(0.3)
                        ax.spines["right"].set_alpha(0.3)

                    fig.suptitle("Age-Stratified Epidemic Dynamics", fontweight="bold", fontsize=14, y=1.01)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    _download_button(fig, "age_stratified.png")

                    st.subheader("📋 Final State by Age Group")
                    final_data = {}
                    for g in range(min(4, n_groups)):
                        age_data = {}
                        for c in range(n_comp):
                            idx = c * n_groups + g
                            if idx < sol_age.y.shape[0]:
                                age_data[comp_names[c]] = f"{sol_age.y[idx][-1]:,.0f}"
                        final_data[AGE_GROUPS[g]] = age_data
                    st.dataframe(pd.DataFrame(final_data), use_container_width=True)

    with tabs[2]:
        spatial_data = section_spatial(model_type, params, sim_days)

        if spatial_data is not None:
            n_regions = spatial_data["n_regions"]
            region_params = spatial_data["region_params"]
            commuting_matrix = spatial_data["commuting_matrix"]
            region_names = spatial_data["region_names"]

            if st.button("🚀 Run Spatial Model", type="primary", key="run_spatial"):
                with st.spinner(f"Running spatial metapopulation model ({n_regions} regions)..."):
                    sol_spatial = run_metapopulation_model(
                        model_type, region_params, commuting_matrix,
                        t_span, t_eval=t_eval
                    )

                    comp_names = get_compartment_names(model_type)
                    n_comp = len(comp_names)
                    I_idx = comp_names.index("I")

                    region_curves = []
                    region_total = []
                    for r in range(n_regions):
                        I_vals = sol_spatial.y[r * n_comp + I_idx]
                        region_curves.append(I_vals)
                        pop = region_params[r]["N"]
                        attack = (np.max(I_vals) / pop * 100) if pop > 0 else 0
                        region_total.append({
                            "Region": region_names[r],
                            "Population": f"{pop:,}",
                            "Peak Infected": f"{np.max(I_vals):,.0f}",
                            "Peak Day": f"Day {t_eval[np.argmax(I_vals)]:.0f}",
                            "Peak Prevalence": f"{attack:.2f}%",
                        })

                    st.subheader("📈 Regional Infection Curves")
                    fig_curves = plot_spatial_curves(
                        t_eval, region_curves, region_names
                    )
                    st.pyplot(fig_curves, use_container_width=True)
                    _download_button(fig_curves, "spatial_curves.png")

                    st.subheader("🗺️ Spatial Infection Heatmap")
                    fig_heatmap = plot_spatial_heatmap(
                        t_eval, region_curves, region_names
                    )
                    st.pyplot(fig_heatmap, use_container_width=True)
                    _download_button(fig_heatmap, "spatial_heatmap.png")

                    st.subheader("📋 Regional Summary")
                    st.dataframe(pd.DataFrame(region_total), use_container_width=True, hide_index=True)

    with tabs[3]:
        interventions = section_interventions(model_type, params, N)

        if interventions:
            if st.button("🚀 Run Model with Interventions", type="primary", key="run_interv"):
                with st.spinner("Running model with interventions..."):
                    ode_func = build_intervention_ode(model_type, params, N, interventions)
                    from models import get_initial_conditions
                    y0 = get_initial_conditions(model_type, N, I0, R0_init)

                    from scipy.integrate import solve_ivp
                    sol_interv = solve_ivp(
                        ode_func, t_span, y0,
                        t_eval=t_eval, method="RK45",
                        max_step=1.0, rtol=1e-8, atol=1e-10,
                    )

                    fig = plot_epidemic_curve(
                        sol_interv, model_type, chart_type="line",
                        interventions=interventions,
                        title=f"{model_type} with {len(interventions)} Intervention(s)"
                    )
                    st.pyplot(fig, use_container_width=True)
                    _download_button(fig, "intervention_curve.png")

                    st.subheader("📅 Intervention Timeline")
                    fig_timeline = plot_intervention_timeline(interventions, sim_days)
                    st.pyplot(fig_timeline, use_container_width=True)
                    _download_button(fig_timeline, "intervention_timeline.png")

                    st.subheader("⚖️ Impact Comparison: With vs Without Interventions")
                    sol_base = run_model(model_type, params, N, I0, R0_init, t_span, t_eval=t_eval)

                    fig_compare, ax = plt.subplots(figsize=(12, 7))
                    comp_names = get_compartment_names(model_type)
                    I_idx = comp_names.index("I")
                    ax.plot(sol_base.t, sol_base.y[I_idx], label="I (No Intervention)",
                           color="#e74c3c", linewidth=2.5, linestyle="--", alpha=0.8)
                    ax.plot(sol_interv.t, sol_interv.y[I_idx], label="I (With Intervention)",
                           color="#3498db", linewidth=2.5, alpha=0.9)

                    for interv in interventions:
                        start = interv.get("start_day", 0)
                        duration = interv.get("duration", 30)
                        ax.axvspan(start, start + duration, alpha=0.1, color="#f39c12")

                    ax.set_xlabel("Days")
                    ax.set_ylabel("Infected Population")
                    ax.set_title("Intervention Impact on Infected Population", fontweight="bold", fontsize=14)
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3, linestyle="--")
                    ax.spines["top"].set_alpha(0.3)
                    ax.spines["right"].set_alpha(0.3)
                    fig_compare.tight_layout()
                    st.pyplot(fig_compare, use_container_width=True)
                    _download_button(fig_compare, "intervention_compare.png")

                    col1, col2, col3 = st.columns(3)
                    base_peak = np.max(sol_base.y[I_idx])
                    interv_peak = np.max(sol_interv.y[I_idx])
                    reduction_pct = (1 - interv_peak / base_peak) * 100 if base_peak > 0 else 0

                    with col1:
                        st.metric("📉 Peak Reduction", f"{reduction_pct:.1f}%",
                                  delta=f"-{base_peak - interv_peak:,.0f} cases")
                    with col2:
                        base_cum = sol_base.y[comp_names.index("R")][-1] if "R" in comp_names else 0
                        interv_cum = sol_interv.y[comp_names.index("R")][-1] if "R" in comp_names else 0
                        cum_reduction = (1 - interv_cum / base_cum) * 100 if base_cum > 0 else 0
                        st.metric("🎯 Cumulative Reduction", f"{cum_reduction:.1f}%",
                                  delta=f"-{base_cum - interv_cum:,.0f} cases")
                    with col3:
                        base_peak_day = sol_base.t[np.argmax(sol_base.y[I_idx])]
                        interv_peak_day = sol_interv.t[np.argmax(sol_interv.y[I_idx])]
                        delay = interv_peak_day - base_peak_day
                        st.metric("⏰ Peak Delay", f"{delay:.0f} days",
                                  delta=f"from Day {base_peak_day:.0f} to Day {interv_peak_day:.0f}")

    with tabs[4]:
        scenario_configs = section_scenario_comparison(model_type, params, N, I0, R0_init, sim_days)

        if st.button("🚀 Run All Scenarios", type="primary", key="run_scenarios"):
            with st.spinner(f"Running {len(scenario_configs)} scenarios..."):
                from models import get_initial_conditions
                from scipy.integrate import solve_ivp

                all_results = []
                metrics_dict = {}

                for sc in scenario_configs:
                    sc_params = dict(sc["params"])
                    sc_interventions = sc["interventions"]

                    if sc_interventions:
                        ode_func = build_intervention_ode(model_type, sc_params, N, sc_interventions)
                        y0 = get_initial_conditions(model_type, N, I0, R0_init)
                        sol = solve_ivp(
                            ode_func, t_span, y0,
                            t_eval=t_eval, method="RK45",
                            max_step=1.0, rtol=1e-8, atol=1e-10,
                        )
                    else:
                        sol = run_model(model_type, sc_params, N, I0, R0_init, t_span, t_eval=t_eval)

                    all_results.append((sc["name"], sol))
                    metrics_dict[sc["name"]] = compute_scenario_metrics(sol, model_type)

                comp_names = get_compartment_names(model_type)
                I_idx = comp_names.index("I")

                st.subheader("📈 Infected Population Comparison")
                fig_comp, ax = plt.subplots(figsize=(12, 7))
                for i, (name, sol) in enumerate(all_results):
                    color = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"][i % 4]
                    ax.plot(sol.t, sol.y[I_idx], label=name, color=color, linewidth=2.5, alpha=0.9)
                ax.set_xlabel("Days")
                ax.set_ylabel("Infected Population")
                ax.set_title("Scenario Comparison - Infected (I) Population", fontweight="bold", fontsize=14)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3, linestyle="--")
                ax.spines["top"].set_alpha(0.3)
                ax.spines["right"].set_alpha(0.3)
                fig_comp.tight_layout()
                st.pyplot(fig_comp, use_container_width=True)
                _download_button(fig_comp, "scenario_comparison.png")

                st.subheader("📋 Comparison Metrics Table")
                metrics_df = pd.DataFrame(metrics_dict).T
                metrics_df.columns = [
                    "🦠 Cumulative Infections", "📈 Peak Infections",
                    "📅 Peak Time (days)", "✅ R_t < 1 Day"
                ]
                metrics_df = metrics_df.style.format({
                    "🦠 Cumulative Infections": "{:,.0f}",
                    "📈 Peak Infections": "{:,.0f}",
                    "📅 Peak Time (days)": "{:.0f}",
                    "✅ R_t < 1 Day": lambda x: f"{int(x)}" if pd.notna(x) else "—"
                })
                st.dataframe(metrics_df, use_container_width=True)

                st.subheader("📊 Quantitative Metrics Comparison")
                fig_bar = plot_scenario_metrics_bar(metrics_dict)
                st.pyplot(fig_bar, use_container_width=True)
                _download_button(fig_bar, "scenario_metrics.png")

    with tabs[5]:
        sol = run_model(model_type, params, N, I0, R0_init, t_span, t_eval=t_eval)
        section_rt_estimation(sol, model_type)

    with tabs[6]:
        section_sensitivity(model_type, params, N, I0, R0_init, t_span)

    with tabs[7]:
        section_optimal_timing(model_type, params, N, I0, R0_init, t_span)

    with tabs[8]:
        section_vaccine_allocation(
            model_type, params, N, I0, R0_init,
            vaccine_contact_matrix, vaccine_age_props
        )

    with tabs[9]:
        st.header("🏥 Healthcare Resource Stress Early Warning")
        st.markdown(
            "Dynamic assessment of healthcare system capacity stress based on epidemic simulation results, "
            "with multi-level early warning and mortality impact analysis."
        )

        st.subheader("⚙️ Resource Capacity Configuration")

        col_bed, col_icu, col_vent = st.columns(3)

        with col_bed:
            st.markdown("**🛏️ General Ward Beds**")
            bed_capacity = st.slider(
                "Total Beds",
                min_value=100, max_value=50000,
                value=5000, step=100,
                key="hc_bed_cap",
                help="Total number of general ward beds"
            )
            bed_threshold = st.slider(
                "Alert Threshold (%)",
                min_value=50, max_value=95,
                value=80, step=1,
                key="hc_bed_thresh",
                help="Occupancy rate threshold to trigger alert"
            ) / 100.0
            st.metric("Capacity", f"{bed_capacity:,} beds")

        with col_icu:
            st.markdown("**🏥 ICU Beds**")
            icu_capacity = st.slider(
                "Total ICU Beds",
                min_value=10, max_value=5000,
                value=500, step=10,
                key="hc_icu_cap",
                help="Total number of ICU beds"
            )
            icu_threshold = st.slider(
                "Alert Threshold (%)",
                min_value=50, max_value=95,
                value=70, step=1,
                key="hc_icu_thresh",
                help="Occupancy rate threshold to trigger alert"
            ) / 100.0
            st.metric("Capacity", f"{icu_capacity:,} ICU beds")

        with col_vent:
            st.markdown("**💨 Ventilators**")
            ventilator_capacity = st.slider(
                "Total Ventilators",
                min_value=5, max_value=2000,
                value=200, step=5,
                key="hc_vent_cap",
                help="Total number of ventilators"
            )
            vent_threshold = st.slider(
                "Alert Threshold (%)",
                min_value=50, max_value=95,
                value=60, step=1,
                key="hc_vent_thresh",
                help="Occupancy rate threshold to trigger alert"
            ) / 100.0
            st.metric("Capacity", f"{ventilator_capacity:,} ventilators")

        with st.expander("📊 Clinical Parameters", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                hosp_rate = st.slider(
                    "Hospitalization Rate (%)",
                    min_value=1.0, max_value=30.0,
                    value=15.0, step=0.5,
                    key="hc_hosp_rate",
                    help="Percentage of infected requiring general ward"
                ) / 100.0
                icu_rate = st.slider(
                    "ICU Admission Rate (%)",
                    min_value=0.5, max_value=10.0,
                    value=3.0, step=0.1,
                    key="hc_icu_rate",
                    help="Percentage of infected requiring ICU"
                ) / 100.0
                vent_rate = st.slider(
                    "Ventilator Rate (%)",
                    min_value=0.1, max_value=5.0,
                    value=1.0, step=0.1,
                    key="hc_vent_rate",
                    help="Percentage of infected requiring ventilator"
                ) / 100.0

            with col2:
                bed_stay = st.slider(
                    "Avg Hospital Stay (days)",
                    min_value=3, max_value=30,
                    value=10, step=1,
                    key="hc_bed_stay",
                    help="Average length of stay in general ward"
                )
                icu_stay = st.slider(
                    "Avg ICU Stay (days)",
                    min_value=5, max_value=40,
                    value=14, step=1,
                    key="hc_icu_stay",
                    help="Average length of stay in ICU"
                )
                vent_stay = st.slider(
                    "Avg Ventilator Days",
                    min_value=7, max_value=60,
                    value=21, step=1,
                    key="hc_vent_stay",
                    help="Average duration of mechanical ventilation"
                )

            with col3:
                stay_cv = st.slider(
                    "Stay Duration CV",
                    min_value=0.1, max_value=0.8,
                    value=0.3, step=0.05,
                    key="hc_stay_cv",
                    help="Coefficient of variation for length of stay (log-normal)"
                )
                baseline_mort = st.slider(
                    "Baseline Mortality Rate (%)",
                    min_value=0.1, max_value=5.0,
                    value=0.5, step=0.1,
                    key="hc_base_mort",
                    help="Baseline mortality rate for infected"
                ) / 100.0

            age_strat_hc = st.checkbox(
                "Enable Age-Stratified Hospitalization Rates",
                value=st.session_state.get("enable_age", False),
                key="hc_age_strat",
                help="Apply age-specific multipliers to hospitalization rates"
            )

            if age_strat_hc:
                st.info(
                    "💡 Age-specific multipliers:\n"
                    "- 0-17: 20% of baseline\n"
                    "- 18-44: 60% of baseline\n"
                    "- 45-64: 150% of baseline\n"
                    "- 65+: 300% of baseline"
                )

        with st.expander("🔄 Overflow Rules", expanded=False):
            st.markdown(
                "**Cascade overflow rules (top to bottom):**\n\n"
                "1. **Ventilator → ICU**: When ventilators are full, patients overflow to ICU "
                "(mortality +50%)\n"
                "2. **ICU → General Ward**: When ICU is full, patients overflow to general ward "
                "(mortality ×2)\n"
                "3. **General Ward → Untreated**: When general ward is full, patients cannot be "
                "admitted (mortality ×3 vs baseline)"
            )
            st.caption(
                "Mortality multipliers stack: e.g., a ventilator patient overflowing through "
                "ICU to general ward has combined increased mortality."
            )

        hc_config = {
            "bed_capacity": bed_capacity,
            "icu_capacity": icu_capacity,
            "ventilator_capacity": ventilator_capacity,
            "bed_threshold": bed_threshold,
            "icu_threshold": icu_threshold,
            "vent_threshold": vent_threshold,
            "hospitalization_rate": hosp_rate,
            "icu_rate": icu_rate,
            "ventilator_rate": vent_rate,
            "bed_stay_mean": bed_stay,
            "icu_stay_mean": icu_stay,
            "ventilator_stay_mean": vent_stay,
            "stay_cv": stay_cv,
            "baseline_mortality": baseline_mort,
        }

        sol_baseline = run_model(model_type, params, N, I0, R0_init, t_span, t_eval=t_eval)
        comp_names = get_compartment_names(model_type)
        I_idx = comp_names.index("I")
        I_vals = sol_baseline.y[I_idx]

        daily_new_infections = compute_daily_new_infections(sol_baseline.y, comp_names)

        hc_age_props = None
        if age_strat_hc and vaccine_age_props is not None:
            hc_age_props = vaccine_age_props
        elif age_strat_hc:
            hc_age_props = np.array(DEFAULT_AGE_PROPS)

        hc_result = run_healthcare_simulation(
            daily_new_infections,
            config=hc_config,
            age_stratified=age_strat_hc,
            age_props=hc_age_props,
        )

        st.markdown("---")

        st.subheader("📈 Resource Occupancy Over Time")
        fig_occ = plot_healthcare_occupancy(hc_result, title="Healthcare Resource Occupancy (Stacked)")
        st.pyplot(fig_occ, use_container_width=True)
        _download_button(fig_occ, "healthcare_occupancy.png")

        with st.expander("📊 Occupancy Details by Layer", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            for idx, (layer, label, emoji) in enumerate([
                ("bed", "General Beds", "🛏️"),
                ("icu", "ICU Beds", "🏥"),
                ("ventilator", "Ventilators", "💨"),
            ]):
                layer_data = hc_result["flow"][layer]
                peak_occ = np.max(layer_data["occupied"])
                peak_day = hc_result["flow"]["days"][np.argmax(layer_data["occupied"])]
                peak_rate = peak_occ / layer_data["capacity"] * 100 if layer_data["capacity"] > 0 else 0

                cols = [col_a, col_b, col_c]
                with cols[idx]:
                    st.metric(
                        f"{emoji} Peak {label}",
                        f"{peak_occ:,.0f}",
                        delta=f"{peak_rate:.1f}% occupancy at Day {int(peak_day)}",
                    )

        st.markdown("---")

        st.subheader("🚨 Alert Level Timeline")
        fig_alert = plot_alert_timeline(hc_result, title="Healthcare Alert Levels by Resource Layer")
        st.pyplot(fig_alert, use_container_width=True)
        _download_button(fig_alert, "alert_timeline.png")

        with st.expander("⏱️ Alert Level Details", expanded=False):
            st.markdown("**Alert Level Definitions:**")
            col_l1, col_l2, col_l3, col_l4 = st.columns(4)
            with col_l1:
                st.success("🟢 **Green** – Below 60% of threshold")
            with col_l2:
                st.warning("🟡 **Yellow** – 60% to 100% of threshold")
            with col_l3:
                st.info("🟠 **Orange** – Above threshold, still capacity remaining")
            with col_l4:
                st.error("🔴 **Red** – Capacity exhausted or ≤3 days from exhaustion")

            for layer, label in [
                ("bed", "General Beds"),
                ("icu", "ICU Beds"),
                ("ventilator", "Ventilators"),
            ]:
                levels = hc_result["alerts"][layer]["levels"]
                green_days = np.sum(levels == 0)
                yellow_days = np.sum(levels == 1)
                orange_days = np.sum(levels == 2)
                red_days = np.sum(levels == 3)
                total_days = len(levels)

                st.markdown(f"**{label}:**")
                col_g, col_y, col_o, col_r = st.columns(4)
                with col_g:
                    st.metric("🟢 Green", f"{green_days} days",
                              delta=f"{green_days/total_days*100:.0f}%")
                with col_y:
                    st.metric("🟡 Yellow", f"{yellow_days} days",
                              delta=f"{yellow_days/total_days*100:.0f}%")
                with col_o:
                    st.metric("🟠 Orange", f"{orange_days} days",
                              delta=f"{orange_days/total_days*100:.0f}%")
                with col_r:
                    st.metric("🔴 Red", f"{red_days} days",
                              delta=f"{red_days/total_days*100:.0f}%")

        st.markdown("---")

        st.subheader("💀 Mortality Impact Analysis")

        col_mort_chart, col_mort_cards = st.columns([2, 1])

        with col_mort_chart:
            fig_mort = plot_mortality_comparison(
                hc_result,
                title="Cumulative Mortality: Ideal vs Resource-Constrained"
            )
            st.pyplot(fig_mort, use_container_width=True)
            _download_button(fig_mort, "mortality_comparison.png")

        with col_mort_cards:
            mort = hc_result["mortality"]
            st.metric(
                "Ideal Deaths (Unlimited Resources)",
                f"{mort['ideal_cumulative'][-1]:,.0f}"
            )
            st.metric(
                "Actual Deaths (Resource Constrained)",
                f"{mort['actual_cumulative'][-1]:,.0f}",
                delta=f"+{mort['excess_deaths']:,.0f} ({mort['excess_pct']:.1f}%)",
                delta_color="inverse"
            )
            st.metric(
                "Excess Death Toll",
                f"{mort['excess_deaths']:,.0f}",
                delta=f"{mort['excess_pct']:.1f}% increase",
                delta_color="inverse"
            )

        st.markdown("---")

        st.subheader("📋 Summary Dashboard")
        fig_summary = plot_healthcare_summary_cards(hc_result)
        st.pyplot(fig_summary, use_container_width=True)
        _download_button(fig_summary, "healthcare_summary.png")

        summary = hc_result["summary"]
        flow = hc_result["flow"]

        with st.expander("📊 Detailed Metrics Table", expanded=True):
            metrics_data = []
            for layer, label in [
                ("bed", "General Beds"),
                ("icu", "ICU Beds"),
                ("ventilator", "Ventilators"),
            ]:
                layer_data = flow[layer]
                alert_data = hc_result["alerts"][layer]
                peak_occ = np.max(layer_data["occupied"])
                peak_rate = peak_occ / layer_data["capacity"] * 100 if layer_data["capacity"] > 0 else 0
                peak_day = flow["days"][np.argmax(layer_data["occupied"])]
                final_occ = layer_data["occupied"][-1]
                final_rate = final_occ / layer_data["capacity"] * 100 if layer_data["capacity"] > 0 else 0

                dte = alert_data["days_to_exhaust"]
                finite_dte = dte[np.isfinite(dte)]
                min_dte = np.min(finite_dte) if len(finite_dte) > 0 else None

                red_days = np.sum(alert_data["levels"] == 3)

                metrics_data.append({
                    "Resource": label,
                    "Total Capacity": f"{layer_data['capacity']:,}",
                    "Peak Occupancy": f"{peak_occ:,.0f} ({peak_rate:.1f}%)",
                    "Peak Day": f"Day {int(peak_day)}",
                    "Final Occupancy": f"{final_occ:,.0f} ({final_rate:.1f}%)",
                    "Days at Red Alert": f"{int(red_days)}",
                    "Earliest Exhaust (days)": f"{int(min_dte)}" if min_dte is not None else "N/A",
                })

            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        "🦠 Epidemic Modeling Platform | Powered by Streamlit, SciPy, and Matplotlib | "
        "For public health research and planning purposes."
    )


if __name__ == "__main__":
    main()
