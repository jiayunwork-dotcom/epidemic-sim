import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvals


def sir_ode(t, y, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def seir_ode(t, y, beta, gamma, sigma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


def seirs_ode(t, y, beta, gamma, sigma, omega, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N + omega * R
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - omega * R
    return [dSdt, dEdt, dIdt, dRdt]


def sis_ode(t, y, beta, gamma, omega, N):
    S, I = y
    dSdt = -beta * S * I / N + (gamma + omega) * I
    dIdt = beta * S * I / N - (gamma + omega) * I
    return [dSdt, dIdt]


def get_ode_func(model_type):
    funcs = {
        "SIR": lambda t, y, p: sir_ode(t, y, p["beta"], p["gamma"], p["N"]),
        "SEIR": lambda t, y, p: seir_ode(t, y, p["beta"], p["gamma"], p["sigma"], p["N"]),
        "SEIRS": lambda t, y, p: seirs_ode(t, y, p["beta"], p["gamma"], p["sigma"], p["omega"], p["N"]),
        "SIS": lambda t, y, p: sis_ode(t, y, p["beta"], p["gamma"], p["omega"], p["N"]),
    }
    return funcs[model_type]


def get_initial_conditions(model_type, N, I0, R0):
    S0 = N - I0 - R0
    if S0 < 0:
        S0 = 0
        I0 = N - R0
    if model_type == "SIR":
        return [S0, I0, R0]
    elif model_type in ("SEIR", "SEIRS"):
        E0 = 0
        return [S0, E0, I0, R0]
    elif model_type == "SIS":
        return [S0, I0]
    else:
        return [S0, I0, R0]


def get_compartment_names(model_type):
    names = {
        "SIR": ["S", "I", "R"],
        "SEIR": ["S", "E", "I", "R"],
        "SEIRS": ["S", "E", "I", "R"],
        "SIS": ["S", "I"],
    }
    return names[model_type]


def run_model(model_type, params, N, I0, R0, t_span, t_eval=None):
    y0 = get_initial_conditions(model_type, N, I0, R0)
    ode_func = get_ode_func(model_type)
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], int(t_span[1]) + 1)
    sol = solve_ivp(
        lambda t, y: ode_func(t, y, params),
        t_span, y0, t_eval=t_eval, method="RK45",
        max_step=1.0, rtol=1e-8, atol=1e-10,
    )
    return sol


def compute_R0_basic(model_type, beta, gamma, sigma=None, omega=None):
    if model_type == "SIR":
        return beta / gamma
    elif model_type == "SEIR":
        return beta / gamma
    elif model_type == "SEIRS":
        if omega and omega > 0:
            return (beta / gamma) * (1.0 / (1.0 + omega / gamma))
        return beta / gamma
    elif model_type == "SIS":
        denom = gamma + (omega or 0)
        if denom <= 0:
            return 0.0
        return beta / denom
    return beta / gamma


def compute_R0_age_stratified(beta, gamma, contact_matrix, age_props, sigma=None, model_type="SIR"):
    n_groups = len(age_props)
    contact_matrix = np.array(contact_matrix, dtype=float)
    age_props = np.array(age_props, dtype=float)

    infectious_duration = 1.0 / gamma if gamma > 0 else 0.0

    if model_type == "SIR":
        D = np.diag(np.ones(n_groups) * infectious_duration)
    elif model_type in ("SEIR", "SEIRS"):
        D = np.diag(np.ones(n_groups) * infectious_duration)
    elif model_type == "SIS":
        D = np.diag(np.ones(n_groups) * infectious_duration)
    else:
        D = np.diag(np.ones(n_groups) * infectious_duration)

    P = np.diag(age_props)
    M = contact_matrix * beta
    NGM = D @ M @ P

    try:
        eigenvalues = eigvals(NGM)
        r0 = np.max(np.real(eigenvalues))
        return float(max(0.0, r0))
    except Exception:
        return float(beta / gamma if gamma > 0 else 0.0)


def age_stratified_ode(t, y, model_type, beta, gamma, sigma, omega,
                        contact_matrix, age_props, N, interventions=None):
    n_groups = len(age_props)
    comp_names = get_compartment_names(model_type)
    n_comp = len(comp_names)
    total_per_group = N * np.array(age_props)

    states = np.array(y).reshape(n_comp, n_groups)
    S = states[0].copy()

    if model_type == "SIR":
        I = states[1].copy()
        R = states[2].copy()
    elif model_type in ("SEIR", "SEIRS"):
        E = states[1].copy()
        I = states[2].copy()
        R = states[3].copy()
    elif model_type == "SIS":
        I = states[1].copy()

    effective_contact = contact_matrix.copy()
    effective_beta = beta

    if interventions:
        for interv in interventions:
            start = interv.get("start_day", 0)
            duration = interv.get("duration", 30)
            if start <= t <= start + duration:
                itype = interv.get("type", "")
                if itype == "social_distance":
                    reduction = interv.get("reduction", 0.5)
                    effective_contact = effective_contact * reduction

    force_of_infection = np.zeros(n_groups)
    for i in range(n_groups):
        for j in range(n_groups):
            if total_per_group[j] > 0:
                force_of_infection[i] += effective_contact[i, j] * I[j] / total_per_group[j]
    force_of_infection *= effective_beta

    dydt = np.zeros_like(states)

    dydt[0] = -S * force_of_infection

    if model_type == "SIR":
        dydt[1] = S * force_of_infection - gamma * I
        dydt[2] = gamma * I
    elif model_type in ("SEIR", "SEIRS"):
        dydt[1] = S * force_of_infection - sigma * E
        dydt[2] = sigma * E - gamma * I
        dydt[3] = gamma * I
        if model_type == "SEIRS":
            dydt[0] += omega * R
            dydt[3] -= omega * R
    elif model_type == "SIS":
        dydt[1] = S * force_of_infection - gamma * I - (omega or 0) * I
        dydt[0] += (gamma + (omega or 0)) * I

    if interventions:
        for interv in interventions:
            start = interv.get("start_day", 0)
            duration = interv.get("duration", 30)
            if not (start <= t <= start + duration):
                continue
            itype = interv.get("type", "")

            if itype == "quarantine":
                q_rate = interv.get("q_rate", 0.1)
                I_idx_local = comp_names.index("I")
                R_idx_local = comp_names.index("R") if "R" in comp_names else None
                for g in range(n_groups):
                    q_transfer = q_rate * states[I_idx_local, g]
                    dydt[I_idx_local, g] -= q_transfer
                    if R_idx_local is not None:
                        dydt[R_idx_local, g] += q_transfer

            elif itype == "vaccination":
                v_rate = interv.get("vacc_rate", 0.01)
                efficacy = interv.get("efficacy", 0.8)
                actual_rate = v_rate * efficacy
                R_idx_local = comp_names.index("R") if "R" in comp_names else None
                if R_idx_local is not None:
                    for g in range(n_groups):
                        v_transfer = actual_rate * states[0, g]
                        dydt[0, g] -= v_transfer
                        dydt[R_idx_local, g] += v_transfer

    return dydt.flatten().tolist()


def _vaccine_age_ode(t, y, model_type, beta, gamma, sigma, omega,
                      contact_matrix, age_props, N, daily_vacc):
    n_groups = len(age_props)
    comp_names = get_compartment_names(model_type)
    n_comp = len(comp_names)
    total_per_group = N * np.array(age_props)

    states = np.array(y).reshape(n_comp, n_groups)
    S = states[0].copy()

    if model_type == "SIR":
        I = states[1].copy()
        R = states[2].copy()
    elif model_type in ("SEIR", "SEIRS"):
        E = states[1].copy()
        I = states[2].copy()
        R = states[3].copy()
    elif model_type == "SIS":
        I = states[1].copy()

    force_of_infection = np.zeros(n_groups)
    for i in range(n_groups):
        for j in range(n_groups):
            if total_per_group[j] > 0:
                force_of_infection[i] += contact_matrix[i, j] * I[j] / total_per_group[j]
    force_of_infection *= beta

    dydt = np.zeros_like(states)

    dydt[0] = -S * force_of_infection

    if model_type == "SIR":
        dydt[1] = S * force_of_infection - gamma * I
        dydt[2] = gamma * I
    elif model_type in ("SEIR", "SEIRS"):
        dydt[1] = S * force_of_infection - sigma * E
        dydt[2] = sigma * E - gamma * I
        dydt[3] = gamma * I
        if model_type == "SEIRS":
            dydt[0] += omega * R
            dydt[3] -= omega * R
    elif model_type == "SIS":
        dydt[1] = S * force_of_infection - gamma * I - (omega or 0) * I
        dydt[0] += (gamma + (omega or 0)) * I

    if daily_vacc is not None:
        R_idx_local = comp_names.index("R") if "R" in comp_names else None
        if R_idx_local is not None:
            for g in range(n_groups):
                v_transfer = daily_vacc[g]
                dydt[0, g] -= v_transfer
                dydt[R_idx_local, g] += v_transfer

    return dydt.flatten().tolist()


def compute_vaccine_allocations(total_vaccines, vacc_days, age_props, contact_matrix,
                                custom_props=None):
    n_groups = len(age_props)
    daily_total = total_vaccines / vacc_days

    uniform_alloc = np.array(age_props) * daily_total

    elderly_priority = np.zeros(n_groups)
    remaining = daily_total
    elderly_idx = n_groups - 1
    for g in range(elderly_idx, -1, -1):
        if remaining <= 0:
            break
        elderly_priority[g] = remaining
        remaining = 0

    contact_sums = contact_matrix.sum(axis=1)
    sorted_indices = np.argsort(-contact_sums)
    high_contact_alloc = np.zeros(n_groups)
    remaining = daily_total
    for g in sorted_indices:
        if remaining <= 0:
            break
        high_contact_alloc[g] = remaining
        remaining = 0

    custom_alloc = np.zeros(n_groups)
    if custom_props is not None and len(custom_props) == n_groups:
        total_custom = sum(custom_props)
        if total_custom > 0:
            custom_alloc = np.array(custom_props) / total_custom * daily_total

    return {
        "uniform": uniform_alloc,
        "elderly_priority": elderly_priority,
        "high_contact": high_contact_alloc,
        "custom": custom_alloc,
    }


def run_vaccine_strategy_model(model_type, params, contact_matrix, age_props,
                                N, I0, R0, total_vaccines, vacc_days, efficacy,
                                custom_props=None, sim_days=300):
    n_groups = len(age_props)
    comp_names = get_compartment_names(model_type)
    n_comp = len(comp_names)
    total_per_group = N * np.array(age_props)

    age_props_sum = sum(age_props)
    if abs(age_props_sum - 1.0) > 0.01:
        age_props = [p / age_props_sum for p in age_props]
        total_per_group = N * np.array(age_props)

    I0_dist = I0 * np.array(age_props)
    R0_dist = R0 * np.array(age_props)
    S0_dist = np.maximum(total_per_group - I0_dist - R0_dist, 0)

    y0 = np.zeros((n_comp, n_groups))
    y0[0] = S0_dist
    if model_type == "SIR":
        y0[1] = I0_dist
        y0[2] = R0_dist
    elif model_type in ("SEIR", "SEIRS"):
        y0[1] = 0
        y0[2] = I0_dist
        y0[3] = R0_dist
    elif model_type == "SIS":
        y0[1] = I0_dist

    allocations = compute_vaccine_allocations(
        total_vaccines, vacc_days, age_props, contact_matrix, custom_props
    )

    elderly_priority_order = list(range(n_groups - 1, -1, -1))
    contact_sums = contact_matrix.sum(axis=1)
    high_contact_priority_order = list(np.argsort(-contact_sums))

    priority_orders = {
        "uniform": None,
        "elderly_priority": elderly_priority_order,
        "high_contact": high_contact_priority_order,
        "custom": None,
    }

    beta = params["beta"]
    gamma = params["gamma"]
    sigma = params.get("sigma", 0.0)
    omega = params.get("omega", 0.0)

    t_eval = np.linspace(0, sim_days, sim_days + 1)
    daily_total = total_vaccines / vacc_days

    def run_single(alloc_daily, priority_order=None):
        y_current = y0.flatten().copy()
        dt = 1.0
        results = np.zeros((n_comp * n_groups, sim_days + 1))
        results[:, 0] = y_current
        total_actual_doses = 0.0
        total_immune_conversions = np.zeros(n_groups)
        cum_immune_ts = np.zeros((n_groups, sim_days + 1))

        for day in range(sim_days):
            if day < vacc_days:
                target_daily = alloc_daily * efficacy
            else:
                target_daily = np.zeros(n_groups)

            states = y_current.reshape(n_comp, n_groups)
            S_vals = np.maximum(states[0].copy(), 0)

            if priority_order is not None and day < vacc_days:
                actual_vacc = np.zeros(n_groups)
                remaining_daily = daily_total * efficacy
                for g in priority_order:
                    if remaining_daily <= 0:
                        break
                    can_vacc = min(remaining_daily, S_vals[g])
                    actual_vacc[g] = max(can_vacc, 0)
                    remaining_daily -= actual_vacc[g]
            elif priority_order is not None:
                actual_vacc = np.zeros(n_groups)
            else:
                actual_vacc = np.minimum(target_daily, S_vals)
                actual_vacc = np.maximum(actual_vacc, 0)

            total_immune_conversions += actual_vacc
            cum_immune_ts[:, day + 1] = total_immune_conversions.copy()

            if efficacy > 0:
                daily_doses_consumed = np.sum(actual_vacc) / efficacy
            else:
                daily_doses_consumed = 0.0
            total_actual_doses += daily_doses_consumed

            def ode_func(t_local, y_local):
                return _vaccine_age_ode(
                    t_local, y_local, model_type, beta, gamma, sigma, omega,
                    contact_matrix, age_props, N, actual_vacc
                )

            sol_step = solve_ivp(
                ode_func, (day, day + dt), y_current,
                method="RK45", max_step=1.0, rtol=1e-8, atol=1e-10,
            )

            y_current = sol_step.y[:, -1].copy()

            states_new = y_current.reshape(n_comp, n_groups)
            for g in range(n_groups):
                if states_new[0, g] < 0:
                    states_new[0, g] = 0

            y_current = states_new.flatten()
            results[:, day + 1] = y_current

        return results, total_actual_doses, total_immune_conversions, cum_immune_ts

    baseline_alloc = np.zeros(n_groups)
    baseline_result, _, _, baseline_immune_ts = run_single(baseline_alloc)

    strategy_results = {}
    strategy_actual_doses = {}
    strategy_immune_conversions = {}
    strategy_immune_ts = {}
    for strategy_name, alloc_daily in allocations.items():
        priority_order = priority_orders.get(strategy_name)
        strat_result, actual_doses, immune_conv, immune_ts = run_single(alloc_daily, priority_order)
        strategy_results[strategy_name] = strat_result
        strategy_actual_doses[strategy_name] = actual_doses
        strategy_immune_conversions[strategy_name] = immune_conv
        strategy_immune_ts[strategy_name] = immune_ts

    return {
        "t_eval": t_eval,
        "baseline": baseline_result,
        "baseline_immune_ts": baseline_immune_ts,
        "strategies": strategy_results,
        "strategy_actual_doses": strategy_actual_doses,
        "strategy_immune_conversions": strategy_immune_conversions,
        "strategy_immune_ts": strategy_immune_ts,
        "n_groups": n_groups,
        "n_comp": n_comp,
        "comp_names": comp_names,
        "age_props": np.array(age_props),
        "total_per_group": total_per_group,
    }


def compute_vaccine_metrics(result, N, total_vaccines):
    comp_names = result["comp_names"]
    n_comp = result["n_comp"]
    n_groups = result["n_groups"]
    age_props = result["age_props"]
    total_per_group = result["total_per_group"]
    baseline = result["baseline"]
    strategy_actual_doses = result.get("strategy_actual_doses", {})
    strategy_immune_conversions = result.get("strategy_immune_conversions", {})

    if "R" in comp_names:
        R_idx = comp_names.index("R")
        baseline_cum = 0
        for g in range(n_groups):
            baseline_cum += baseline[R_idx * n_groups + g][-1]
    else:
        I_idx = comp_names.index("I")
        baseline_cum = 0
        for g in range(n_groups):
            I_vals = baseline[I_idx * n_groups + g]
            baseline_cum += np.sum(np.diff(I_vals, prepend=0))

    I_idx = comp_names.index("I")
    baseline_peak = 0
    baseline_peak_day = 0
    total_I_baseline = np.zeros(len(result["t_eval"]))
    for g in range(n_groups):
        total_I_baseline += baseline[I_idx * n_groups + g]
    baseline_peak = np.max(total_I_baseline)
    baseline_peak_day = result["t_eval"][np.argmax(total_I_baseline)]

    all_attack_std = []
    strategy_attack_rates = {}
    strategy_cum_infections = {}

    for strategy_name, strat_data in result["strategies"].items():
        immune_conv = strategy_immune_conversions.get(strategy_name, np.zeros(n_groups))
        attack_rates = np.zeros(n_groups)
        cum_infections = 0
        for g in range(n_groups):
            if "R" in comp_names:
                R_idx = comp_names.index("R")
                group_R = strat_data[R_idx * n_groups + g][-1]
                group_cum = max(0.0, group_R - immune_conv[g])
            else:
                I_vals_g = strat_data[I_idx * n_groups + g]
                group_cum = np.sum(np.diff(I_vals_g, prepend=0))
            cum_infections += group_cum
            if total_per_group[g] > 0:
                attack_rates[g] = group_cum / total_per_group[g]
        strategy_attack_rates[strategy_name] = attack_rates
        strategy_cum_infections[strategy_name] = cum_infections
        all_attack_std.append(np.std(attack_rates))

    max_std = max(all_attack_std) if all_attack_std and max(all_attack_std) > 0 else 1.0

    metrics = {}
    for strategy_name, strat_data in result["strategies"].items():
        cum_infections = strategy_cum_infections[strategy_name]

        total_I_strat = np.zeros(len(result["t_eval"]))
        for g in range(n_groups):
            total_I_strat += strat_data[I_idx * n_groups + g]
        peak_val = np.max(total_I_strat)
        peak_day = result["t_eval"][np.argmax(total_I_strat)]

        cum_reduction = (baseline_cum - cum_infections) / baseline_cum if baseline_cum > 0 else 0
        peak_reduction = (baseline_peak - peak_val) / baseline_peak if baseline_peak > 0 else 0
        peak_delay = peak_day - baseline_peak_day

        actual_doses = strategy_actual_doses.get(strategy_name, total_vaccines)
        infections_avoided = max(0, baseline_cum - cum_infections)
        if actual_doses > 0:
            vacc_efficiency = infections_avoided / actual_doses
        else:
            vacc_efficiency = 0.0

        attack_rates = strategy_attack_rates[strategy_name]
        std_attack = np.std(attack_rates)

        if std_attack == 0:
            fairness = 1.0
        else:
            fairness = 1.0 / (1.0 + std_attack)

        metrics[strategy_name] = {
            "cum_infections": cum_infections,
            "cum_reduction_pct": cum_reduction * 100,
            "peak_val": peak_val,
            "peak_reduction_pct": peak_reduction * 100,
            "peak_day": peak_day,
            "peak_delay": peak_delay,
            "vacc_efficiency": vacc_efficiency,
            "actual_doses": actual_doses,
            "fairness": fairness,
            "attack_rates": attack_rates,
        }

    return {
        "baseline_cum": baseline_cum,
        "baseline_peak": baseline_peak,
        "baseline_peak_day": baseline_peak_day,
        "strategies": metrics,
    }


def run_age_stratified_model(model_type, params, contact_matrix, age_props,
                              N, I0, R0, t_span, t_eval=None, interventions=None):
    n_groups = len(age_props)
    comp_names = get_compartment_names(model_type)
    n_comp = len(comp_names)
    total_per_group = N * np.array(age_props)

    age_props_sum = sum(age_props)
    if abs(age_props_sum - 1.0) > 0.01:
        age_props = [p / age_props_sum for p in age_props]
        total_per_group = N * np.array(age_props)

    I0_dist = I0 * np.array(age_props)
    R0_dist = R0 * np.array(age_props)
    S0_dist = np.maximum(total_per_group - I0_dist - R0_dist, 0)

    y0 = np.zeros((n_comp, n_groups))
    y0[0] = S0_dist
    if model_type == "SIR":
        y0[1] = I0_dist
        y0[2] = R0_dist
    elif model_type in ("SEIR", "SEIRS"):
        y0[1] = 0
        y0[2] = I0_dist
        y0[3] = R0_dist
    elif model_type == "SIS":
        y0[1] = I0_dist

    y0_flat = y0.flatten().tolist()

    beta = params["beta"]
    gamma = params["gamma"]
    sigma = params.get("sigma", 0.0)
    omega = params.get("omega", 0.0)

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], int(t_span[1]) + 1)

    sol = solve_ivp(
        lambda t, y: age_stratified_ode(
            t, y, model_type, beta, gamma, sigma, omega,
            contact_matrix, age_props, N, interventions),
        t_span, y0_flat, t_eval=t_eval, method="RK45",
        max_step=1.0, rtol=1e-8, atol=1e-10,
    )
    return sol
