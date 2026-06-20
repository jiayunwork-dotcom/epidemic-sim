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
