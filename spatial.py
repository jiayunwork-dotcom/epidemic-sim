import numpy as np
from scipy.integrate import solve_ivp
from models import get_compartment_names


def metapopulation_ode(t, y, model_type, params_list, commuting_matrix, N_regions):
    n_regions = len(params_list)
    comp_names = get_compartment_names(model_type)
    n_comp = len(comp_names)

    states = np.array(y).reshape(n_regions, n_comp)
    dydt = np.zeros((n_regions, n_comp))

    I_idx = comp_names.index("I")
    S_idx = 0

    for r in range(n_regions):
        p = params_list[r]
        N = p["N"]
        beta = p["beta"]
        gamma = p["gamma"]
        sigma = p.get("sigma", 0.0)
        omega = p.get("omega", 0.0)

        S_r = states[r, S_idx]
        I_r = states[r, I_idx]

        effective_I = 0.0
        for j in range(n_regions):
            I_j = states[j, I_idx]
            N_j = params_list[j]["N"]
            effective_I += commuting_matrix[r, j] * I_j

        N_eff = 0.0
        for j in range(n_regions):
            N_eff += commuting_matrix[r, j] * params_list[j]["N"]

        if N_eff > 0:
            lambda_r = beta * effective_I / N_eff
        else:
            lambda_r = 0

        if model_type == "SIR":
            R_idx = 2
            dydt[r, S_idx] = -lambda_r * S_r
            dydt[r, I_idx] = lambda_r * S_r - gamma * I_r
            dydt[r, R_idx] = gamma * I_r

        elif model_type in ("SEIR", "SEIRS"):
            E_idx = 1
            R_idx = 3
            E_r = states[r, E_idx]
            R_r = states[r, R_idx]

            dydt[r, S_idx] = -lambda_r * S_r
            dydt[r, E_idx] = lambda_r * S_r - sigma * E_r
            dydt[r, I_idx] = sigma * E_r - gamma * I_r
            dydt[r, R_idx] = gamma * I_r

            if model_type == "SEIRS":
                dydt[r, S_idx] += omega * R_r
                dydt[r, R_idx] -= omega * R_r

        elif model_type == "SIS":
            dydt[r, S_idx] = -lambda_r * S_r + (gamma + omega) * I_r
            dydt[r, I_idx] = lambda_r * S_r - (gamma + omega) * I_r

    return dydt.flatten().tolist()


def run_metapopulation_model(model_type, params_list, commuting_matrix,
                              t_span, t_eval=None, interventions=None):
    n_regions = len(params_list)
    comp_names = get_compartment_names(model_type)
    n_comp = len(comp_names)

    y0 = np.zeros((n_regions, n_comp))
    for r in range(n_regions):
        p = params_list[r]
        N = p["N"]
        I0 = p.get("I0", 10)
        R0_init = p.get("R0", 0)
        S0 = max(N - I0 - R0_init, 0)

        y0[r, 0] = S0
        if model_type == "SIR":
            y0[r, 1] = I0
            y0[r, 2] = R0_init
        elif model_type in ("SEIR", "SEIRS"):
            y0[r, 1] = 0
            y0[r, 2] = I0
            y0[r, 3] = R0_init
        elif model_type == "SIS":
            y0[r, 1] = I0

    y0_flat = y0.flatten().tolist()

    commuting_matrix = np.array(commuting_matrix, dtype=float)

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], int(t_span[1]) + 1)

    def ode_with_interventions(t, y):
        current_commuting = commuting_matrix.copy()
        current_params = [dict(p) for p in params_list]

        if interventions:
            for interv in interventions:
                itype = interv.get("type", "")
                start = interv.get("start_day", 0)
                duration = interv.get("duration", 30)
                if start <= t <= start + duration:
                    if itype == "regional_lockdown":
                        factor = interv.get("lockdown_factor", 0.0)
                        diag = np.diag(np.diag(current_commuting))
                        off_diag = current_commuting - diag
                        current_commuting = diag + off_diag * factor
                        row_sums = current_commuting.sum(axis=1, keepdims=True)
                        current_commuting = current_commuting / np.maximum(row_sums, 1e-10)
                    elif itype == "social_distance":
                        reduction = interv.get("reduction", 0.5)
                        for rp in current_params:
                            rp["beta"] = rp["beta"] * reduction

        base_dydt = np.array(metapopulation_ode(
            t, y, model_type, current_params, current_commuting, n_regions))

        if interventions:
            states = np.array(y).reshape(n_regions, n_comp)
            dydt_reshaped = base_dydt.reshape(n_regions, n_comp)
            for interv in interventions:
                itype = interv.get("type", "")
                start = interv.get("start_day", 0)
                duration = interv.get("duration", 30)
                if not (start <= t <= start + duration):
                    continue

                if itype == "quarantine":
                    q_rate = interv.get("q_rate", 0.1)
                    I_idx_local = comp_names.index("I")
                    R_idx_local = comp_names.index("R") if "R" in comp_names else None
                    for r in range(n_regions):
                        q_transfer = q_rate * states[r, I_idx_local]
                        dydt_reshaped[r, I_idx_local] -= q_transfer
                        if R_idx_local is not None:
                            dydt_reshaped[r, R_idx_local] += q_transfer
                    base_dydt = dydt_reshaped.flatten()

                elif itype == "vaccination":
                    v_rate = interv.get("vacc_rate", 0.01)
                    efficacy = interv.get("efficacy", 0.8)
                    actual_rate = v_rate * efficacy
                    R_idx_local = comp_names.index("R") if "R" in comp_names else None
                    if R_idx_local is not None:
                        for r in range(n_regions):
                            v_transfer = actual_rate * states[r, 0]
                            dydt_reshaped[r, 0] -= v_transfer
                            dydt_reshaped[r, R_idx_local] += v_transfer
                        base_dydt = dydt_reshaped.flatten()

        return base_dydt.tolist()

    sol = solve_ivp(
        ode_with_interventions, t_span, y0_flat,
        t_eval=t_eval, method="RK45",
        max_step=1.0, rtol=1e-8, atol=1e-10,
    )
    return sol


def create_default_commuting_matrix(n_regions):
    matrix = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        stay = 0.85 if n_regions > 1 else 1.0
        matrix[i, i] = stay
        if n_regions > 1:
            off_diag_total = 1.0 - stay
            others = [j for j in range(n_regions) if j != i]
            for j in others:
                matrix[i, j] = off_diag_total / len(others)
    return matrix
