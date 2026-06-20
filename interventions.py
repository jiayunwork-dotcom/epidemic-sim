import numpy as np
from scipy.integrate import solve_ivp


def _get_comp_names(model_type):
    if model_type == "SIR":
        return ["S", "I", "R"]
    elif model_type in ("SEIR", "SEIRS"):
        return ["S", "E", "I", "R"]
    elif model_type == "SIS":
        return ["S", "I"]
    return ["S", "I", "R"]


def build_intervention_ode(model_type, params, N, interventions,
                           contact_matrix=None, age_props=None):
    from models import get_ode_func, sir_ode, seir_ode, seirs_ode, sis_ode

    beta = params["beta"]
    gamma = params["gamma"]
    sigma = params.get("sigma", 0.0)
    omega = params.get("omega", 0.0)

    def ode_with_interventions(t, y):
        y = np.array(y, dtype=float)
        comp_names = _get_comp_names(model_type)
        n_comp = len(comp_names)

        effective_beta = beta
        effective_gamma = gamma

        social_distance_factor = 1.0

        if interventions:
            for interv in interventions:
                start = interv.get("start_day", 0)
                duration = interv.get("duration", 30)
                if not (start <= t <= start + duration):
                    continue
                itype = interv.get("type", "")

                if itype == "social_distance":
                    reduction = interv.get("reduction", 0.5)
                    social_distance_factor *= reduction

        effective_beta = beta * social_distance_factor

        if model_type == "SIR":
            dydt = np.array(sir_ode(t, y, effective_beta, effective_gamma, N), dtype=float)
        elif model_type == "SEIR":
            dydt = np.array(seir_ode(t, y, effective_beta, effective_gamma, sigma, N), dtype=float)
        elif model_type == "SEIRS":
            dydt = np.array(seirs_ode(t, y, effective_beta, effective_gamma, sigma, omega, N), dtype=float)
        elif model_type == "SIS":
            dydt = np.array(sis_ode(t, y, effective_beta, effective_gamma, omega, N), dtype=float)
        else:
            dydt = np.array(sir_ode(t, y, effective_beta, effective_gamma, N), dtype=float)

        if interventions:
            for interv in interventions:
                start = interv.get("start_day", 0)
                duration = interv.get("duration", 30)
                if not (start <= t <= start + duration):
                    continue
                itype = interv.get("type", "")

                if itype == "quarantine":
                    q_rate = interv.get("q_rate", 0.1)
                    I_idx = comp_names.index("I")
                    R_idx = comp_names.index("R") if "R" in comp_names else None
                    q_transfer = q_rate * y[I_idx]
                    dydt[I_idx] -= q_transfer
                    if R_idx is not None:
                        dydt[R_idx] += q_transfer

                elif itype == "vaccination":
                    v_rate = interv.get("vacc_rate", 0.01)
                    efficacy = interv.get("efficacy", 0.8)
                    actual_rate = v_rate * efficacy
                    S_idx = 0
                    R_idx = comp_names.index("R") if "R" in comp_names else None
                    if R_idx is not None:
                        v_transfer = actual_rate * y[S_idx]
                        dydt[S_idx] -= v_transfer
                        dydt[R_idx] += v_transfer

        return dydt.tolist()

    return ode_with_interventions


def build_age_intervention_ode(model_type, params, N, interventions,
                                contact_matrix, age_props):
    from models import age_stratified_ode

    beta = params["beta"]
    gamma = params["gamma"]
    sigma = params.get("sigma", 0.0)
    omega = params.get("omega", 0.0)

    def ode_with_interventions(t, y):
        return age_stratified_ode(
            t, y, model_type, beta, gamma, sigma, omega,
            contact_matrix, age_props, N, interventions
        )

    return ode_with_interventions


def build_quarantine_extended_ode(model_type, params, N, interventions):
    """ODE with explicit Q (quarantine) compartment for SIQR/SEIQR models"""
    from models import get_initial_conditions

    beta = params["beta"]
    gamma = params["gamma"]
    sigma = params.get("sigma", 0.0)
    omega = params.get("omega", 0.0)
    comp_names_base = _get_comp_names(model_type)

    def ode_with_q(t, y):
        y = np.array(y, dtype=float)

        if model_type == "SIR":
            S, I, R, Q = y[0], y[1], y[2], y[3]
            q_rate = 0.0
            social_factor = 1.0

            if interventions:
                for interv in interventions:
                    start = interv.get("start_day", 0)
                    duration = interv.get("duration", 30)
                    if not (start <= t <= start + duration):
                        continue
                    itype = interv.get("type", "")
                    if itype == "quarantine":
                        q_rate = interv.get("q_rate", 0.1)
                    elif itype == "social_distance":
                        social_factor *= interv.get("reduction", 0.5)
                    elif itype == "vaccination":
                        pass

            effective_beta = beta * social_factor
            dSdt = -effective_beta * S * I / N
            dIdt = effective_beta * S * I / N - gamma * I - q_rate * I
            dRdt = gamma * I
            dQdt = q_rate * I

            v_rate_total = 0.0
            v_efficacy = 0.0
            if interventions:
                for interv in interventions:
                    start = interv.get("start_day", 0)
                    duration = interv.get("duration", 30)
                    if not (start <= t <= start + duration):
                        continue
                    itype = interv.get("type", "")
                    if itype == "vaccination":
                        v_rate_total = interv.get("vacc_rate", 0.01)
                        v_efficacy = interv.get("efficacy", 0.8)

            actual_v_rate = v_rate_total * v_efficacy
            if actual_v_rate > 0:
                v_transfer = actual_v_rate * S
                dSdt -= v_transfer
                dRdt += v_transfer

            return [dSdt, dIdt, dRdt, dQdt]

        elif model_type in ("SEIR", "SEIRS"):
            S, E, I, R, Q = y[0], y[1], y[2], y[3], y[4]
            q_rate = 0.0
            social_factor = 1.0

            if interventions:
                for interv in interventions:
                    start = interv.get("start_day", 0)
                    duration = interv.get("duration", 30)
                    if not (start <= t <= start + duration):
                        continue
                    itype = interv.get("type", "")
                    if itype == "quarantine":
                        q_rate = interv.get("q_rate", 0.1)
                    elif itype == "social_distance":
                        social_factor *= interv.get("reduction", 0.5)

            effective_beta = beta * social_factor
            dSdt = -effective_beta * S * I / N
            dEdt = effective_beta * S * I / N - sigma * E
            dIdt = sigma * E - gamma * I - q_rate * I
            dRdt = gamma * I
            dQdt = q_rate * I

            if model_type == "SEIRS":
                dSdt += omega * R
                dRdt -= omega * R

            v_rate_total = 0.0
            v_efficacy = 0.0
            if interventions:
                for interv in interventions:
                    start = interv.get("start_day", 0)
                    duration = interv.get("duration", 30)
                    if not (start <= t <= start + duration):
                        continue
                    itype = interv.get("type", "")
                    if itype == "vaccination":
                        v_rate_total = interv.get("vacc_rate", 0.01)
                        v_efficacy = interv.get("efficacy", 0.8)

            actual_v_rate = v_rate_total * v_efficacy
            if actual_v_rate > 0:
                v_transfer = actual_v_rate * S
                dSdt -= v_transfer
                dRdt += v_transfer

            return [dSdt, dEdt, dIdt, dRdt, dQdt]

        elif model_type == "SIS":
            S, I, Q = y[0], y[1], y[2]
            q_rate = 0.0
            social_factor = 1.0

            if interventions:
                for interv in interventions:
                    start = interv.get("start_day", 0)
                    duration = interv.get("duration", 30)
                    if not (start <= t <= start + duration):
                        continue
                    itype = interv.get("type", "")
                    if itype == "quarantine":
                        q_rate = interv.get("q_rate", 0.1)
                    elif itype == "social_distance":
                        social_factor *= interv.get("reduction", 0.5)

            effective_beta = beta * social_factor
            dSdt = -effective_beta * S * I / N + (gamma + omega) * I
            dIdt = effective_beta * S * I / N - (gamma + omega) * I - q_rate * I
            dQdt = q_rate * I

            return [dSdt, dIdt, dQdt]

        else:
            return np.zeros_like(y).tolist()

    return ode_with_q


def get_q_initial_conditions(model_type, N, I0, R0):
    """Get initial conditions including Q compartment"""
    from models import get_initial_conditions
    base_y0 = get_initial_conditions(model_type, N, I0, R0)

    if model_type == "SIR":
        return base_y0 + [0.0]
    elif model_type in ("SEIR", "SEIRS"):
        return base_y0 + [0.0]
    elif model_type == "SIS":
        return base_y0 + [0.0]
    return base_y0


def get_q_compartment_names(model_type):
    """Get compartment names including Q"""
    base_names = _get_comp_names(model_type)
    return base_names + ["Q"]
