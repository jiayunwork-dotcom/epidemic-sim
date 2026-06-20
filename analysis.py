import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.stats import spearmanr


def estimate_rt(daily_new_infections, mean_serial=5.0, std_serial=2.0, window=7):
    n = len(daily_new_infections)
    if n < 3:
        return np.array([]), np.array([])

    shape = (mean_serial / std_serial) ** 2
    scale = std_serial ** 2 / mean_serial

    max_lag = int(mean_serial + 4 * std_serial) + 1
    weights = np.zeros(max_lag)
    for lag in range(1, max_lag):
        weights[lag] = gamma_dist.pdf(lag, a=shape, scale=scale)
    weights /= weights.sum() if weights.sum() > 0 else 1

    rt_values = []
    rt_times = []
    last_valid_rt = None

    for t in range(max_lag, n):
        weighted_past = 0.0
        for lag in range(1, min(max_lag, t + 1)):
            weighted_past += weights[lag] * daily_new_infections[t - lag]

        if weighted_past > 0 and daily_new_infections[t] >= 0:
            rt = daily_new_infections[t] / weighted_past
            last_valid_rt = rt
        elif last_valid_rt is not None:
            rt = last_valid_rt * np.exp(-0.05 * (t - rt_times[-1] if rt_times else 1))
            rt = max(rt, 0.05)
        else:
            rt = 0.0

        rt_values.append(rt)
        rt_times.append(t)

    if len(rt_values) > 5:
        kernel = np.ones(5) / 5.0
        rt_values = np.convolve(rt_values, kernel, mode='same')

    return np.array(rt_times), np.array(rt_values)


def compute_daily_new_infections(compartments, comp_names=None):
    """计算每日新增感染数（发病率）
    
    正确方法：累计感染 = 当前感染I + 已恢复R（所有曾感染过的人），
    对累计感染做差分得到每日新增。
    
    参数:
        compartments: 可以是I数组（旧API兼容）或 (n_comp, n_time) 矩阵
        comp_names: 仓室名称列表，提供时用于识别I和R索引
    """
    if comp_names is not None and isinstance(compartments, np.ndarray) and compartments.ndim == 2:
        I_idx = comp_names.index("I")
        I_vals = compartments[I_idx]
        if "R" in comp_names:
            R_idx = comp_names.index("R")
            R_vals = compartments[R_idx]
            cumulative_infected = I_vals + R_vals
            daily_new = np.diff(cumulative_infected, prepend=cumulative_infected[0])
        elif "Q" in comp_names:
            Q_idx = comp_names.index("Q")
            Q_vals = compartments[Q_idx]
            cumulative_infected = I_vals + Q_vals
            if "R" in comp_names:
                R_idx = comp_names.index("R")
                cumulative_infected += compartments[R_idx]
            daily_new = np.diff(cumulative_infected, prepend=cumulative_infected[0])
        else:
            daily_new = np.diff(I_vals, prepend=I_vals[0])
    else:
        I_vals = np.asarray(compartments, dtype=float)
        daily_new = np.diff(I_vals, prepend=I_vals[0])

    daily_new = np.maximum(daily_new, 0.0)
    return daily_new


def latin_hypercube_sampling(param_ranges, n_samples=100, seed=42):
    try:
        from pyDOE2 import lhs
        n_params = len(param_ranges)
        lhs_samples = lhs(n_params, samples=n_samples, random_state=seed)
    except ImportError:
        from scipy.stats.qmc import LatinHypercube
        n_params = len(param_ranges)
        sampler = LatinHypercube(d=n_params, seed=seed)
        lhs_samples = sampler.random(n=n_samples)

    samples = np.zeros_like(lhs_samples)
    param_names = []
    for i, (name, (low, high)) in enumerate(param_ranges.items()):
        samples[:, i] = low + lhs_samples[:, i] * (high - low)
        param_names.append(name)

    return samples, param_names


def compute_prcc(param_samples, output_values):
    n_params = param_samples.shape[1]
    n_samples = param_samples.shape[0]

    prcc_values = np.zeros(n_params)
    p_values = np.zeros(n_params)

    ranked_params = np.zeros_like(param_samples)
    for i in range(n_params):
        ranked_params[:, i] = np.argsort(np.argsort(param_samples[:, i]))

    ranked_output = np.argsort(np.argsort(output_values))

    for i in range(n_params):
        others = [j for j in range(n_params) if j != i]
        if len(others) == 0:
            corr, pval = spearmanr(ranked_params[:, i], ranked_output)
            prcc_values[i] = corr
            p_values[i] = pval if pval is not None else 1.0
        else:
            from numpy.linalg import lstsq
            X = np.column_stack([ranked_params[:, j] for j in others] + [np.ones(n_samples)])
            y_param = ranked_params[:, i]
            y_out = ranked_output

            beta_param, _, _, _ = lstsq(X, y_param, rcond=None)
            beta_out, _, _, _ = lstsq(X, y_out, rcond=None)

            resid_param = y_param - X @ beta_param
            resid_out = y_out - X @ beta_out

            corr, pval = spearmanr(resid_param, resid_out)
            prcc_values[i] = corr
            p_values[i] = pval if pval is not None else 1.0

    return prcc_values, p_values


def run_sensitivity_analysis(model_type, param_ranges, base_params, N, I0, R0_init,
                              t_span, output_metric="cumulative", n_samples=100):
    from models import run_model, get_compartment_names

    samples, param_names = latin_hypercube_sampling(param_ranges, n_samples)

    cumulative_infections = np.zeros(n_samples)
    peak_infections = np.zeros(n_samples)
    peak_times = np.zeros(n_samples)

    comp_names = get_compartment_names(model_type)
    I_idx = comp_names.index("I")

    for i in range(n_samples):
        params = dict(base_params)
        for j, name in enumerate(param_names):
            params[name] = samples[i, j]

        try:
            sol = run_model(model_type, params, N, I0, R0_init, t_span)
            I_vals = sol.y[I_idx]
            if "R" in comp_names:
                cumulative_infections[i] = sol.y[comp_names.index("R")][-1]
            else:
                cumulative_infections[i] = np.sum(np.diff(I_vals, prepend=0))
            peak_infections[i] = np.max(I_vals)
            peak_times[i] = sol.t[np.argmax(I_vals)]
        except Exception:
            cumulative_infections[i] = np.nan
            peak_infections[i] = np.nan
            peak_times[i] = np.nan

    valid = ~np.isnan(cumulative_infections)
    if valid.sum() > 1:
        prcc_cum, pval_cum = compute_prcc(samples[valid], cumulative_infections[valid])
        prcc_peak, pval_peak = compute_prcc(samples[valid], peak_times[valid])
    else:
        prcc_cum = np.zeros(len(param_names))
        pval_cum = np.ones(len(param_names))
        prcc_peak = np.zeros(len(param_names))
        pval_peak = np.ones(len(param_names))

    return {
        "samples": samples,
        "param_names": param_names,
        "cumulative_infections": cumulative_infections,
        "peak_infections": peak_infections,
        "peak_times": peak_times,
        "prcc_cumulative": prcc_cum,
        "prcc_peak_time": prcc_peak,
        "pval_cumulative": pval_cum,
        "pval_peak_time": pval_peak,
    }


def run_monte_carlo_ci(model_type, params, N, I0, R0_init, t_span,
                        param_ranges, n_runs=50, confidence=0.95):
    from models import run_model, get_compartment_names

    comp_names = get_compartment_names(model_type)
    I_idx = comp_names.index("I")
    n_comp = len(comp_names)
    t_eval = np.linspace(t_span[0], t_span[1], int(t_span[1]) + 1)
    n_days = len(t_eval)

    all_curves = np.zeros((n_runs, n_comp, n_days))

    for run in range(n_runs):
        perturbed = dict(params)
        for name, (low, high) in param_ranges.items():
            perturbed[name] = np.random.uniform(low, high)

        try:
            sol = run_model(model_type, perturbed, N, I0, R0_init, t_span, t_eval=t_eval)
            if sol.success and sol.y.shape[1] == n_days:
                curve = np.clip(sol.y, 0.0, None)
                all_curves[run] = curve
            else:
                all_curves[run] = np.zeros((n_comp, n_days))
        except Exception:
            all_curves[run] = np.zeros((n_comp, n_days))

    alpha = (1 - confidence) / 2
    lower = np.percentile(all_curves, alpha * 100, axis=0)
    upper = np.percentile(all_curves, (1 - alpha) * 100, axis=0)
    median = np.median(all_curves, axis=0)

    lower = np.clip(lower, 0.0, None)
    upper = np.clip(upper, 0.0, None)
    median = np.clip(median, 0.0, None)

    return {
        "lower": lower,
        "upper": upper,
        "median": median,
        "t_eval": t_eval,
    }


def run_optimal_timing_analysis(model_type, params, N, I0, R0_init, t_span,
                                 intervention_config, max_start_day=60):
    from models import run_model, get_compartment_names, get_initial_conditions
    from interventions import build_intervention_ode
    from scipy.integrate import solve_ivp

    comp_names = get_compartment_names(model_type)
    n_comp = len(comp_names)
    I_idx = comp_names.index("I")

    t_eval = np.linspace(t_span[0], t_span[1], int(t_span[1]) + 1)
    cumulative_infections = np.zeros(max_start_day)
    peak_infections = np.zeros(max_start_day)
    peak_times_arr = np.zeros(max_start_day)
    all_I_curves = []

    for start_day in range(1, max_start_day + 1):
        interv = dict(intervention_config)
        interv["start_day"] = start_day
        interventions = [interv]

        ode_func = build_intervention_ode(model_type, params, N, interventions)
        y0 = get_initial_conditions(model_type, N, I0, R0_init)

        try:
            sol = solve_ivp(
                ode_func, t_span, y0,
                t_eval=t_eval, method="RK45",
                max_step=1.0, rtol=1e-8, atol=1e-10,
            )
            I_vals = sol.y[I_idx]
            if "R" in comp_names:
                cumulative_infections[start_day - 1] = sol.y[comp_names.index("R")][-1]
            else:
                cumulative_infections[start_day - 1] = np.sum(np.diff(I_vals, prepend=0))
            peak_infections[start_day - 1] = np.max(I_vals)
            peak_times_arr[start_day - 1] = sol.t[np.argmax(I_vals)]
            all_I_curves.append(I_vals)
        except Exception:
            cumulative_infections[start_day - 1] = np.nan
            peak_infections[start_day - 1] = np.nan
            peak_times_arr[start_day - 1] = np.nan
            all_I_curves.append(np.zeros(len(t_eval)))

    valid_mask = ~np.isnan(cumulative_infections)
    if valid_mask.sum() > 0:
        best_start = np.nanargmin(cumulative_infections) + 1
        threshold = np.nanmin(cumulative_infections) * 1.05
        optimal_window = np.where(cumulative_infections <= threshold)[0] + 1
    else:
        best_start = 1
        optimal_window = np.array([1])

    return {
        "start_days": np.arange(1, max_start_day + 1),
        "cumulative_infections": cumulative_infections,
        "peak_infections": peak_infections,
        "peak_times": peak_times_arr,
        "all_I_curves": all_I_curves,
        "best_start_day": best_start,
        "optimal_window": optimal_window,
        "t_eval": t_eval,
    }
