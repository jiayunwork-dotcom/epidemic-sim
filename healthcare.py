import numpy as np
from scipy.stats import lognorm


AGE_GROUP_MULTIPLIERS = {
    "0-17": 0.20,
    "18-44": 0.60,
    "45-64": 1.50,
    "65+": 3.00,
}

BASELINE_MORTALITY_RATE = 0.005

RESOURCE_LAYERS = ["bed", "icu", "ventilator"]


def _lognormal_cdf(x, mu, sigma):
    if x <= 0:
        return 0.0
    return lognorm.cdf(x, s=sigma, scale=np.exp(mu))


def _compute_lognormal_params(mean, cv=0.3):
    sigma = np.sqrt(np.log(1 + cv ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return mu, sigma


def compute_daily_new_infections_from_I(I_vals):
    cum_infected = np.array(I_vals, dtype=float)
    daily_new = np.diff(cum_infected, prepend=cum_infected[0])
    daily_new = np.maximum(daily_new, 0.0)
    return daily_new


def compute_admission_rates(age_stratified=False, age_props=None,
                            base_hospitalization_rate=0.15,
                            base_icu_rate=0.03,
                            base_ventilator_rate=0.01):
    if not age_stratified:
        return {
            "bed": base_hospitalization_rate,
            "icu": base_icu_rate,
            "ventilator": base_ventilator_rate,
        }

    age_groups = ["0-17", "18-44", "45-64", "65+"]
    if age_props is None:
        age_props = [0.20, 0.35, 0.30, 0.15]
    age_props = np.array(age_props) / np.sum(age_props)

    weighted_rates = {}
    for layer, base_rate in [("bed", base_hospitalization_rate),
                              ("icu", base_icu_rate),
                              ("ventilator", base_ventilator_rate)]:
        weighted = 0.0
        for i, age in enumerate(age_groups):
            mult = AGE_GROUP_MULTIPLIERS[age]
            weighted += age_props[i] * base_rate * mult
        weighted_rates[layer] = weighted

    return weighted_rates


def simulate_hospital_flow(daily_new_infections, total_days,
                          hospitalization_rate=0.15,
                          icu_rate=0.03,
                          ventilator_rate=0.01,
                          bed_capacity=5000,
                          icu_capacity=500,
                          ventilator_capacity=200,
                          bed_stay_mean=10,
                          icu_stay_mean=14,
                          ventilator_stay_mean=21,
                          stay_cv=0.3,
                          age_stratified=False,
                          age_props=None,
                          base_hospitalization_rate=0.15,
                          base_icu_rate=0.03,
                          base_ventilator_rate=0.01):
    if age_stratified:
        rates = compute_admission_rates(
            age_stratified=True, age_props=age_props,
            base_hospitalization_rate=base_hospitalization_rate,
            base_icu_rate=base_icu_rate,
            base_ventilator_rate=base_ventilator_rate,
        )
        hospitalization_rate = rates["bed"]
        icu_rate = rates["icu"]
        ventilator_rate = rates["ventilator"]

    daily_new = np.array(daily_new_infections, dtype=float)
    n_days = min(len(daily_new), total_days)

    bed_admissions = daily_new[:n_days] * hospitalization_rate
    icu_admissions = daily_new[:n_days] * icu_rate
    vent_admissions = daily_new[:n_days] * ventilator_rate

    bed_mu, bed_sigma = _compute_lognormal_params(bed_stay_mean, stay_cv)
    icu_mu, icu_sigma = _compute_lognormal_params(icu_stay_mean, stay_cv)
    vent_mu, vent_sigma = _compute_lognormal_params(ventilator_stay_mean, stay_cv)

    bed_occupied = np.zeros(n_days)
    icu_occupied = np.zeros(n_days)
    vent_occupied = np.zeros(n_days)

    bed_discharges = np.zeros(n_days)
    icu_discharges = np.zeros(n_days)
    vent_discharges = np.zeros(n_days)

    bed_overflow_to = np.zeros(n_days)
    icu_overflow_to_bed = np.zeros(n_days)
    vent_overflow_to_icu = np.zeros(n_days)

    bed_untreated = np.zeros(n_days)
    icu_untreated = np.zeros(n_days)
    vent_untreated = np.zeros(n_days)

    for day in range(n_days):
        if day == 0:
            continue

        bed_occ_prev = bed_occupied[day - 1]
        icu_occ_prev = icu_occupied[day - 1]
        vent_occ_prev = vent_occupied[day - 1]

        bed_disch = 0.0
        icu_disch = 0.0
        vent_disch = 0.0

        for past_day in range(day):
            days_since_admit = day - past_day

            bed_disch += bed_admissions[past_day] * (
                _lognormal_cdf(days_since_admit + 0.5, bed_mu, bed_sigma) -
                _lognormal_cdf(days_since_admit - 0.5, bed_mu, bed_sigma)
            )

            icu_disch += icu_admissions[past_day] * (
                _lognormal_cdf(days_since_admit + 0.5, icu_mu, icu_sigma) -
                _lognormal_cdf(days_since_admit - 0.5, icu_mu, icu_sigma)
            )

            vent_disch += vent_admissions[past_day] * (
                _lognormal_cdf(days_since_admit + 0.5, vent_mu, vent_sigma) -
                _lognormal_cdf(days_since_admit - 0.5, vent_mu, vent_sigma)
            )

        bed_discharges[day] = bed_disch
        icu_discharges[day] = icu_disch
        vent_discharges[day] = vent_disch

        vent_new = vent_admissions[day]
        vent_after_discharge = max(0.0, vent_occ_prev - vent_disch)
        vent_total = vent_after_discharge + vent_new

        if vent_total <= ventilator_capacity:
            vent_occupied[day] = vent_total
            vent_overflow_to_icu[day] = 0.0
        else:
            vent_occupied[day] = ventilator_capacity
            vent_overflow = vent_total - ventilator_capacity
            vent_overflow_to_icu[day] = vent_overflow

        icu_new = icu_admissions[day] + vent_overflow_to_icu[day]
        icu_after_discharge = max(0.0, icu_occ_prev - icu_disch)
        icu_total = icu_after_discharge + icu_new

        if icu_total <= icu_capacity:
            icu_occupied[day] = icu_total
            icu_overflow_to_bed[day] = 0.0
        else:
            icu_occupied[day] = icu_capacity
            icu_overflow = icu_total - icu_capacity
            icu_overflow_to_bed[day] = icu_overflow

        bed_new = bed_admissions[day] + icu_overflow_to_bed[day]
        bed_after_discharge = max(0.0, bed_occ_prev - bed_disch)
        bed_total = bed_after_discharge + bed_new

        if bed_total <= bed_capacity:
            bed_occupied[day] = bed_total
            bed_untreated[day] = 0.0
        else:
            bed_occupied[day] = bed_capacity
            bed_untreated[day] = bed_total - bed_capacity

    result = {
        "days": np.arange(n_days),
        "daily_new_infections": daily_new[:n_days],
        "bed": {
            "admissions": bed_admissions,
            "discharges": bed_discharges,
            "occupied": bed_occupied,
            "capacity": bed_capacity,
            "overflow_from_icu": icu_overflow_to_bed,
            "untreated": bed_untreated,
        },
        "icu": {
            "admissions": icu_admissions,
            "discharges": icu_discharges,
            "occupied": icu_occupied,
            "capacity": icu_capacity,
            "overflow_from_ventilator": vent_overflow_to_icu,
            "overflow_to_bed": icu_overflow_to_bed,
        },
        "ventilator": {
            "admissions": vent_admissions,
            "discharges": vent_discharges,
            "occupied": vent_occupied,
            "capacity": ventilator_capacity,
            "overflow_to_icu": vent_overflow_to_icu,
        },
    }

    return result


def compute_occupancy_rates(result):
    rates = {}
    for layer in ["bed", "icu", "ventilator"]:
        layer_data = result[layer]
        capacity = layer_data["capacity"]
        occupied = layer_data["occupied"]
        rates[layer] = occupied / capacity if capacity > 0 else np.zeros_like(occupied)
    return rates


def compute_alert_levels(occupancy_rates, capacities,
                        bed_threshold=0.80,
                        icu_threshold=0.70,
                        vent_threshold=0.60,
                        window_days=7):
    thresholds = {
        "bed": bed_threshold,
        "icu": icu_threshold,
        "ventilator": vent_threshold,
    }

    alert_results = {}

    for layer in ["bed", "icu", "ventilator"]:
        occ = occupancy_rates[layer]
        cap = capacities[layer]
        threshold = thresholds[layer]
        n_days = len(occ)

        alert_levels = np.zeros(n_days, dtype=int)
        days_to_exhaust = np.full(n_days, np.inf)

        for day in range(n_days):
            current_occ = occ[day]

            if current_occ < threshold * 0.6:
                alert_levels[day] = 0
            elif current_occ < threshold:
                alert_levels[day] = 1
            elif current_occ < 1.0:
                alert_levels[day] = 2
            else:
                alert_levels[day] = 3

            if current_occ >= 1.0:
                days_to_exhaust[day] = 0
            else:
                start_idx = max(0, day - window_days + 1)
                recent_occ = occ[start_idx:day + 1]
                if len(recent_occ) >= 2:
                    growth_rates = np.diff(recent_occ)
                    avg_growth = np.mean(growth_rates)
                    if avg_growth > 0:
                        remaining = 1.0 - current_occ
                        days_to_exhaust[day] = remaining / avg_growth if avg_growth > 0 else np.inf

        for day in range(n_days):
            if days_to_exhaust[day] <= 3 and alert_levels[day] < 3:
                alert_levels[day] = 3

        alert_results[layer] = {
            "levels": alert_levels,
            "days_to_exhaust": days_to_exhaust,
            "threshold": threshold,
        }

    return alert_results


def compute_mortality(result, baseline_mortality=BASELINE_MORTALITY_RATE):
    n_days = len(result["days"])
    daily_new = result["daily_new_infections"]

    ideal_deaths_daily = daily_new * baseline_mortality
    ideal_deaths_cumulative = np.cumsum(ideal_deaths_daily)

    bed_data = result["bed"]
    icu_data = result["icu"]
    vent_data = result["ventilator"]

    actual_deaths_daily = np.zeros(n_days)

    for day in range(n_days):
        total_new = daily_new[day]
        if total_new <= 0:
            continue

        bed_adm = bed_data["admissions"][day]
        icu_adm = icu_data["admissions"][day]
        vent_adm = vent_data["admissions"][day]

        bed_untreated = bed_data["untreated"][day]
        vent_overflow_to_icu = vent_data["overflow_to_icu"][day]
        icu_overflow_to_bed = icu_data["overflow_to_bed"][day]

        deaths = 0.0

        mild_cases = total_new - bed_adm - icu_adm - vent_adm
        mild_cases = max(0.0, mild_cases)
        deaths += mild_cases * baseline_mortality

        vent_treated = vent_adm - vent_overflow_to_icu
        vent_treated = max(0.0, vent_treated)
        deaths += vent_treated * baseline_mortality

        if vent_overflow_to_icu > 0:
            deaths += vent_overflow_to_icu * baseline_mortality * 1.5

        icu_treated = icu_adm - icu_overflow_to_bed
        icu_treated = max(0.0, icu_treated)
        deaths += icu_treated * baseline_mortality

        if icu_overflow_to_bed > 0:
            deaths += icu_overflow_to_bed * baseline_mortality * 2.0

        bed_treated = bed_adm - bed_untreated
        bed_treated = max(0.0, bed_treated)
        deaths += bed_treated * baseline_mortality

        if bed_untreated > 0:
            deaths += bed_untreated * baseline_mortality * 3.0

        actual_deaths_daily[day] = deaths

    actual_deaths_cumulative = np.cumsum(actual_deaths_daily)

    excess_deaths = actual_deaths_cumulative[-1] - ideal_deaths_cumulative[-1]
    excess_pct = (excess_deaths / ideal_deaths_cumulative[-1] * 100) if ideal_deaths_cumulative[-1] > 0 else 0.0

    return {
        "ideal_daily": ideal_deaths_daily,
        "ideal_cumulative": ideal_deaths_cumulative,
        "actual_daily": actual_deaths_daily,
        "actual_cumulative": actual_deaths_cumulative,
        "excess_deaths": excess_deaths,
        "excess_pct": excess_pct,
    }


def compute_summary_metrics(result, alert_results, mortality_result):
    n_days = len(result["days"])

    first_red_alert = None
    for layer in ["ventilator", "icu", "bed"]:
        levels = alert_results[layer]["levels"]
        red_days = np.where(levels == 3)[0]
        if len(red_days) > 0:
            first_red = result["days"][red_days[0]]
            if first_red_alert is None or first_red < first_red_alert:
                first_red_alert = first_red

    max_gap_days = 0
    for layer in ["bed", "icu", "ventilator"]:
        occ = result[layer]["occupied"]
        cap = result[layer]["capacity"]
        gap_days = np.sum(occ >= cap)
        if gap_days > max_gap_days:
            max_gap_days = gap_days

    return {
        "excess_deaths": mortality_result["excess_deaths"],
        "excess_pct": mortality_result["excess_pct"],
        "resource_gap_peak_days": max_gap_days,
        "first_red_alert_day": first_red_alert,
    }


def run_healthcare_simulation(daily_new_infections, config=None,
                             age_stratified=False, age_props=None):
    if config is None:
        config = {}

    bed_capacity = config.get("bed_capacity", 5000)
    icu_capacity = config.get("icu_capacity", 500)
    ventilator_capacity = config.get("ventilator_capacity", 200)

    bed_threshold = config.get("bed_threshold", 0.80)
    icu_threshold = config.get("icu_threshold", 0.70)
    vent_threshold = config.get("vent_threshold", 0.60)

    base_hospitalization_rate = config.get("hospitalization_rate", 0.15)
    base_icu_rate = config.get("icu_rate", 0.03)
    base_ventilator_rate = config.get("ventilator_rate", 0.01)

    bed_stay_mean = config.get("bed_stay_mean", 10)
    icu_stay_mean = config.get("icu_stay_mean", 14)
    ventilator_stay_mean = config.get("ventilator_stay_mean", 21)
    stay_cv = config.get("stay_cv", 0.3)

    baseline_mortality = config.get("baseline_mortality", BASELINE_MORTALITY_RATE)

    total_days = len(daily_new_infections)

    flow_result = simulate_hospital_flow(
        daily_new_infections, total_days,
        hospitalization_rate=base_hospitalization_rate,
        icu_rate=base_icu_rate,
        ventilator_rate=base_ventilator_rate,
        bed_capacity=bed_capacity,
        icu_capacity=icu_capacity,
        ventilator_capacity=ventilator_capacity,
        bed_stay_mean=bed_stay_mean,
        icu_stay_mean=icu_stay_mean,
        ventilator_stay_mean=ventilator_stay_mean,
        stay_cv=stay_cv,
        age_stratified=age_stratified,
        age_props=age_props,
        base_hospitalization_rate=base_hospitalization_rate,
        base_icu_rate=base_icu_rate,
        base_ventilator_rate=base_ventilator_rate,
    )

    occupancy = compute_occupancy_rates(flow_result)

    capacities = {
        "bed": bed_capacity,
        "icu": icu_capacity,
        "ventilator": ventilator_capacity,
    }

    alert_results = compute_alert_levels(
        occupancy, capacities,
        bed_threshold=bed_threshold,
        icu_threshold=icu_threshold,
        vent_threshold=vent_threshold,
    )

    mortality_result = compute_mortality(flow_result, baseline_mortality)

    summary = compute_summary_metrics(flow_result, alert_results, mortality_result)

    return {
        "flow": flow_result,
        "occupancy": occupancy,
        "alerts": alert_results,
        "mortality": mortality_result,
        "summary": summary,
        "config": {
            "bed_capacity": bed_capacity,
            "icu_capacity": icu_capacity,
            "ventilator_capacity": ventilator_capacity,
            "bed_threshold": bed_threshold,
            "icu_threshold": icu_threshold,
            "vent_threshold": vent_threshold,
        },
    }
