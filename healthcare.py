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

DEFAULT_EXPANSION_CONFIG = {
    "bed": {"reserve": 10000, "period_days": 3, "trigger_occupancy": None},
    "icu": {"reserve": 500, "period_days": 5, "trigger_occupancy": None},
    "ventilator": {"reserve": 150, "period_days": 7, "trigger_occupancy": None},
}

DEFAULT_BORROW_CONFIG = {
    "enabled": True,
    "bed_to_icu_rate": 3.0,
    "bed_to_icu_convert_days": 1,
    "icu_to_vent_rate": 2.0,
    "icu_to_vent_convert_days": 2,
    "return_threshold": 0.70,
    "borrow_low_threshold": 0.40,
    "return_recovery_days": 1,
}

ALERT_LEVEL_GREEN = 0
ALERT_LEVEL_YELLOW = 1
ALERT_LEVEL_ORANGE = 2
ALERT_LEVEL_RED = 3
ALERT_LEVEL_PURPLE = 4


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


def _process_daily_discharges(day, admissions, mu, sigma):
    disch = 0.0
    for past_day in range(day):
        days_since_admit = day - past_day
        disch += admissions[past_day] * (
            _lognormal_cdf(days_since_admit + 0.5, mu, sigma) -
            _lognormal_cdf(days_since_admit - 0.5, mu, sigma)
        )
    return disch


def simulate_hospital_flow_dynamic(daily_new_infections, total_days,
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
                                   base_ventilator_rate=0.01,
                                   expansion_config=None,
                                   borrow_config=None,
                                   alert_thresholds=None):
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

    if expansion_config is None:
        expansion_config = dict(DEFAULT_EXPANSION_CONFIG)
    else:
        tmp = dict(DEFAULT_EXPANSION_CONFIG)
        for k in expansion_config:
            if k in tmp:
                tmp[k].update(expansion_config[k])
        expansion_config = tmp

    if borrow_config is None:
        borrow_config = dict(DEFAULT_BORROW_CONFIG)
    else:
        tmp = dict(DEFAULT_BORROW_CONFIG)
        tmp.update(borrow_config)
        borrow_config = tmp

    if alert_thresholds is None:
        alert_thresholds = {"bed": 0.80, "icu": 0.70, "ventilator": 0.60}

    initial_capacities = {
        "bed": bed_capacity,
        "icu": icu_capacity,
        "ventilator": ventilator_capacity,
    }

    effective_capacity = {
        "bed": np.zeros(n_days),
        "icu": np.zeros(n_days),
        "ventilator": np.zeros(n_days),
    }
    expansion_released = {
        "bed": np.zeros(n_days),
        "icu": np.zeros(n_days),
        "ventilator": np.zeros(n_days),
    }

    expansion_state = {}
    for layer in RESOURCE_LAYERS:
        expansion_state[layer] = {
            "started": False,
            "start_day": None,
            "days_elapsed": 0,
            "complete": False,
            "total_reserve": expansion_config[layer]["reserve"],
            "period_days": expansion_config[layer]["period_days"],
        }

    borrow_events = []
    return_events = []
    active_borrows = []

    converting_from_bed = 0.0
    converting_to_icu = 0.0
    bed_to_icu_convert_progress = 0
    bed_to_icu_borrowed = 0.0
    bed_to_icu_returning = 0.0
    bed_to_icu_return_progress = 0

    converting_from_icu = 0.0
    converting_to_vent = 0.0
    icu_to_vent_convert_progress = 0
    icu_to_vent_borrowed = 0.0
    icu_to_vent_returning = 0.0
    icu_to_vent_return_progress = 0

    schedule_events = []

    for day in range(n_days):
        if day > 0:
            bed_discharges[day] = _process_daily_discharges(
                day, bed_admissions, bed_mu, bed_sigma)
            icu_discharges[day] = _process_daily_discharges(
                day, icu_admissions, icu_mu, icu_sigma)
            vent_discharges[day] = _process_daily_discharges(
                day, vent_admissions, vent_mu, vent_sigma)

        if day == 0:
            for layer in RESOURCE_LAYERS:
                effective_capacity[layer][day] = initial_capacities[layer]
                expansion_released[layer][day] = 0.0
            continue

        for layer in RESOURCE_LAYERS:
            if expansion_state[layer]["started"] and not expansion_state[layer]["complete"]:
                expansion_state[layer]["days_elapsed"] += 1
                elapsed = expansion_state[layer]["days_elapsed"]
                total = expansion_state[layer]["total_reserve"]
                period = expansion_state[layer]["period_days"]
                released = min(total, total * elapsed / max(period, 1))
                expansion_released[layer][day] = released

                if elapsed >= period:
                    expansion_state[layer]["complete"] = True
                    schedule_events.append({
                        "day": day,
                        "type": "expansion_complete",
                        "layer": layer,
                    })
            else:
                expansion_released[layer][day] = expansion_released[layer][day - 1]

        if day > 0:
            prev_bed_occ = bed_occupied[day - 1]
            prev_icu_occ = icu_occupied[day - 1]
            prev_vent_occ = vent_occupied[day - 1]
            prev_bed_cap = effective_capacity["bed"][day - 1]
            prev_icu_cap = effective_capacity["icu"][day - 1]
            prev_vent_cap = effective_capacity["ventilator"][day - 1]

            bed_occ_rate = prev_bed_occ / prev_bed_cap if prev_bed_cap > 0 else 0
            icu_occ_rate = prev_icu_occ / prev_icu_cap if prev_icu_cap > 0 else 0
            vent_occ_rate = prev_vent_occ / prev_vent_cap if prev_vent_cap > 0 else 0

            for layer, occ_rate in [("bed", bed_occ_rate),
                                     ("icu", icu_occ_rate),
                                     ("ventilator", vent_occ_rate)]:
                threshold = alert_thresholds[layer]
                if (not expansion_state[layer]["started"]
                        and expansion_state[layer]["total_reserve"] > 0
                        and occ_rate >= threshold):
                    expansion_state[layer]["started"] = True
                    expansion_state[layer]["start_day"] = day
                    schedule_events.append({
                        "day": day,
                        "type": "expansion_start",
                        "layer": layer,
                    })

            if borrow_config["enabled"]:
                if bed_to_icu_borrowed > 0 and bed_occ_rate >= borrow_config["return_threshold"]:
                    if bed_to_icu_returning == 0:
                        bed_to_icu_returning = bed_to_icu_borrowed
                        bed_to_icu_return_progress = 0
                        bed_to_icu_borrowed = 0.0
                        schedule_events.append({
                            "day": day,
                            "type": "borrow_return",
                            "from_layer": "icu",
                            "to_layer": "bed",
                            "amount": bed_to_icu_returning,
                        })

                if icu_to_vent_borrowed > 0 and icu_occ_rate >= borrow_config["return_threshold"]:
                    if icu_to_vent_returning == 0:
                        icu_to_vent_returning = icu_to_vent_borrowed
                        icu_to_vent_return_progress = 0
                        icu_to_vent_borrowed = 0.0
                        schedule_events.append({
                            "day": day,
                            "type": "borrow_return",
                            "from_layer": "ventilator",
                            "to_layer": "icu",
                            "amount": icu_to_vent_returning,
                        })

                if (bed_to_icu_convert_progress == 0
                        and converting_from_bed == 0
                        and bed_to_icu_borrowed == 0
                        and bed_to_icu_returning == 0):
                    can_borrow_bed = (bed_occ_rate < borrow_config["borrow_low_threshold"]
                                      and icu_occ_rate >= alert_thresholds["icu"])
                    if can_borrow_bed:
                        available_bed = max(0.0, (initial_capacities["bed"]
                                                  + expansion_released["bed"][day])
                                                  * borrow_config["borrow_low_threshold"]
                                                  - prev_bed_occ)
                        rate = borrow_config["bed_to_icu_rate"]
                        if rate > 0 and available_bed >= rate:
                            convert_units = int(available_bed / rate)
                            if convert_units > 0:
                                converting_from_bed = convert_units * rate
                                converting_to_icu = convert_units
                                bed_to_icu_convert_progress = 1
                                schedule_events.append({
                                    "day": day,
                                    "type": "borrow_start",
                                    "from_layer": "bed",
                                    "to_layer": "icu",
                                    "amount": converting_to_icu,
                                })

                if (icu_to_vent_convert_progress == 0
                        and converting_from_icu == 0
                        and icu_to_vent_borrowed == 0
                        and icu_to_vent_returning == 0):
                    can_borrow_icu = (icu_occ_rate < borrow_config["borrow_low_threshold"]
                                      and vent_occ_rate >= alert_thresholds["ventilator"])
                    if can_borrow_icu:
                        available_icu = max(0.0, (initial_capacities["icu"]
                                                  + expansion_released["icu"][day])
                                                  * borrow_config["borrow_low_threshold"]
                                                  - prev_icu_occ)
                        rate = borrow_config["icu_to_vent_rate"]
                        if rate > 0 and available_icu >= rate:
                            convert_units = int(available_icu / rate)
                            if convert_units > 0:
                                converting_from_icu = convert_units * rate
                                converting_to_vent = convert_units
                                icu_to_vent_convert_progress = 1
                                schedule_events.append({
                                    "day": day,
                                    "type": "borrow_start",
                                    "from_layer": "icu",
                                    "to_layer": "ventilator",
                                    "amount": converting_to_vent,
                                })

                if bed_to_icu_convert_progress > 0:
                    bed_to_icu_convert_progress += 1
                    if bed_to_icu_convert_progress > borrow_config["bed_to_icu_convert_days"]:
                        bed_to_icu_borrowed = converting_to_icu
                        converting_from_bed = 0.0
                        converting_to_icu = 0.0
                        bed_to_icu_convert_progress = 0

                if icu_to_vent_convert_progress > 0:
                    icu_to_vent_convert_progress += 1
                    if icu_to_vent_convert_progress > borrow_config["icu_to_vent_convert_days"]:
                        icu_to_vent_borrowed = converting_to_vent
                        converting_from_icu = 0.0
                        converting_to_vent = 0.0
                        icu_to_vent_convert_progress = 0

                if bed_to_icu_returning > 0:
                    bed_to_icu_return_progress += 1
                    if bed_to_icu_return_progress > borrow_config["return_recovery_days"]:
                        bed_to_icu_returning = 0.0
                        bed_to_icu_return_progress = 0

                if icu_to_vent_returning > 0:
                    icu_to_vent_return_progress += 1
                    if icu_to_vent_return_progress > borrow_config["return_recovery_days"]:
                        icu_to_vent_returning = 0.0
                        icu_to_vent_return_progress = 0

        net_bed_borrow = -converting_from_bed
        if bed_to_icu_returning > 0:
            net_bed_borrow -= bed_to_icu_returning * borrow_config["bed_to_icu_rate"]

        net_icu_borrow = bed_to_icu_borrowed - converting_from_icu
        if converting_to_icu > 0:
            net_icu_borrow += 0
        if icu_to_vent_returning > 0:
            net_icu_borrow -= icu_to_vent_returning * borrow_config["icu_to_vent_rate"]

        net_vent_borrow = icu_to_vent_borrowed
        if converting_to_vent > 0:
            net_vent_borrow += 0

        bed_total_cap = (initial_capacities["bed"]
                         + expansion_released["bed"][day]
                         - converting_from_bed)
        if bed_to_icu_returning > 0:
            bed_total_cap -= bed_to_icu_returning * borrow_config["bed_to_icu_rate"]
        effective_capacity["bed"][day] = max(0.0, bed_total_cap)

        icu_total_cap = (initial_capacities["icu"]
                         + expansion_released["icu"][day]
                         + bed_to_icu_borrowed
                         - converting_from_icu)
        if icu_to_vent_returning > 0:
            icu_total_cap -= icu_to_vent_returning * borrow_config["icu_to_vent_rate"]
        effective_capacity["icu"][day] = max(0.0, icu_total_cap)

        vent_total_cap = (initial_capacities["ventilator"]
                          + expansion_released["ventilator"][day]
                          + icu_to_vent_borrowed)
        effective_capacity["ventilator"][day] = max(0.0, vent_total_cap)

        vent_new = vent_admissions[day]
        vent_after_discharge = max(0.0, vent_occupied[day - 1] - vent_discharges[day])
        vent_total = vent_after_discharge + vent_new
        vent_cap = effective_capacity["ventilator"][day]

        if vent_total <= vent_cap:
            vent_occupied[day] = vent_total
            vent_overflow_to_icu[day] = 0.0
        else:
            vent_occupied[day] = vent_cap
            vent_overflow = vent_total - vent_cap
            vent_overflow_to_icu[day] = vent_overflow

        icu_new = icu_admissions[day] + vent_overflow_to_icu[day]
        icu_after_discharge = max(0.0, icu_occupied[day - 1] - icu_discharges[day])
        icu_total = icu_after_discharge + icu_new
        icu_cap = effective_capacity["icu"][day]

        if icu_total <= icu_cap:
            icu_occupied[day] = icu_total
            icu_overflow_to_bed[day] = 0.0
        else:
            icu_occupied[day] = icu_cap
            icu_overflow = icu_total - icu_cap
            icu_overflow_to_bed[day] = icu_overflow

        bed_new = bed_admissions[day] + icu_overflow_to_bed[day]
        bed_after_discharge = max(0.0, bed_occupied[day - 1] - bed_discharges[day])
        bed_total = bed_after_discharge + bed_new
        bed_cap = effective_capacity["bed"][day]

        if bed_total <= bed_cap:
            bed_occupied[day] = bed_total
            bed_untreated[day] = 0.0
        else:
            bed_occupied[day] = bed_cap
            bed_untreated[day] = bed_total - bed_cap

    borrow_state_timeseries = {
        "bed_to_icu_borrowed": np.zeros(n_days),
        "icu_to_vent_borrowed": np.zeros(n_days),
        "bed_to_icu_converting": np.zeros(n_days),
        "icu_to_vent_converting": np.zeros(n_days),
        "bed_to_icu_returning": np.zeros(n_days),
        "icu_to_vent_returning": np.zeros(n_days),
    }

    expansion_final = {}
    for layer in RESOURCE_LAYERS:
        expansion_final[layer] = {
            "started": expansion_state[layer]["started"],
            "start_day": expansion_state[layer]["start_day"],
            "complete": expansion_state[layer]["complete"],
            "total_reserve": expansion_state[layer]["total_reserve"],
            "period_days": expansion_state[layer]["period_days"],
        }

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
            "effective_capacity": effective_capacity["bed"],
            "expansion_released": expansion_released["bed"],
        },
        "icu": {
            "admissions": icu_admissions,
            "discharges": icu_discharges,
            "occupied": icu_occupied,
            "capacity": icu_capacity,
            "overflow_from_ventilator": vent_overflow_to_icu,
            "overflow_to_bed": icu_overflow_to_bed,
            "effective_capacity": effective_capacity["icu"],
            "expansion_released": expansion_released["icu"],
        },
        "ventilator": {
            "admissions": vent_admissions,
            "discharges": vent_discharges,
            "occupied": vent_occupied,
            "capacity": ventilator_capacity,
            "overflow_to_icu": vent_overflow_to_icu,
            "effective_capacity": effective_capacity["ventilator"],
            "expansion_released": expansion_released["ventilator"],
        },
        "expansion": expansion_final,
        "expansion_released": expansion_released,
        "effective_capacity": effective_capacity,
        "schedule_events": schedule_events,
        "borrow_config": borrow_config,
        "borrow_state": {
            "bed_to_icu_borrowed": bed_to_icu_borrowed,
            "icu_to_vent_borrowed": icu_to_vent_borrowed,
            "bed_to_icu_converting": converting_from_bed,
            "icu_to_vent_converting": converting_from_icu,
        },
    }

    return result


def compute_occupancy_rates_dynamic(result):
    rates = {}
    for layer in ["bed", "icu", "ventilator"]:
        layer_data = result[layer]
        eff_cap = result["effective_capacity"][layer]
        occupied = layer_data["occupied"]
        rates[layer] = np.where(eff_cap > 0, occupied / eff_cap, 0.0)
    return rates


def compute_alert_levels_dynamic(occupancy_rates, result,
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
    n_days = len(occupancy_rates["bed"])

    for layer in ["bed", "icu", "ventilator"]:
        occ = occupancy_rates[layer]
        threshold = thresholds[layer]

        alert_levels = np.zeros(n_days, dtype=int)
        days_to_exhaust = np.full(n_days, np.inf)

        for day in range(n_days):
            current_occ = occ[day]

            if current_occ < threshold * 0.6:
                alert_levels[day] = ALERT_LEVEL_GREEN
            elif current_occ < threshold:
                alert_levels[day] = ALERT_LEVEL_YELLOW
            elif current_occ < 1.0:
                alert_levels[day] = ALERT_LEVEL_ORANGE
            else:
                alert_levels[day] = ALERT_LEVEL_RED

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
            if days_to_exhaust[day] <= 3 and alert_levels[day] < ALERT_LEVEL_RED:
                alert_levels[day] = ALERT_LEVEL_RED

        alert_results[layer] = {
            "levels": alert_levels,
            "days_to_exhaust": days_to_exhaust,
            "threshold": threshold,
        }

    system_collapse_days = np.zeros(n_days, dtype=bool)
    for day in range(n_days):
        orange_or_higher_count = 0
        for layer in ["bed", "icu", "ventilator"]:
            if alert_results[layer]["levels"][day] >= ALERT_LEVEL_ORANGE:
                orange_or_higher_count += 1

        if orange_or_higher_count >= 2:
            for layer in ["bed", "icu", "ventilator"]:
                current_level = alert_results[layer]["levels"][day]
                if current_level < ALERT_LEVEL_RED:
                    alert_results[layer]["levels"][day] = min(
                        ALERT_LEVEL_RED, current_level + 1)

        red_count = 0
        for layer in ["bed", "icu", "ventilator"]:
            if alert_results[layer]["levels"][day] >= ALERT_LEVEL_RED:
                red_count += 1

        if red_count >= 3:
            system_collapse_days[day] = True
            for layer in ["bed", "icu", "ventilator"]:
                alert_results[layer]["levels"][day] = ALERT_LEVEL_PURPLE

    for layer in ["bed", "icu", "ventilator"]:
        alert_results[layer]["system_collapse"] = system_collapse_days

    return alert_results


def compute_mortality_dynamic(result, alert_results,
                              baseline_mortality=BASELINE_MORTALITY_RATE):
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

        system_collapse = alert_results["bed"]["system_collapse"][day]
        collapse_multiplier = 1.5 if system_collapse else 1.0

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

        deaths *= collapse_multiplier
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


def compute_summary_metrics_dynamic(result, alert_results, mortality_result):
    n_days = len(result["days"])

    first_red_alert = None
    for layer in ["ventilator", "icu", "bed"]:
        levels = alert_results[layer]["levels"]
        red_days = np.where(levels >= 3)[0]
        if len(red_days) > 0:
            first_red = result["days"][red_days[0]]
            if first_red_alert is None or first_red < first_red_alert:
                first_red_alert = first_red

    max_gap_days = 0
    for layer in ["bed", "icu", "ventilator"]:
        occ = result[layer]["occupied"]
        eff_cap = result["effective_capacity"][layer]
        gap_days = np.sum(occ >= eff_cap)
        if gap_days > max_gap_days:
            max_gap_days = gap_days

    expansion_starts = {}
    expansion_count = 0
    for layer in RESOURCE_LAYERS:
        started = result["expansion"][layer]["started"]
        expansion_starts[layer] = started
        if started:
            expansion_count += 1

    schedule_events = result.get("schedule_events", [])
    borrow_start_count = sum(1 for e in schedule_events if e["type"] == "borrow_start")
    borrow_return_count = sum(1 for e in schedule_events if e["type"] == "borrow_return")

    net_borrow_days = 0
    bed_to_icu_active_days = 0
    icu_to_vent_active_days = 0

    for day in range(n_days):
        if alert_results["bed"]["system_collapse"][day]:
            break

    system_collapse_days = int(np.sum(alert_results["bed"]["system_collapse"]))

    events = result.get("schedule_events", [])
    bed_borrow_events = [e for e in events
                         if e["type"] == "borrow_start"
                         and e["from_layer"] == "bed"
                         and e["to_layer"] == "icu"]
    icu_borrow_events = [e for e in events
                         if e["type"] == "borrow_start"
                         and e["from_layer"] == "icu"
                         and e["to_layer"] == "ventilator"]

    borrow_count = len(bed_borrow_events) + len(icu_borrow_events)
    bed_borrow_count = len(bed_borrow_events)
    icu_borrow_count = len(icu_borrow_events)

    return {
        "excess_deaths": mortality_result["excess_deaths"],
        "excess_pct": mortality_result["excess_pct"],
        "resource_gap_peak_days": max_gap_days,
        "first_red_alert_day": first_red_alert,
        "expansion_total_count": expansion_count,
        "expansion_by_layer": expansion_starts,
        "borrow_total_count": borrow_count,
        "borrow_bed_to_icu_count": bed_borrow_count,
        "borrow_icu_to_vent_count": icu_borrow_count,
        "system_collapse_days": system_collapse_days,
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

    expansion_config = config.get("expansion_config", None)
    borrow_config = config.get("borrow_config", None)

    enable_expansion = config.get("enable_expansion", True)
    enable_borrow = config.get("enable_borrow", True)

    if not enable_expansion:
        expansion_config = {
            "bed": {"reserve": 0, "period_days": 3},
            "icu": {"reserve": 0, "period_days": 5},
            "ventilator": {"reserve": 0, "period_days": 7},
        }

    if not enable_borrow:
        borrow_config = {"enabled": False}

    total_days = len(daily_new_infections)

    alert_thresholds = {
        "bed": bed_threshold,
        "icu": icu_threshold,
        "ventilator": vent_threshold,
    }

    flow_result = simulate_hospital_flow_dynamic(
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
        expansion_config=expansion_config,
        borrow_config=borrow_config,
        alert_thresholds=alert_thresholds,
    )

    occupancy = compute_occupancy_rates_dynamic(flow_result)

    alert_results = compute_alert_levels_dynamic(
        occupancy, flow_result,
        bed_threshold=bed_threshold,
        icu_threshold=icu_threshold,
        vent_threshold=vent_threshold,
    )

    mortality_result = compute_mortality_dynamic(
        flow_result, alert_results, baseline_mortality)

    summary = compute_summary_metrics_dynamic(
        flow_result, alert_results, mortality_result)

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
            "enable_expansion": enable_expansion,
            "enable_borrow": enable_borrow,
        },
    }
