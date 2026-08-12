from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np
import pandas as pd


SIGMA = 5.670374419e-8
THERMOCOUPLE_POSITIONS_M = np.array([0.000, 0.015, 0.030, 0.045, 0.060, 0.075, 0.090, 0.105])
HOT_INTERFACE_M = 0.0375
COLD_INTERFACE_M = 0.0675
DEFAULT_DIAMETER_MM = 25.0
EXPECTED_CONDUCTIVITY_W_MK = {"Brass": 119.0, "Aluminium": 180.0}
EXPECTED_CONDUCTIVITY_RANGES_W_MK = {"Brass": (110.0, 128.0), "Aluminium": (162.0, 198.0)}

CONDUCTION_COLUMNS = [
    "Material",
    "Trial",
    "Voltage_V",
    "Current_A",
    "Water_flow_L_min",
    "T1_C",
    "T2_C",
    "T3_C",
    "T4_C",
    "T5_C",
    "T6_C",
    "T7_C",
    "T8_C",
]

RADIATION_COLUMNS = [
    "Case",
    "Fan",
    "Shield",
    "Air_velocity_m_s",
    "T6_air_C",
    "T7_polished_C",
    "T8_small_black_C",
    "T9_large_black_C",
    "T10_wall_C",
]


def blank_conduction_data() -> pd.DataFrame:
    rows = []
    for material in ("Brass", "Aluminium"):
        for trial, voltage in enumerate((7.0, 10.0), start=1):
            row = {column: np.nan for column in CONDUCTION_COLUMNS}
            row.update({"Material": material, "Trial": str(trial), "Voltage_V": voltage})
            rows.append(row)
    return normalise_conduction_data(pd.DataFrame(rows, columns=CONDUCTION_COLUMNS))


def _modelled_conduction_row(
    material: str,
    trial: str,
    voltage_V: float,
    current_A: float,
    water_flow_L_min: float,
    sample_conductivity_W_mK: float,
    hot_contact_Rpp_m2K_W: float,
    cold_contact_Rpp_m2K_W: float,
    cold_end_temperature_C: float,
    bar_conductivity_W_mK: float = 119.0,
) -> list[object]:
    """Build a self-consistent one-dimensional HT11C teaching row."""
    area = math.pi * (DEFAULT_DIAMETER_MM / 1000.0) ** 2 / 4.0
    heat_flux = voltage_V * current_A / area
    hot_bar_gradient = -heat_flux / bar_conductivity_W_mK
    sample_gradient = -heat_flux / sample_conductivity_W_mK
    cold_bar_gradient = -heat_flux / bar_conductivity_W_mK

    cold_bar_face = cold_end_temperature_C - cold_bar_gradient * (0.105 - COLD_INTERFACE_M)
    sample_cold_face = cold_bar_face + heat_flux * cold_contact_Rpp_m2K_W
    sample_hot_face = sample_cold_face - sample_gradient * (COLD_INTERFACE_M - HOT_INTERFACE_M)
    hot_bar_face = sample_hot_face + heat_flux * hot_contact_Rpp_m2K_W

    temperatures = []
    for position in THERMOCOUPLE_POSITIONS_M:
        if position < HOT_INTERFACE_M:
            temperature = hot_bar_face + hot_bar_gradient * (position - HOT_INTERFACE_M)
        elif position < COLD_INTERFACE_M:
            temperature = sample_hot_face + sample_gradient * (position - HOT_INTERFACE_M)
        else:
            temperature = cold_bar_face + cold_bar_gradient * (position - COLD_INTERFACE_M)
        temperatures.append(round(float(temperature), 2))
    return [
        material,
        str(trial),
        float(voltage_V),
        float(current_A),
        float(water_flow_L_min),
        *temperatures,
    ]


def demonstration_conduction_data() -> pd.DataFrame:
    """Controlled teaching data with realistic k and deliberately asymmetric contacts."""
    specifications = [
        ("Brass", "1", 7.0, 0.70, 1.50, 116.0, 1.30e-4, 4.20e-4, 24.6),
        ("Brass", "2", 10.0, 1.00, 1.50, 119.0, 1.20e-4, 3.80e-4, 25.0),
        ("Brass", "3", 12.0, 1.25, 1.45, 122.0, 1.10e-4, 3.40e-4, 25.5),
        ("Aluminium", "1", 7.0, 0.70, 1.50, 174.0, 1.60e-4, 4.80e-4, 24.6),
        ("Aluminium", "2", 10.0, 1.00, 1.50, 180.0, 1.45e-4, 4.40e-4, 25.0),
        ("Aluminium", "3", 12.0, 1.25, 1.45, 186.0, 1.30e-4, 4.00e-4, 25.5),
    ]
    rows = [_modelled_conduction_row(*specification) for specification in specifications]
    return normalise_conduction_data(pd.DataFrame(rows, columns=CONDUCTION_COLUMNS))


def blank_radiation_data() -> pd.DataFrame:
    rows = [
        ["1 - natural, exposed", "Off", "Down (exposed)", 0.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2 - natural, shielded", "Off", "Up (shielded)", 0.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["3 - forced, exposed", "On", "Down (exposed)", 4.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["4 - forced, shielded", "On", "Up (shielded)", 4.0, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]
    return normalise_radiation_data(pd.DataFrame(rows, columns=RADIATION_COLUMNS))


def _radiation_case_specifications() -> list[tuple[object, ...]]:
    """Return controlled cases and the effective surroundings seen by the beads."""
    return [
        # case, fan, shield, velocity, air, heated wall, effective Tsur, h(0.5 mm), h(3 mm)
        ("1 - natural, exposed", "Off", "Down (exposed)", 0.0, 23.5, 120.0, 120.0, 70.0, 35.0),
        ("2 - natural, shielded", "Off", "Up (shielded)", 0.0, 23.8, 121.0, 36.0, 70.0, 35.0),
        ("3 - forced, exposed", "On", "Down (exposed)", 4.0, 23.7, 80.0, 80.0, np.nan, np.nan),
        ("4 - forced, shielded", "On", "Up (shielded)", 4.0, 23.9, 79.0, 32.0, np.nan, np.nan),
    ]


def _modelled_radiation_row(specification: tuple[object, ...]) -> list[object]:
    case, fan, shield, velocity, air, wall, effective_surrounding, h_small, h_large = specification
    velocity = float(velocity)
    if velocity > 0:
        h_small = forced_convection_h(velocity, 0.5)[0]
        h_large = forced_convection_h(velocity, 3.0)[0]
    bead_temperatures = [
        equilibrium_sensor_temperature_C(float(air), float(effective_surrounding), float(h_small), 0.17),
        equilibrium_sensor_temperature_C(float(air), float(effective_surrounding), float(h_small), 0.98),
        equilibrium_sensor_temperature_C(float(air), float(effective_surrounding), float(h_large), 0.98),
    ]
    return [
        case,
        fan,
        shield,
        velocity,
        float(air),
        *(round(value, 2) for value in bead_temperatures),
        float(wall),
    ]


def demonstration_radiation_data() -> pd.DataFrame:
    """Controlled energy-balance data that isolate emissivity, size, shielding and airflow."""
    rows = [_modelled_radiation_row(specification) for specification in _radiation_case_specifications()]
    return normalise_radiation_data(pd.DataFrame(rows, columns=RADIATION_COLUMNS))


def normalise_conduction_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return the stable dtypes expected by Streamlit's data editor."""
    table = data.reindex(columns=CONDUCTION_COLUMNS).copy()
    table["Material"] = table["Material"].astype("string")
    table["Trial"] = table["Trial"].astype("string")
    for column in CONDUCTION_COLUMNS[2:]:
        table[column] = pd.to_numeric(table[column], errors="coerce").astype(float)
    return table


def normalise_radiation_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return the stable dtypes expected by Streamlit's data editor."""
    table = data.reindex(columns=RADIATION_COLUMNS).copy()
    for column in ("Case", "Fan", "Shield"):
        table[column] = table[column].astype("string")
    for column in RADIATION_COLUMNS[3:]:
        table[column] = pd.to_numeric(table[column], errors="coerce").astype(float)
    return table


def _online_variant(student_key: str, practical_tag: str) -> int:
    identity = (student_key or "unassigned").strip().lower()
    digest = hashlib.sha256(f"ME3512_THERMALLAB_2026::{practical_tag}::{identity}".encode("utf-8")).digest()
    return digest[0] % 4


def assigned_online_conduction_data(student_key: str) -> pd.DataFrame:
    """Create one of four stable, physically consistent online HT11C datasets."""
    table = demonstration_conduction_data().copy()
    variant = _online_variant(student_key, "conduction")
    gradient_scale = (0.985, 1.000, 1.015, 1.030)[variant]
    ambient_shift = (-0.6, 0.0, 0.5, 0.9)[variant]
    sensor_noise = np.array(
        [
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.04, -0.02, 0.02, 0.015, -0.015, 0.03, -0.02, 0.01],
            [-0.03, 0.03, -0.02, -0.010, 0.010, 0.02, 0.03, -0.03],
            [0.05, 0.01, -0.04, 0.012, -0.012, -0.01, -0.03, 0.02],
        ]
    )[variant]
    temperature_columns = [f"T{i}_C" for i in range(1, 9)]
    for row_index in table.index:
        base = table.loc[row_index, temperature_columns].astype(float).to_numpy()
        cold_reference = base[-1]
        adjusted = cold_reference + ambient_shift + gradient_scale * (base - cold_reference) + sensor_noise
        table.loc[row_index, temperature_columns] = np.round(adjusted, 2)
    table["Current_A"] = np.round(table["Current_A"] * gradient_scale, 3)
    table["Water_flow_L_min"] = np.round(table["Water_flow_L_min"] + (variant - 1.5) * 0.01, 2)
    return normalise_conduction_data(table)


def assigned_online_radiation_data(student_key: str) -> pd.DataFrame:
    """Create one of four stable, energy-balance-consistent online HT16C datasets."""
    variant = _online_variant(student_key, "radiation")
    ambient_shift = (-0.5, 0.0, 0.4, 0.8)[variant]
    wall_scale = (0.980, 1.000, 1.020, 1.035)[variant]
    h_scale = (1.035, 1.000, 0.975, 0.950)[variant]
    shield_scale = (0.96, 1.00, 1.04, 1.06)[variant]
    velocity_scale = (0.96, 1.00, 1.04, 1.02)[variant]
    bead_noise = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.04, -0.03, 0.02],
            [-0.03, 0.05, -0.02],
            [0.05, 0.01, -0.04],
        ]
    )[variant]
    rows = []
    for specification in _radiation_case_specifications():
        case, fan, shield, velocity, base_air, base_wall, base_surrounding, base_h_small, base_h_large = specification
        air = float(base_air) + ambient_shift
        wall = air + (float(base_wall) - float(base_air)) * wall_scale
        if "exposed" in str(shield).lower():
            effective_surrounding = wall
        else:
            effective_surrounding = air + (float(base_surrounding) - float(base_air)) * shield_scale
        velocity = float(velocity) * velocity_scale
        if velocity > 0:
            h_small = forced_convection_h(velocity, 0.5)[0] * h_scale
            h_large = forced_convection_h(velocity, 3.0)[0] * h_scale
        else:
            h_small = float(base_h_small) * h_scale
            h_large = float(base_h_large) * h_scale
        bead_temperatures = np.array(
            [
                equilibrium_sensor_temperature_C(air, effective_surrounding, h_small, 0.17),
                equilibrium_sensor_temperature_C(air, effective_surrounding, h_small, 0.98),
                equilibrium_sensor_temperature_C(air, effective_surrounding, h_large, 0.98),
            ]
        ) + bead_noise
        rows.append(
            [
                case,
                fan,
                shield,
                round(velocity, 2),
                round(air, 2),
                *(round(float(value), 2) for value in bead_temperatures),
                round(wall, 2),
            ]
        )
    table = pd.DataFrame(rows, columns=RADIATION_COLUMNS)
    return normalise_radiation_data(table)


def _numeric_row(row: pd.Series, columns: Iterable[str]) -> bool:
    return all(pd.notna(pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]) for column in columns)


def valid_conduction_rows(data: pd.DataFrame) -> pd.DataFrame:
    required = ["Voltage_V", "Current_A"] + [f"T{i}_C" for i in range(1, 9)]
    if data is None or data.empty or not set(required).issubset(data.columns):
        return pd.DataFrame(columns=CONDUCTION_COLUMNS)
    mask = data.apply(lambda row: _numeric_row(row, required), axis=1)
    valid = data.loc[mask].copy()
    for column in required + ["Water_flow_L_min"]:
        if column in valid:
            valid[column] = pd.to_numeric(valid[column], errors="coerce")
    return valid


def valid_radiation_rows(data: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Air_velocity_m_s",
        "T6_air_C",
        "T7_polished_C",
        "T8_small_black_C",
        "T9_large_black_C",
        "T10_wall_C",
    ]
    if data is None or data.empty or not set(required).issubset(data.columns):
        return pd.DataFrame(columns=RADIATION_COLUMNS)
    mask = data.apply(lambda row: _numeric_row(row, required), axis=1)
    valid = data.loc[mask].copy()
    for column in required:
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    return valid


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if total <= 1e-14 else 1.0 - residual / total
    return float(slope), float(intercept), float(r_squared)


def analyse_conduction(
    data: pd.DataFrame,
    diameter_mm: float = 25.0,
    heat_rate_fraction: float = 1.0,
) -> pd.DataFrame:
    """Analyse each complete HT11C row using three regional linear fits."""
    valid = valid_conduction_rows(data)
    if valid.empty or diameter_mm <= 0 or heat_rate_fraction <= 0:
        return pd.DataFrame()

    area = math.pi * (diameter_mm / 1000.0) ** 2 / 4.0
    results = []
    for source_index, row in valid.iterrows():
        temperatures = np.array([float(row[f"T{i}_C"]) for i in range(1, 9)])
        hot_slope, hot_intercept, hot_r2 = fit_line(THERMOCOUPLE_POSITIONS_M[:3], temperatures[:3])
        sample_slope, sample_intercept, sample_r2 = fit_line(
            THERMOCOUPLE_POSITIONS_M[3:5], temperatures[3:5]
        )
        cold_slope, cold_intercept, cold_r2 = fit_line(THERMOCOUPLE_POSITIONS_M[5:], temperatures[5:])

        hot_bar_face = hot_slope * HOT_INTERFACE_M + hot_intercept
        sample_hot_face = sample_slope * HOT_INTERFACE_M + sample_intercept
        sample_cold_face = sample_slope * COLD_INTERFACE_M + sample_intercept
        cold_bar_face = cold_slope * COLD_INTERFACE_M + cold_intercept

        electrical_power = float(row["Voltage_V"]) * float(row["Current_A"])
        heat_rate = electrical_power * heat_rate_fraction
        heat_flux = heat_rate / area
        delta_hot = hot_bar_face - sample_hot_face
        delta_sample = sample_hot_face - sample_cold_face
        delta_cold = sample_cold_face - cold_bar_face

        k_sample = abs(heat_flux / sample_slope) if abs(sample_slope) > 1e-12 else np.nan
        r_hot = delta_hot / heat_flux
        r_sample = delta_sample / heat_flux
        r_cold = delta_cold / heat_flux
        delta_sum = delta_hot + delta_sample + delta_cold
        contact_fraction = (
            100.0 * (delta_hot + delta_cold) / delta_sum if delta_sum > 1e-12 else np.nan
        )

        material = str(row.get("Material", ""))
        expected_conductivity = EXPECTED_CONDUCTIVITY_W_MK.get(material, np.nan)
        expected_range = EXPECTED_CONDUCTIVITY_RANGES_W_MK.get(material)
        deviation_from_expected = (
            100.0 * (k_sample - expected_conductivity) / expected_conductivity
            if math.isfinite(float(k_sample)) and math.isfinite(float(expected_conductivity))
            else np.nan
        )

        flags = []
        if not np.all(np.diff(temperatures) <= 0.5):
            flags.append("temperature is not consistently decreasing")
        if hot_r2 < 0.95:
            flags.append("hot-section points are not strongly linear")
        if cold_r2 < 0.95:
            flags.append("cold-section points are not strongly linear")
        if delta_hot < 0 or delta_cold < 0:
            flags.append("a fitted contact jump is negative")
        if heat_rate <= 0:
            flags.append("heat rate is not positive")
        if expected_range and math.isfinite(float(k_sample)) and not expected_range[0] <= k_sample <= expected_range[1]:
            flags.append(
                f"calculated k is outside the teaching reference range {expected_range[0]:.0f}-{expected_range[1]:.0f} W/(m·K)"
            )

        results.append(
            {
                "Source_row": int(source_index),
                "Material": str(row.get("Material", "")),
                "Trial": row.get("Trial", ""),
                "Voltage_V": float(row["Voltage_V"]),
                "Current_A": float(row["Current_A"]),
                "Electrical_power_W": electrical_power,
                "Assumed_conduction_heat_W": heat_rate,
                "Area_m2": area,
                "Heat_flux_W_m2": heat_flux,
                "Hot_bar_face_C": hot_bar_face,
                "Sample_hot_face_C": sample_hot_face,
                "Sample_cold_face_C": sample_cold_face,
                "Cold_bar_face_C": cold_bar_face,
                "Hot_contact_jump_K": delta_hot,
                "Sample_drop_K": delta_sample,
                "Cold_contact_jump_K": delta_cold,
                "Hot_contact_Rpp_m2K_W": r_hot,
                "Sample_Rpp_m2K_W": r_sample,
                "Cold_contact_Rpp_m2K_W": r_cold,
                "Thermal_conductivity_W_mK": k_sample,
                "Expected_conductivity_W_mK": expected_conductivity,
                "Deviation_from_expected_pct": deviation_from_expected,
                "Contact_share_pct": contact_fraction,
                "Hot_slope_K_m": hot_slope,
                "Sample_slope_K_m": sample_slope,
                "Cold_slope_K_m": cold_slope,
                "Hot_intercept_C": hot_intercept,
                "Sample_intercept_C": sample_intercept,
                "Cold_intercept_C": cold_intercept,
                "Hot_fit_R2": hot_r2,
                "Sample_fit_R2": sample_r2,
                "Cold_fit_R2": cold_r2,
                "Quality_flags": "; ".join(flags) if flags else "No automatic flags",
            }
        )
    return pd.DataFrame(results)


def conduction_uncertainty_components(
    voltage: float,
    current: float,
    delta_temperature: float,
    diameter_mm: float,
    spacing_mm: float,
    voltage_uncertainty: float,
    current_uncertainty: float,
    temperature_uncertainty: float,
    diameter_uncertainty_mm: float,
    spacing_uncertainty_mm: float,
) -> dict[str, float]:
    """Return relative standard-uncertainty terms for k = 4VI L/(pi D^2 deltaT)."""
    if min(abs(voltage), abs(current), abs(delta_temperature), diameter_mm, spacing_mm) <= 0:
        return {
            "Voltage": float("nan"),
            "Current": float("nan"),
            "Sensor spacing": float("nan"),
            "Diameter": float("nan"),
            "Temperature difference": float("nan"),
            "Combined": float("nan"),
        }
    delta_delta_t = math.sqrt(2.0) * abs(temperature_uncertainty)
    components = {
        "Voltage": voltage_uncertainty / abs(voltage),
        "Current": current_uncertainty / abs(current),
        "Sensor spacing": spacing_uncertainty_mm / spacing_mm,
        "Diameter": 2.0 * diameter_uncertainty_mm / diameter_mm,
        "Temperature difference": delta_delta_t / abs(delta_temperature),
    }
    components["Combined"] = math.sqrt(sum(term**2 for term in components.values()))
    return components


def conduction_uncertainty_percent(
    voltage: float,
    current: float,
    delta_temperature: float,
    diameter_mm: float,
    spacing_mm: float,
    voltage_uncertainty: float,
    current_uncertainty: float,
    temperature_uncertainty: float,
    diameter_uncertainty_mm: float,
    spacing_uncertainty_mm: float,
) -> float:
    """RSS relative uncertainty in k, returned as a percentage."""
    components = conduction_uncertainty_components(
        voltage,
        current,
        delta_temperature,
        diameter_mm,
        spacing_mm,
        voltage_uncertainty,
        current_uncertainty,
        temperature_uncertainty,
        diameter_uncertainty_mm,
        spacing_uncertainty_mm,
    )
    return 100.0 * components["Combined"]


def analyse_radiation(data: pd.DataFrame) -> pd.DataFrame:
    valid = valid_radiation_rows(data)
    if valid.empty:
        return pd.DataFrame()
    analysed = valid.copy()
    analysed["T7_error_K"] = analysed["T7_polished_C"] - analysed["T6_air_C"]
    analysed["T8_error_K"] = analysed["T8_small_black_C"] - analysed["T6_air_C"]
    analysed["T9_error_K"] = analysed["T9_large_black_C"] - analysed["T6_air_C"]
    analysed["Mean_sensor_error_K"] = analysed[["T7_error_K", "T8_error_K", "T9_error_K"]].mean(axis=1)
    analysed["Maximum_abs_error_K"] = analysed[["T7_error_K", "T8_error_K", "T9_error_K"]].abs().max(axis=1)
    return analysed


def forced_convection_h(
    air_velocity_m_s: float,
    bead_diameter_mm: float,
    air_kinematic_viscosity_m2_s: float = 15.9e-6,
    air_conductivity_W_mK: float = 0.0263,
    prandtl: float = 0.707,
) -> tuple[float, float, float]:
    """Ranz-Marshall estimate for crossflow over a small spherical bead."""
    diameter_m = bead_diameter_mm / 1000.0
    if air_velocity_m_s <= 0 or diameter_m <= 0:
        return float("nan"), float("nan"), float("nan")
    reynolds = air_velocity_m_s * diameter_m / air_kinematic_viscosity_m2_s
    nusselt = 2.0 + 0.6 * math.sqrt(reynolds) * prandtl ** (1.0 / 3.0)
    h = nusselt * air_conductivity_W_mK / diameter_m
    return h, reynolds, nusselt


def radiation_corrected_medium_temperature_C(
    thermocouple_temperature_C: float,
    surrounding_temperature_C: float,
    h_W_m2K: float,
    emissivity: float,
) -> float:
    if h_W_m2K <= 0 or not 0 <= emissivity <= 1:
        return float("nan")
    t_th = thermocouple_temperature_C + 273.15
    t_sur = surrounding_temperature_C + 273.15
    t_medium = t_th + emissivity * SIGMA * (t_th**4 - t_sur**4) / h_W_m2K
    return t_medium - 273.15


def linearised_radiation_coefficient_W_m2K(
    thermocouple_temperature_C: float,
    surrounding_temperature_C: float,
    emissivity: float,
) -> float:
    t_th = thermocouple_temperature_C + 273.15
    t_sur = surrounding_temperature_C + 273.15
    return emissivity * SIGMA * (t_th + t_sur) * (t_th**2 + t_sur**2)


def equilibrium_sensor_temperature_C(
    medium_temperature_C: float,
    surrounding_temperature_C: float,
    h_W_m2K: float,
    emissivity: float,
) -> float:
    """Solve h(Tm-Tth) = eps*sigma(Tth^4-Tsur^4) by bisection."""
    if h_W_m2K <= 0 or not 0 <= emissivity <= 1:
        return float("nan")
    if emissivity == 0 or medium_temperature_C == surrounding_temperature_C:
        return float(medium_temperature_C)

    tm = medium_temperature_C + 273.15
    ts = surrounding_temperature_C + 273.15

    def residual(tth: float) -> float:
        return h_W_m2K * (tm - tth) - emissivity * SIGMA * (tth**4 - ts**4)

    lower, upper = sorted((tm, ts))
    for _ in range(120):
        midpoint = 0.5 * (lower + upper)
        value = residual(midpoint)
        if abs(value) < 1e-10:
            return midpoint - 273.15
        if residual(lower) * value <= 0:
            upper = midpoint
        else:
            lower = midpoint
    return 0.5 * (lower + upper) - 273.15
