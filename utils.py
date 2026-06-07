"""
F1 Race Intelligence Engine — Utility Functions
================================================
All data processing, analysis, and ML helper functions.
"""

import numpy as np
import pandas as pd
import streamlit as st
from functools import lru_cache
from typing import Optional, Dict, List, Tuple, Any

# =========================================
# CONSTANTS
# =========================================

MODERN_ERA_START: int = 2010
RACES_PER_SEASON: int = 23
MONTE_CARLO_RUNS: int = 5000
ROLLING_WINDOW: int = 5
RECENT_RACES: int = 23  # ~1 season of recent data
RECENT_CONSTRUCTOR_RACES: int = 46  # ~2 seasons

# F1 Points table (2010+ era)
POINTS_TABLE: Dict[int, int] = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1,
}

# Sprint points table (2021+ era)
SPRINT_POINTS_TABLE: Dict[int, int] = {
    1: 8, 2: 7, 3: 6, 4: 5, 5: 4,
    6: 3, 7: 2, 8: 1,
}

# Status IDs that mean the car finished the race (even if lapped)
FINISHED_STATUS_IDS: set = {1, 11, 12, 13, 14, 15, 16, 17, 18, 19}
# statusId 1 = "Finished", 11-19 = "+1 Lap" through "+9 Laps"

# DNF cause categories for grouping
DNF_CATEGORIES: Dict[str, List[str]] = {
    "Mechanical": ["Engine", "Gearbox", "Transmission", "Clutch", "Hydraulics",
                   "Electrical", "Brakes", "Suspension", "Fuel pressure",
                   "Oil pressure", "Wheel", "Throttle", "Steering",
                   "Turbo", "Exhaust", "Differential", "Overheating",
                   "Mechanical", "Tyre puncture", "Fuel system",
                   "Oil leak", "Water leak", "Fuel leak", "Power Unit",
                   "ERS", "Battery", "Energy Store", "Control electronics",
                   "Heat shield fire", "Power loss", "Vibrations",
                   "Radiator", "Water pressure", "Undertray"],
    "Collision": ["Accident", "Collision", "Collision damage", "Damage",
                  "Debris", "Puncture", "Tyre"],
    "Driver": ["Spun off", "Driver Seat", "Illness", "Injury",
               "Seat", "Driver unwell", "Eye injury"],
    "Other": ["Disqualified", "Retired", "Withdrew", "Did not qualify",
              "Not classified", "Excluded", "107% Rule"],
}

# =========================================
# ENCODING FUNCTIONS
# =========================================

@lru_cache(maxsize=10)
def encode_weather(w: str) -> int:
    """Encode weather condition to numeric code."""
    return {"Dry": 0, "Mixed": 1, "Wet": 2}[w]

@lru_cache(maxsize=10)
def encode_tyre(t: str) -> int:
    """Encode tyre strategy to numeric code."""
    return {"Conservative": 0, "Balanced": 1, "Aggressive": 2}[t]

# =========================================
# ID LOOKUP HELPERS (DICT-BASED, O(1))
# =========================================

# Module-level lookup dictionaries (populated at load time)
_driver_name_to_id: Dict[str, int] = {}
_constructor_name_to_id: Dict[str, int] = {}
_circuit_name_to_id: Dict[str, int] = {}


def _build_lookup_dicts(drivers_df: pd.DataFrame,
                        constructors_df: pd.DataFrame,
                        circuits_df: pd.DataFrame) -> None:
    """Build O(1) lookup dictionaries from DataFrames. Called once at load time."""
    global _driver_name_to_id, _constructor_name_to_id, _circuit_name_to_id

    _driver_name_to_id = dict(zip(
        drivers_df["forename"] + " " + drivers_df["surname"],
        drivers_df["driverId"].astype(int)
    ))
    _constructor_name_to_id = dict(zip(
        constructors_df["name"],
        constructors_df["constructorId"].astype(int)
    ))
    _circuit_name_to_id = dict(zip(
        circuits_df["name"],
        circuits_df["circuitId"].astype(int)
    ))


def get_driver_id(driver_name: str, drivers_df: pd.DataFrame = None) -> Optional[int]:
    """Get driver ID from full name via O(1) dict lookup."""
    if _driver_name_to_id:
        return _driver_name_to_id.get(driver_name)
    # Fallback to DataFrame filter if dicts not built yet
    if drivers_df is not None:
        match = drivers_df[(drivers_df["forename"] + " " + drivers_df["surname"]) == driver_name]
        return int(match["driverId"].values[0]) if not match.empty else None
    return None


def get_constructor_id(constructor_name: str, constructors_df: pd.DataFrame = None) -> Optional[int]:
    """Get constructor ID from name via O(1) dict lookup."""
    if _constructor_name_to_id:
        return _constructor_name_to_id.get(constructor_name)
    if constructors_df is not None:
        match = constructors_df[constructors_df["name"] == constructor_name]
        return int(match["constructorId"].values[0]) if not match.empty else None
    return None


def get_circuit_id(circuit_name: str, circuits_df: pd.DataFrame = None) -> Optional[int]:
    """Get circuit ID from name via O(1) dict lookup."""
    if _circuit_name_to_id:
        return _circuit_name_to_id.get(circuit_name)
    if circuits_df is not None:
        match = circuits_df[circuits_df["name"] == circuit_name]
        return int(match["circuitId"].values[0]) if not match.empty else None
    return None

# =========================================
# POINTS & POSITION CALCULATION
# =========================================

def normalize_probability(prob: float) -> float:
    """Normalize probability to 0-100 percentage scale."""
    prob = float(prob)
    prob_pct = prob * 100 if 0 <= prob <= 1 else prob
    return min(max(prob_pct, 0.0), 100.0)


def official_points_for_position(position: int) -> int:
    """Get championship points for a finishing position (2010+ rules)."""
    return POINTS_TABLE.get(position, 0)


def expected_position(prob: float) -> int:
    """
    Map podium probability to expected finish position.
    Uses calibrated tiers based on historical F1 data distribution.
    """
    prob = normalize_probability(prob)
    if prob >= 90: return 1
    if prob >= 75: return 2
    if prob >= 60: return 3
    if prob >= 45: return 4
    if prob >= 35: return 5
    if prob >= 25: return 7
    if prob >= 15: return 9
    if prob >= 8:  return 12
    if prob >= 3:  return 15
    return 18


def projected_points(prob: float) -> float:
    """Calculate projected championship points from podium probability."""
    position = expected_position(prob)
    return float(official_points_for_position(position))

# =========================================
# STRATEGY ADJUSTMENT (VECTORIZED)
# =========================================

def strategy_adjustment(grid: int, weather: str, tyre: str,
                        pit: int, form: int, risk: int,
                        aggro: int, pressure: int) -> Tuple[float, Dict[str, float]]:
    """
    Apply rule-based strategy adjustments on top of ML prediction.

    Each effect is calibrated from historical F1 data analysis:
    - Weather: Dry conditions favour front-runners (+3%), wet increases chaos (-5%)
    - Tyre: Aggressive strategies gain ~3% via optimal pit windows
    - Pit Crew: Every point above average (5) adds ~1.1% via faster stops
    - Form: Recent podium rate deviation from baseline (50%), scaled at 0.2x
    - Reliability: Each % of failure risk reduces podium chance by 0.18%
    - Aggression: Overtaking tendency deviation from neutral (50), scaled at 0.08x
    - Teammate Pressure: Higher pressure slightly reduces focus, -0.07x per unit
    """
    effects: Dict[str, float] = {}

    effects["Weather"] = {"Dry": 3.0, "Mixed": 0.0, "Wet": -5.0}[weather]
    effects["Tyre"] = {"Conservative": -1.5, "Balanced": 1.0, "Aggressive": 3.0}[tyre]
    effects["Pit Crew"] = (pit - 5) * 1.1
    effects["Form"] = (form - 50) * 0.2
    effects["Reliability"] = -risk * 0.18
    effects["Aggression"] = (aggro - 50) * 0.08
    effects["Teammate Pressure"] = -pressure * 0.07

    if weather in ["Mixed", "Wet"]:
        if grid >= 10:
            effects["Chaos Bonus"] = 2.0
        elif grid <= 3:
            effects["Chaos Bonus"] = -1.0

    delta = sum(effects.values())
    return delta, effects


def strategy_adjustment_vectorized(grids: np.ndarray, weather: str, tyre: str,
                                   pit: int, form: int, risk: int,
                                   aggro: int, pressure: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Vectorized strategy adjustment for multiple grid positions at once.
    Returns array of deltas and a sample effects dict (for grid=1).
    """
    # Base effects (same for all grid positions)
    base_delta = (
        {"Dry": 3.0, "Mixed": 0.0, "Wet": -5.0}[weather] +
        {"Conservative": -1.5, "Balanced": 1.0, "Aggressive": 3.0}[tyre] +
        (pit - 5) * 1.1 +
        (form - 50) * 0.2 +
        (-risk * 0.18) +
        (aggro - 50) * 0.08 +
        (-pressure * 0.07)
    )

    deltas = np.full(len(grids), base_delta)

    # Chaos bonus (grid-dependent, only in Mixed/Wet)
    if weather in ["Mixed", "Wet"]:
        deltas[grids >= 10] += 2.0
        deltas[grids <= 3] -= 1.0

    # Build sample effects dict for display
    _, effects = strategy_adjustment(int(grids[0]) if len(grids) > 0 else 1,
                                     weather, tyre, pit, form, risk, aggro, pressure)
    return deltas, effects

# =========================================
# MONTE CARLO SIMULATION
# =========================================

def monte_carlo(prob: float, runs: int = MONTE_CARLO_RUNS) -> float:
    """
    Simulate race outcomes accounting for model uncertainty.
    Adds Gaussian noise to the base probability to model
    real-world variance (safety cars, weather changes, incidents).
    """
    noise = np.random.normal(0, max(prob * 0.10, 1.0), runs)
    noisy_probs = np.clip(prob + noise, 0, 100)
    outcomes = (np.random.rand(runs) * 100) < noisy_probs
    return round(outcomes.mean() * 100, 2)


def monte_carlo_batch(probs: np.ndarray, runs: int = MONTE_CARLO_RUNS) -> np.ndarray:
    """Vectorized Monte Carlo for multiple probabilities at once."""
    n = len(probs)
    noise = np.random.normal(0, 1, (runs, n)) * np.maximum(probs * 0.10, 1.0)
    noisy = np.clip(probs + noise, 0, 100)
    outcomes = (np.random.rand(runs, n) * 100) < noisy
    return np.round(outcomes.mean(axis=0) * 100, 2)

# =========================================
# DNF PREDICTION (USING STATUS DATA)
# =========================================

def predict_dnf_probability(reliability_risk: int,
                            constructor_dnf_rate: float = 0.15) -> float:
    """
    Predict DNF probability based on reliability risk and constructor history.

    Args:
        reliability_risk: User-supplied reliability risk (0-100)
        constructor_dnf_rate: Historical DNF rate for the constructor (0-1)

    Returns:
        DNF probability as a percentage (0-100)
    """
    risk_factor = reliability_risk / 100.0
    base_dnf_prob = constructor_dnf_rate
    adjusted_dnf = base_dnf_prob * (1 + risk_factor * 0.5)
    adjusted_dnf = min(max(adjusted_dnf * 100, 0), 100)
    return round(adjusted_dnf, 2)


def get_actual_dnf_rate(constructor_id: int, results_df: pd.DataFrame,
                        status_df: pd.DataFrame) -> float:
    """
    Calculate actual DNF rate for a constructor using status.csv data.
    Much more accurate than the old `points == 0` heuristic.
    """
    const_results = results_df[results_df["constructorId"] == constructor_id]
    if const_results.empty:
        return 0.15  # Default fallback

    total = len(const_results)
    finished = const_results["statusId"].isin(FINISHED_STATUS_IDS).sum()
    dnf_count = total - finished
    return dnf_count / total if total > 0 else 0.15

# =========================================
# DATA LOADING (OPTIMIZED)
# =========================================

@st.cache_data
def load_all_datasets():
    """
    Load all F1 datasets with modern era filtering.
    Now also loads: status, pit_stops, lap_times, sprint_results.
    Builds O(1) lookup dictionaries after loading.
    """
    # Core datasets
    circuits = pd.read_csv("datasets/circuits.csv")
    drivers = pd.read_csv("datasets/drivers.csv")
    constructors = pd.read_csv("datasets/constructors.csv")
    races = pd.read_csv("datasets/races.csv")
    results = pd.read_csv("datasets/results.csv")
    driver_standings = pd.read_csv("datasets/driver_standings.csv")
    constructor_standings = pd.read_csv("datasets/constructor_standings.csv")
    qualifying = pd.read_csv("datasets/qualifying.csv")

    # New datasets
    status = pd.read_csv("datasets/status.csv")
    pit_stops = pd.read_csv("datasets/pit_stops.csv")
    lap_times = pd.read_csv("datasets/lap_times.csv",
                            dtype={"milliseconds": "int32", "lap": "int16", "position": "int16"})
    sprint_results = pd.read_csv("datasets/sprint_results.csv")

    # Modern era filtering
    modern_races = races[races["year"] >= MODERN_ERA_START].copy()
    race_ids = set(modern_races["raceId"])

    modern_results = results[results["raceId"].isin(race_ids)].copy()
    driver_ids = set(modern_results["driverId"])
    constructor_ids = set(modern_results["constructorId"])
    circuit_ids = set(modern_races["circuitId"])

    drivers_modern = drivers[drivers["driverId"].isin(driver_ids)].copy()
    constructors_modern = constructors[constructors["constructorId"].isin(constructor_ids)].copy()
    circuits_modern = circuits[circuits["circuitId"].isin(circuit_ids)].copy()

    modern_qualifying = qualifying[qualifying["raceId"].isin(race_ids)].copy()
    modern_pit_stops = pit_stops[pit_stops["raceId"].isin(race_ids)].copy()
    modern_lap_times = lap_times[lap_times["raceId"].isin(race_ids)].copy()
    modern_sprint = sprint_results[sprint_results["raceId"].isin(race_ids)].copy()

    # Build O(1) lookup dicts
    _build_lookup_dicts(drivers_modern, constructors_modern, circuits_modern)

    return (
        drivers_modern, constructors_modern, circuits_modern,
        modern_races, modern_results, driver_standings,
        constructor_standings, modern_qualifying,
        status, modern_pit_stops, modern_lap_times, modern_sprint
    )

# =========================================
# HISTORICAL ANALYSIS FUNCTIONS
# =========================================

@st.cache_data
def driver_matchup_analysis(driver1_id: int, driver2_id: int,
                            modern_results: pd.DataFrame,
                            modern_races: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Compare two drivers in races they both competed in.
    Enhanced with race-by-race timeline data.
    """
    d1_results = modern_results[modern_results["driverId"] == driver1_id].copy()
    d2_results = modern_results[modern_results["driverId"] == driver2_id].copy()

    common_races = set(d1_results["raceId"]) & set(d2_results["raceId"])

    d1_common = d1_results[d1_results["raceId"].isin(common_races)].copy()
    d2_common = d2_results[d2_results["raceId"].isin(common_races)].copy()

    races_together = len(common_races)
    if races_together == 0:
        return None

    # Convert positionOrder to int
    d1_common["pos_int"] = pd.to_numeric(d1_common["positionOrder"], errors="coerce")
    d2_common["pos_int"] = pd.to_numeric(d2_common["positionOrder"], errors="coerce")

    d1_podiums = (d1_common["pos_int"] <= 3).sum()
    d2_podiums = (d2_common["pos_int"] <= 3).sum()
    d1_wins = (d1_common["pos_int"] == 1).sum()
    d2_wins = (d2_common["pos_int"] == 1).sum()
    d1_points = d1_common["points"].sum()
    d2_points = d2_common["points"].sum()

    # Race-by-race timeline
    d1_timeline = d1_common[["raceId", "pos_int", "points"]].rename(
        columns={"pos_int": "d1_pos", "points": "d1_points"})
    d2_timeline = d2_common[["raceId", "pos_int", "points"]].rename(
        columns={"pos_int": "d2_pos", "points": "d2_points"})

    timeline = d1_timeline.merge(d2_timeline, on="raceId", how="inner")
    timeline = timeline.merge(
        modern_races[["raceId", "date", "name", "year"]],
        on="raceId", how="left"
    ).sort_values("date")

    # Cumulative points
    timeline["d1_cum_points"] = timeline["d1_points"].cumsum()
    timeline["d2_cum_points"] = timeline["d2_points"].cumsum()
    timeline["d1_beat_d2"] = timeline["d1_pos"] < timeline["d2_pos"]

    return {
        "races_together": races_together,
        "d1_podiums": int(d1_podiums),
        "d2_podiums": int(d2_podiums),
        "d1_wins": int(d1_wins),
        "d2_wins": int(d2_wins),
        "d1_points": float(d1_points),
        "d2_points": float(d2_points),
        "d1_avg_finish": float(d1_common["pos_int"].mean()),
        "d2_avg_finish": float(d2_common["pos_int"].mean()),
        "d1_podium_rate": d1_podiums / races_together * 100,
        "d2_podium_rate": d2_podiums / races_together * 100,
        "timeline": timeline,
    }


@st.cache_data
def circuit_statistics(circuit_id: int, modern_results: pd.DataFrame,
                       modern_races: pd.DataFrame, drivers: pd.DataFrame,
                       constructors: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Get historical statistics for a specific circuit."""
    circuit_races = modern_races[modern_races["circuitId"] == circuit_id]["raceId"]
    circuit_results = modern_results[modern_results["raceId"].isin(circuit_races)].copy()

    if circuit_results.empty:
        return None

    # Merge with driver/constructor names
    circuit_results = circuit_results.merge(
        drivers[["driverId", "forename", "surname"]], on="driverId", how="left"
    )
    circuit_results = circuit_results.merge(
        constructors[["constructorId", "name"]], on="constructorId", how="left"
    )
    circuit_results["driver_name"] = circuit_results["forename"] + " " + circuit_results["surname"]

    # Wins and podiums
    winners = circuit_results[circuit_results["positionOrder"] == 1]
    podium = circuit_results[circuit_results["positionOrder"] <= 3]

    top_winners = winners["driver_name"].value_counts().head(5)
    top_constructors = winners["name"].value_counts().head(5)
    top_podium_drivers = podium["driver_name"].value_counts().head(5)

    avg_winner_grid = winners["grid"].astype(int).mean()

    return {
        "races_held": len(circuit_races),
        "top_winners": top_winners,
        "top_constructors": top_constructors,
        "top_podium_drivers": top_podium_drivers,
        "avg_winner_grid": avg_winner_grid,
        "total_races_data": circuit_results,
    }


@st.cache_data
def driver_trend_analysis(driver_id: int, modern_results: pd.DataFrame,
                          modern_races: pd.DataFrame,
                          drivers: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Analyze driver form trends over last 2 seasons."""
    driver_results = modern_results[modern_results["driverId"] == driver_id].copy()

    if driver_results.empty:
        return None

    driver_results = driver_results.merge(
        modern_races[["raceId", "date", "year", "round", "circuitId"]], on="raceId", how="left"
    )
    driver_results = driver_results.sort_values("date")

    driver_results["positionOrder_int"] = pd.to_numeric(driver_results["positionOrder"], errors="coerce")
    driver_results["is_podium"] = driver_results["positionOrder_int"] <= 3

    # Rolling stats (5-race window)
    driver_results["rolling_avg_finish"] = (
        driver_results["positionOrder_int"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    driver_results["rolling_podium_rate"] = (
        driver_results["is_podium"].rolling(ROLLING_WINDOW, min_periods=1).mean() * 100
    )
    driver_results["points_per_race"] = driver_results["points"]

    # Recent season (last ~23 races)
    recent = driver_results.tail(RECENT_RACES)

    # Circuit affinity
    circuit_performance = driver_results.groupby("circuitId").agg(
        avg_position=("positionOrder_int", "mean"),
        podium_rate=("is_podium", "mean")
    ).reset_index()

    circuit_performance = circuit_performance.merge(
        modern_races[["circuitId", "name"]].drop_duplicates(), on="circuitId", how="left"
    )

    best_circuits = circuit_performance.nsmallest(5, "avg_position")
    worst_circuits = circuit_performance.nlargest(5, "avg_position")

    return {
        "all_results": driver_results,
        "recent_season": recent,
        "best_circuits": best_circuits,
        "worst_circuits": worst_circuits,
        "total_races": len(driver_results),
    }


@st.cache_data
def constructor_analysis(constructor_id: int, modern_results: pd.DataFrame,
                         modern_races: pd.DataFrame, driver_standings: pd.DataFrame,
                         constructor_standings: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Analyze constructor performance trends."""
    const_results = modern_results[modern_results["constructorId"] == constructor_id].copy()

    if const_results.empty:
        return None

    const_results = const_results.merge(
        modern_races[["raceId", "date", "year"]], on="raceId", how="left"
    )
    const_results = const_results.sort_values("date")

    const_results["positionOrder_int"] = pd.to_numeric(const_results["positionOrder"], errors="coerce")

    # Use statusId for accurate DNF detection
    wins = (const_results["positionOrder_int"] == 1).sum()
    podiums = (const_results["positionOrder_int"] <= 3).sum()
    total_races = len(const_results)
    dnfs = (~const_results["statusId"].isin(FINISHED_STATUS_IDS)).sum()

    # Trend
    const_results["rolling_avg_points"] = const_results["points"].rolling(10, min_periods=1).mean()

    # Last 2 seasons
    recent = const_results.tail(RECENT_CONSTRUCTOR_RACES)

    return {
        "wins": int(wins),
        "podiums": int(podiums),
        "total_races": total_races,
        "dnfs": int(dnfs),
        "podium_rate": (podiums / total_races * 100) if total_races > 0 else 0,
        "win_rate": (wins / total_races * 100) if total_races > 0 else 0,
        "dnf_rate": (dnfs / total_races * 100) if total_races > 0 else 0,
        "all_results": const_results,
        "recent_results": recent,
    }

# =========================================
# SENSITIVITY ANALYSIS (OPTIMIZED)
# =========================================

def parameter_sensitivity_analysis(model, feature_cols: List[str],
                                   base_input: Dict, drivers_df: pd.DataFrame,
                                   constructors_df: pd.DataFrame,
                                   circuits_df: pd.DataFrame,
                                   weather: str, tyre: str,
                                   pit: int, form: int, risk: int,
                                   aggro: int, pressure: int) -> Dict[str, Dict]:
    """
    Analyze sensitivity of prediction to each strategy parameter.
    OPTIMIZED: Pre-computes base ML probabilities once (20 predictions instead of 500).
    """
    results_dict = {}

    # Pre-compute base ML probabilities for all grid positions ONCE
    sim_inputs = []
    for gp in range(1, 21):
        d = base_input.copy()
        d["grid"] = gp
        sim_inputs.append(d)
    sim_df = pd.DataFrame(sim_inputs).reindex(columns=feature_cols, fill_value=0)
    base_probs = model.predict_proba(sim_df)[:, 1] * 100 if model else np.full(20, 50)

    grids = np.arange(1, 21)

    ranges = {
        "pit": (1, 10),
        "form": (0, 100),
        "risk": (0, 100),
        "aggro": (0, 100),
        "pressure": (0, 100),
    }

    for param_name, (min_val, max_val) in ranges.items():
        param_values = np.linspace(min_val, max_val, 5).tolist()
        predictions = []

        for param_val in param_values:
            temp_params = {
                "pit": param_val if param_name == "pit" else pit,
                "form": param_val if param_name == "form" else form,
                "risk": param_val if param_name == "risk" else risk,
                "aggro": param_val if param_name == "aggro" else aggro,
                "pressure": param_val if param_name == "pressure" else pressure,
            }

            # Vectorized strategy adjustment
            deltas, _ = strategy_adjustment_vectorized(
                grids, weather, tyre,
                int(temp_params["pit"]), int(temp_params["form"]),
                int(temp_params["risk"]), int(temp_params["aggro"]),
                int(temp_params["pressure"])
            )
            final_probs = np.clip(base_probs + deltas, 0, 100)
            predictions.append(float(np.mean(final_probs)))

        results_dict[param_name] = {
            "values": param_values,
            "predictions": predictions,
        }

    return results_dict

# =========================================
# BATCH RACE SIMULATION (VECTORIZED)
# =========================================

def batch_race_simulation(model, feature_cols: List[str],
                          drivers_list: List[str], constructor: str,
                          circuit: str, grid_positions: List[int],
                          weather: str, tyre: str, pit: int, form: int,
                          risk: int, aggro: int, pressure: int,
                          drivers_df: pd.DataFrame,
                          constructors_df: pd.DataFrame,
                          circuits_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate multiple drivers across same conditions efficiently."""
    results = []
    constructor_id = get_constructor_id(constructor, constructors_df)
    circuit_id = get_circuit_id(circuit, circuits_df)

    grids = np.array(grid_positions)

    for driver_name in drivers_list:
        driver_id = get_driver_id(driver_name, drivers_df)
        if driver_id is None:
            continue

        base_input = {
            "circuitId": circuit_id,
            "constructorId": constructor_id,
            "driverId": driver_id,
            "weather_code": encode_weather(weather),
            "tyre_strategy_code": encode_tyre(tyre),
            "pit_crew_rating": pit,
            "recent_form": form,
            "reliability_risk": risk,
            "aggression_level": aggro,
            "teammate_pressure": pressure
        }

        # Vectorized prediction
        sim_inputs = []
        for gp in grid_positions:
            d = base_input.copy()
            d["grid"] = gp
            sim_inputs.append(d)

        sim_df = pd.DataFrame(sim_inputs).reindex(columns=feature_cols, fill_value=0)
        base_probs = model.predict_proba(sim_df)[:, 1] * 100 if model else np.full(len(grid_positions), 50)

        # Vectorized strategy deltas
        deltas, _ = strategy_adjustment_vectorized(
            grids, weather, tyre, pit, form, risk, aggro, pressure
        )
        final_probs = np.clip(base_probs + deltas, 0, 100)

        for i, gp in enumerate(grid_positions):
            results.append({
                "Driver": driver_name,
                "Grid": gp,
                "ML Prediction": round(float(base_probs[i]), 2),
                "Strategy Impact": round(float(deltas[i]), 2),
                "Final Prediction": round(float(final_probs[i]), 2),
                "Projected Points": projected_points(float(final_probs[i]))
            })

    return pd.DataFrame(results)

# =========================================
# QUALIFYING IMPACT ANALYSIS
# =========================================

@st.cache_data
def qualifying_impact_analysis(drivers_df: pd.DataFrame,
                               constructors_df: pd.DataFrame,
                               circuits_df: pd.DataFrame,
                               results_df: pd.DataFrame,
                               qualifying_df: pd.DataFrame,
                               races_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze impact of qualifying position on race results."""
    # Merge qualifying with race results
    merged = qualifying_df.merge(
        results_df[["raceId", "driverId", "positionOrder", "points"]],
        on=["raceId", "driverId"], how="inner"
    )
    merged = merged.merge(races_df[["raceId", "circuitId"]], on="raceId", how="left")

    merged["positionOrder"] = pd.to_numeric(merged["positionOrder"], errors="coerce")
    merged["position"] = pd.to_numeric(merged["position"], errors="coerce")

    # By qualifying position
    qual_impact = merged.groupby("position").agg(
        avg_finish_pos=("positionOrder", "mean"),
        std_finish_pos=("positionOrder", "std"),
        avg_points=("points", "mean"),
        races_count=("driverId", "count")
    ).round(2)

    # By circuit — compute position deltas
    circuit_groups = merged.groupby("circuitId").apply(
        lambda x: pd.Series({
            "avg_qual_to_finish_delta": (x["positionOrder"] - x["position"]).mean(),
            "overtakes_per_race": (x["position"] - x["positionOrder"]).clip(lower=0).mean()
        })
    ).reset_index()
    circuit_groups = circuit_groups.merge(
        circuits_df[["circuitId", "name"]], on="circuitId", how="left"
    )

    return qual_impact, circuit_groups

# =========================================
# STREAK ANALYSIS (FIXED)
# =========================================

@st.cache_data
def streak_analysis(driver_id: int, results_df: pd.DataFrame,
                    races_df: pd.DataFrame) -> Dict[str, int]:
    """
    Analyze consecutive podiums, DNFs, and wins.
    FIXED: Now correctly counts only consecutive True values,
    not longest streak of any value.
    """
    driver_results = results_df[results_df["driverId"] == driver_id].copy()
    driver_results = driver_results.merge(races_df[["raceId", "date"]], on="raceId")
    driver_results = driver_results.sort_values("date")

    driver_results["is_podium"] = pd.to_numeric(driver_results["positionOrder"], errors="coerce") <= 3
    driver_results["is_dnf"] = ~driver_results["statusId"].isin(FINISHED_STATUS_IDS)
    driver_results["is_win"] = pd.to_numeric(driver_results["positionOrder"], errors="coerce") == 1

    def get_longest_true_streak(series: pd.Series) -> int:
        """Count longest consecutive run of True values only."""
        if series.empty:
            return 0
        max_streak = 0
        current = 0
        for val in series:
            if val:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    def get_current_true_streak(series: pd.Series) -> int:
        """Count current consecutive True streak from the end."""
        if series.empty:
            return 0
        streak = 0
        for val in reversed(series.tolist()):
            if val:
                streak += 1
            else:
                break
        return streak

    return {
        "longest_podium_streak": get_longest_true_streak(driver_results["is_podium"]),
        "longest_dnf_streak": get_longest_true_streak(driver_results["is_dnf"]),
        "longest_win_streak": get_longest_true_streak(driver_results["is_win"]),
        "current_podium_streak": get_current_true_streak(driver_results["is_podium"]),
        "recent_podiums_10": int(driver_results["is_podium"].iloc[-10:].sum()) if len(driver_results) > 0 else 0,
    }

# =========================================
# SEASON PROJECTION
# =========================================

@st.cache_data
def season_projection(driver_standings: pd.DataFrame, races: pd.DataFrame,
                      modern_results: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Project final championship standings based on current data."""
    if driver_standings.empty or races.empty:
        return None

    latest_round = driver_standings["raceId"].max()
    current_standings = driver_standings[driver_standings["raceId"] == latest_round].copy()

    if current_standings.empty:
        return None

    current_standings = current_standings.merge(
        races[["raceId", "year"]], on="raceId", how="left"
    )

    current_year = current_standings["year"].iloc[0]
    total_races_planned = len(races[races["year"] == current_year])
    races_completed = len(races[(races["year"] == current_year) & (races["raceId"] <= latest_round)])

    if races_completed == 0:
        return None

    current_standings["avg_points_per_race"] = current_standings["points"] / races_completed
    current_standings["projected_final_points"] = (
        current_standings["points"] +
        current_standings["avg_points_per_race"] * (total_races_planned - races_completed)
    )

    current_standings = current_standings.sort_values("projected_final_points", ascending=False)
    current_standings["projected_position"] = range(1, len(current_standings) + 1)

    return {
        "standings": current_standings,
        "races_completed": races_completed,
        "total_races_planned": total_races_planned,
        "current_year": current_year,
    }

# =========================================
# NEW: LAP TIME ANALYSIS
# =========================================

@st.cache_data
def lap_time_analysis(driver_id: int, circuit_id: int,
                      lap_times_df: pd.DataFrame,
                      races_df: pd.DataFrame,
                      pit_stops_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Comprehensive lap time analysis for a driver at a circuit.

    Returns:
        - Average lap time
        - Best/worst lap
        - Consistency score (lower std = more consistent)
        - Tire degradation curve (lap time progression)
        - Stint analysis (between pit stops)
    """
    # Get races at this circuit
    circuit_races = races_df[races_df["circuitId"] == circuit_id]["raceId"]

    # Filter lap times for this driver at this circuit
    driver_laps = lap_times_df[
        (lap_times_df["driverId"] == driver_id) &
        (lap_times_df["raceId"].isin(circuit_races))
    ].copy()

    if driver_laps.empty:
        return None

    ms = driver_laps["milliseconds"]

    avg_lap = ms.mean()
    best_lap = ms.min()
    worst_lap = ms.max()
    consistency = ms.std()

    # Lap time progression (average by lap number)
    lap_progression = driver_laps.groupby("lap")["milliseconds"].mean()

    # Stint analysis: group laps by pit stops
    stint_data = []
    for race_id in driver_laps["raceId"].unique():
        race_laps = driver_laps[driver_laps["raceId"] == race_id].sort_values("lap")
        race_pits = pit_stops_df[
            (pit_stops_df["raceId"] == race_id) &
            (pit_stops_df["driverId"] == driver_id)
        ].sort_values("lap")

        pit_laps = race_pits["lap"].tolist() if not race_pits.empty else []
        stint_boundaries = [0] + pit_laps + [race_laps["lap"].max() + 1]

        for i in range(len(stint_boundaries) - 1):
            stint_laps = race_laps[
                (race_laps["lap"] > stint_boundaries[i]) &
                (race_laps["lap"] <= stint_boundaries[i + 1])
            ]
            if not stint_laps.empty:
                stint_data.append({
                    "race_id": race_id,
                    "stint": i + 1,
                    "laps": len(stint_laps),
                    "avg_ms": stint_laps["milliseconds"].mean(),
                    "degradation": (
                        stint_laps["milliseconds"].iloc[-1] - stint_laps["milliseconds"].iloc[0]
                        if len(stint_laps) > 1 else 0
                    )
                })

    stint_df = pd.DataFrame(stint_data) if stint_data else pd.DataFrame()

    # Average degradation per stint
    avg_degradation = stint_df["degradation"].mean() if not stint_df.empty else 0

    return {
        "avg_lap_ms": avg_lap,
        "best_lap_ms": best_lap,
        "worst_lap_ms": worst_lap,
        "consistency_ms": consistency,
        "total_laps": len(driver_laps),
        "races_analyzed": driver_laps["raceId"].nunique(),
        "lap_progression": lap_progression,
        "stint_data": stint_df,
        "avg_degradation_ms": avg_degradation,
    }


def format_lap_time(ms: float) -> str:
    """Format milliseconds to mm:ss.SSS."""
    minutes = int(ms // 60000)
    seconds = (ms % 60000) / 1000
    return f"{minutes}:{seconds:06.3f}"

# =========================================
# NEW: PIT STOP STRATEGY ANALYSIS
# =========================================

@st.cache_data
def pit_stop_analysis(circuit_id: int, pit_stops_df: pd.DataFrame,
                      results_df: pd.DataFrame, races_df: pd.DataFrame,
                      constructors_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Analyze pit stop strategies at a specific circuit.

    Returns:
        - Average pit stop duration by team
        - Optimal pit window
        - Strategy breakdown (1-stop vs 2-stop vs 3-stop)
        - Position delta from pit strategy
    """
    circuit_races = races_df[races_df["circuitId"] == circuit_id]["raceId"]
    circuit_pits = pit_stops_df[pit_stops_df["raceId"].isin(circuit_races)].copy()

    if circuit_pits.empty:
        return None

    # Convert duration to numeric
    circuit_pits["duration_ms"] = pd.to_numeric(circuit_pits["milliseconds"], errors="coerce")

    # Filter outliers (pit lane penalties, red flags, etc.) — keep under 60 seconds
    normal_pits = circuit_pits[circuit_pits["duration_ms"] < 60000].copy()

    # Average stop duration
    avg_stop = normal_pits["duration_ms"].mean()
    fastest_stop = normal_pits["duration_ms"].min()

    # Strategy breakdown: count stops per driver per race
    stops_per_driver = circuit_pits.groupby(["raceId", "driverId"])["stop"].max().reset_index()
    strategy_counts = stops_per_driver["stop"].value_counts().sort_index()
    strategy_pct = (strategy_counts / strategy_counts.sum() * 100).round(1)

    # Average pit stop duration by constructor
    circuit_pits_with_results = circuit_pits.merge(
        results_df[["raceId", "driverId", "constructorId"]],
        on=["raceId", "driverId"], how="left"
    )
    team_avg = circuit_pits_with_results.merge(
        constructors_df[["constructorId", "name"]], on="constructorId", how="left"
    )
    team_performance = team_avg.groupby("name").agg(
        avg_duration_ms=("duration_ms", "mean"),
        total_stops=("stop", "count"),
        fastest_stop_ms=("duration_ms", "min")
    ).sort_values("avg_duration_ms")

    # Pit window analysis: which laps are most common for stops
    pit_window = circuit_pits.groupby("lap").size().reset_index(name="count")

    # Strategy vs result: does more stops help?
    stops_with_results = stops_per_driver.merge(
        results_df[["raceId", "driverId", "positionOrder", "points"]],
        on=["raceId", "driverId"], how="left"
    )
    stops_with_results["positionOrder"] = pd.to_numeric(stops_with_results["positionOrder"], errors="coerce")
    strategy_results = stops_with_results.groupby("stop").agg(
        avg_finish=("positionOrder", "mean"),
        avg_points=("points", "mean"),
        count=("driverId", "count")
    ).round(2)

    return {
        "avg_stop_ms": avg_stop,
        "fastest_stop_ms": fastest_stop,
        "strategy_counts": strategy_counts,
        "strategy_pct": strategy_pct,
        "team_performance": team_performance,
        "pit_window": pit_window,
        "strategy_results": strategy_results,
        "total_stops_analyzed": len(circuit_pits),
    }

# =========================================
# NEW: RACE INCIDENTS / DNF ANALYSIS
# =========================================

@st.cache_data
def retirement_analysis(circuit_id: int, results_df: pd.DataFrame,
                        races_df: pd.DataFrame,
                        status_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Analyze race retirements and incidents at a circuit using status.csv.

    Returns:
        - DNF rate at circuit
        - Cause breakdown (mechanical, collision, driver, other)
        - Circuit danger rating
        - Most common retirement causes
    """
    circuit_races = races_df[races_df["circuitId"] == circuit_id]["raceId"]
    circuit_results = results_df[results_df["raceId"].isin(circuit_races)].copy()

    if circuit_results.empty:
        return None

    # Merge with status descriptions
    circuit_results = circuit_results.merge(status_df, on="statusId", how="left")

    total_entries = len(circuit_results)
    dnf_mask = ~circuit_results["statusId"].isin(FINISHED_STATUS_IDS)
    dnf_results = circuit_results[dnf_mask]
    dnf_count = len(dnf_results)
    dnf_rate = (dnf_count / total_entries * 100) if total_entries > 0 else 0

    # Cause breakdown by category
    def categorize_dnf(status_text: str) -> str:
        for category, keywords in DNF_CATEGORIES.items():
            if status_text in keywords:
                return category
        return "Other"

    dnf_results = dnf_results.copy()
    dnf_results["category"] = dnf_results["status"].apply(categorize_dnf)
    cause_breakdown = dnf_results["category"].value_counts()

    # Top specific causes
    top_causes = dnf_results["status"].value_counts().head(10)

    # DNF rate per race
    race_dnf_rates = circuit_results.groupby("raceId").apply(
        lambda x: (~x["statusId"].isin(FINISHED_STATUS_IDS)).mean() * 100
    ).reset_index(name="dnf_rate_pct")

    # Circuit danger rating (0-10 scale)
    avg_dnf_rate = dnf_rate
    mechanical_pct = (cause_breakdown.get("Mechanical", 0) / max(dnf_count, 1)) * 100
    collision_pct = (cause_breakdown.get("Collision", 0) / max(dnf_count, 1)) * 100
    danger_rating = min(10, (avg_dnf_rate * 0.3 + collision_pct * 0.05 + mechanical_pct * 0.02))

    return {
        "total_entries": total_entries,
        "dnf_count": dnf_count,
        "dnf_rate": round(dnf_rate, 2),
        "cause_breakdown": cause_breakdown,
        "top_causes": top_causes,
        "race_dnf_rates": race_dnf_rates,
        "danger_rating": round(danger_rating, 1),
        "races_analyzed": circuit_results["raceId"].nunique(),
    }

# =========================================
# NEW: CONSTRUCTOR DOMINANCE INDEX
# =========================================

@st.cache_data
def constructor_dominance(constructor_id: int, results_df: pd.DataFrame,
                          races_df: pd.DataFrame,
                          constructors_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Calculate constructor dominance metrics.

    Returns:
        - Points share per season
        - 1-2 finish count
        - Gap to next constructor
        - Season-by-season dominance score
    """
    const_results = results_df[results_df["constructorId"] == constructor_id].copy()

    if const_results.empty:
        return None

    const_results = const_results.merge(
        races_df[["raceId", "year"]], on="raceId", how="left"
    )
    const_results["positionOrder_int"] = pd.to_numeric(const_results["positionOrder"], errors="coerce")

    # Season-by-season analysis
    seasons = []
    for year, group in const_results.groupby("year"):
        year_total = results_df[results_df["raceId"].isin(
            races_df[races_df["year"] == year]["raceId"]
        )]["points"].sum()

        team_points = group["points"].sum()
        races_in_year = group["raceId"].nunique()
        wins = (group["positionOrder_int"] == 1).sum()
        podiums = (group["positionOrder_int"] <= 3).sum()

        # Count 1-2 finishes: races where team had P1 and P2
        race_positions = group.groupby("raceId")["positionOrder_int"].apply(
            lambda x: set(x.dropna().astype(int).tolist())
        )
        one_two_finishes = sum(1 for positions in race_positions if {1, 2}.issubset(positions))

        points_share = (team_points / year_total * 100) if year_total > 0 else 0
        dominance_score = (wins * 3 + podiums * 1 + one_two_finishes * 5) / max(races_in_year, 1)

        seasons.append({
            "year": int(year),
            "points": team_points,
            "points_share": round(points_share, 1),
            "wins": int(wins),
            "podiums": int(podiums),
            "one_two_finishes": one_two_finishes,
            "races": races_in_year,
            "dominance_score": round(dominance_score, 2),
        })

    seasons_df = pd.DataFrame(seasons).sort_values("year")

    # All-time totals
    total_one_two = seasons_df["one_two_finishes"].sum()
    peak_season = seasons_df.loc[seasons_df["dominance_score"].idxmax()] if not seasons_df.empty else None

    return {
        "seasons": seasons_df,
        "total_wins": int(seasons_df["wins"].sum()),
        "total_podiums": int(seasons_df["podiums"].sum()),
        "total_one_two": int(total_one_two),
        "avg_dominance_score": round(seasons_df["dominance_score"].mean(), 2),
        "peak_season": peak_season,
    }

# =========================================
# PDF REPORT GENERATION
# =========================================

def generate_pdf_report(report_data: Dict[str, Any]) -> Optional[bytes]:
    """Generate PDF report from race prediction data."""
    from io import BytesIO

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
    except ImportError:
        return None

    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#E10600"),
        spaceAfter=30,
        alignment=1,
    )

    elements = []

    # Title
    elements.append(Paragraph("F1 Race Simulation Report", title_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Prediction metrics
    elements.append(Paragraph("<b>Prediction Results</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))

    metrics_data = [
        ["Metric", "Value"],
        ["Driver", report_data.get("Driver", "N/A")],
        ["Grid Position", str(report_data.get("Grid", "N/A"))],
        ["ML Prediction", f"{report_data.get('ML Prediction', 0):.1f}%"],
        ["Strategy Impact", f"{report_data.get('Strategy Impact', 0):+.1f}%"],
        ["Final Prediction", f"{report_data.get('Final Prediction', 0):.1f}%"],
        ["Projected Points", str(report_data.get("Projected Points", 0))],
        ["DNF Probability", f"{report_data.get('DNF Probability', 0):.1f}%"],
    ]

    metrics_table = Table(metrics_data, colWidths=[2.5 * inch, 2.5 * inch])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E10600")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))

    elements.append(metrics_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Footer
    elements.append(Paragraph(
        "Generated by F1 Race Intelligence Engine",
        ParagraphStyle("footer", parent=styles["Normal"], fontSize=9, textColor=colors.grey)
    ))

    doc.build(elements)
    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()
