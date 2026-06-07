import numpy as np
import pandas as pd
import streamlit as st

# =========================================
# ENCODING FUNCTIONS
# =========================================

def encode_weather(w):
    return {"Dry": 0, "Mixed": 1, "Wet": 2}[w]

def encode_tyre(t):
    return {"Conservative": 0, "Balanced": 1, "Aggressive": 2}[t]

# =========================================
# POINTS & POSITION CALCULATION
# =========================================

def normalize_probability(prob):
    prob = float(prob)
    prob_pct = prob * 100 if 0 <= prob <= 1 else prob
    return min(max(prob_pct, 0.0), 100.0)

def official_points_for_position(position):
    points_table = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1,
    }
    return points_table.get(position, 0)

def expected_position(prob):
    prob = normalize_probability(prob)
    if prob >= 90: return 1
    if prob >= 75: return 2
    if prob >= 60: return 3
    if prob >= 45: return 4
    if prob >= 35: return 5
    if prob >= 25: return 7
    if prob >= 15: return 9
    if prob >= 8: return 12
    if prob >= 3: return 15
    return 18

def projected_points(prob):
    position = expected_position(prob)
    return float(official_points_for_position(position))

# =========================================
# STRATEGY ADJUSTMENT
# =========================================

def strategy_adjustment(grid, weather, tyre, pit, form, risk, aggro, pressure):
    """Apply rule-based strategy adjustments on top of ML prediction."""
    effects = {}
    effects["Weather"] = {"Dry": 3, "Mixed": 0, "Wet": -5}[weather]
    effects["Tyre"] = {"Conservative": -1.5, "Balanced": 1, "Aggressive": 3}[tyre]
    effects["Pit Crew"] = (pit - 5) * 1.1
    effects["Form"] = (form - 50) * 0.2
    effects["Reliability"] = -risk * 0.18
    effects["Aggression"] = (aggro - 50) * 0.08
    effects["Teammate Pressure"] = -pressure * 0.07

    if weather in ["Mixed", "Wet"]:
        if grid >= 10:
            effects["Chaos Bonus"] = 2
        elif grid <= 3:
            effects["Chaos Bonus"] = -1

    delta = sum(effects.values())
    return delta, effects

# =========================================
# MONTE CARLO SIMULATION
# =========================================

def monte_carlo(prob, runs=5000):
    """Simulate race outcomes with Gaussian noise for uncertainty."""
    noise = np.random.normal(0, max(prob * 0.10, 1.0), runs)
    noisy_probs = np.clip(prob + noise, 0, 100)
    outcomes = (np.random.rand(runs) * 100) < noisy_probs
    return round(outcomes.mean() * 100, 2)

# =========================================
# DATA LOADING (OPTIMIZED)
# =========================================

@st.cache_data
def load_all_datasets():
    """Load all F1 datasets with modern era filtering."""
    circuits = pd.read_csv("datasets/circuits.csv")
    drivers = pd.read_csv("datasets/drivers.csv")
    constructors = pd.read_csv("datasets/constructors.csv")
    races = pd.read_csv("datasets/races.csv")
    results = pd.read_csv("datasets/results.csv")
    driver_standings = pd.read_csv("datasets/driver_standings.csv")
    constructor_standings = pd.read_csv("datasets/constructor_standings.csv")
    qualifying = pd.read_csv("datasets/qualifying.csv")

    modern_races = races[races["year"] >= 2010].copy()
    race_ids = set(modern_races["raceId"])

    modern_results = results[results["raceId"].isin(race_ids)].copy()
    driver_ids = set(modern_results["driverId"])
    constructor_ids = set(modern_results["constructorId"])
    circuit_ids = set(modern_races["circuitId"])

    drivers_modern = drivers[drivers["driverId"].isin(driver_ids)].copy()
    constructors_modern = constructors[constructors["constructorId"].isin(constructor_ids)].copy()
    circuits_modern = circuits[circuits["circuitId"].isin(circuit_ids)].copy()

    modern_qualifying = qualifying[qualifying["raceId"].isin(race_ids)].copy()

    return (
        drivers_modern, constructors_modern, circuits_modern,
        modern_races, modern_results, driver_standings,
        constructor_standings, modern_qualifying
    )

# =========================================
# HISTORICAL ANALYSIS FUNCTIONS
# =========================================

@st.cache_data
def driver_matchup_analysis(driver1_id, driver2_id, modern_results, modern_races):
    """Compare two drivers in races they both competed in."""
    d1_results = modern_results[modern_results["driverId"] == driver1_id].copy()
    d2_results = modern_results[modern_results["driverId"] == driver2_id].copy()

    common_races = set(d1_results["raceId"]) & set(d2_results["raceId"])

    d1_common = d1_results[d1_results["raceId"].isin(common_races)].copy()
    d2_common = d2_results[d2_results["raceId"].isin(common_races)].copy()

    races_together = len(common_races)

    if races_together == 0:
        return None

    # Podium: positionOrder <= 3 or positionOrder is '1', '2', '3'
    def is_podium(pos):
        try:
            return int(pos) <= 3
        except:
            return pos in ['1', '2', '3']

    d1_podiums = d1_common[d1_common["positionOrder"].apply(is_podium)].shape[0]
    d2_podiums = d2_common[d2_common["positionOrder"].apply(is_podium)].shape[0]

    d1_wins = d1_common[d1_common["positionOrder"] == 1].shape[0]
    d2_wins = d2_common[d2_common["positionOrder"] == 1].shape[0]

    d1_points = d1_common["points"].sum()
    d2_points = d2_common["points"].sum()

    return {
        "races_together": races_together,
        "d1_podiums": d1_podiums,
        "d2_podiums": d2_podiums,
        "d1_wins": d1_wins,
        "d2_wins": d2_wins,
        "d1_points": d1_points,
        "d2_points": d2_points,
        "d1_avg_finish": d1_common["positionOrder"].astype(int).mean(),
        "d2_avg_finish": d2_common["positionOrder"].astype(int).mean(),
        "d1_podium_rate": d1_podiums / races_together * 100,
        "d2_podium_rate": d2_podiums / races_together * 100,
    }

@st.cache_data
def circuit_statistics(circuit_id, modern_results, modern_races, drivers, constructors):
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
    top_constructors = circuit_results[circuit_results["positionOrder"] == 1]["name"].value_counts().head(5)
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
def driver_trend_analysis(driver_id, modern_results, modern_races, drivers):
    """Analyze driver form trends over last 2 seasons."""
    driver_results = modern_results[modern_results["driverId"] == driver_id].copy()

    if driver_results.empty:
        return None

    driver_results = driver_results.merge(
        modern_races[["raceId", "date", "year", "round"]], on="raceId", how="left"
    )
    driver_results = driver_results.sort_values("date")

    driver_results["positionOrder_int"] = pd.to_numeric(driver_results["positionOrder"], errors="coerce")
    driver_results["is_podium"] = driver_results["positionOrder_int"] <= 3

    # Rolling stats (5-race window)
    driver_results["rolling_avg_finish"] = driver_results["positionOrder_int"].rolling(5, min_periods=1).mean()
    driver_results["rolling_podium_rate"] = driver_results["is_podium"].rolling(5, min_periods=1).mean() * 100
    driver_results["points_per_race"] = driver_results["points"]

    # Recent season (last ~23 races)
    recent = driver_results.tail(23)

    # Circuit affinity
    circuit_performance = driver_results.groupby("circuitId").agg({
        "positionOrder_int": "mean",
        "is_podium": "mean"
    }).reset_index()
    circuit_performance.columns = ["circuitId", "avg_position", "podium_rate"]
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
def constructor_analysis(constructor_id, modern_results, modern_races, driver_standings, constructor_standings):
    """Analyze constructor performance trends."""
    const_results = modern_results[modern_results["constructorId"] == constructor_id].copy()

    if const_results.empty:
        return None

    const_results = const_results.merge(
        modern_races[["raceId", "date", "year"]], on="raceId", how="left"
    )
    const_results = const_results.sort_values("date")

    const_results["positionOrder_int"] = pd.to_numeric(const_results["positionOrder"], errors="coerce")

    # Stats
    wins = (const_results["positionOrder_int"] == 1).sum()
    podiums = (const_results["positionOrder_int"] <= 3).sum()
    total_races = len(const_results)
    dnfs = (const_results["points"] == 0).sum()

    # Trend
    const_results["rolling_avg_points"] = const_results["points"].rolling(10, min_periods=1).mean()

    # Last 2 seasons
    recent = const_results.tail(46)  # ~2 seasons

    return {
        "wins": wins,
        "podiums": podiums,
        "total_races": total_races,
        "dnfs": dnfs,
        "podium_rate": (podiums / total_races * 100) if total_races > 0 else 0,
        "win_rate": (wins / total_races * 100) if total_races > 0 else 0,
        "dnf_rate": (dnfs / total_races * 100) if total_races > 0 else 0,
        "all_results": const_results,
        "recent_results": recent,
    }

# =========================================
# SENSITIVITY ANALYSIS
# =========================================

def parameter_sensitivity_analysis(model, feature_cols, base_input, drivers_df,
                                   constructors_df, circuits_df, weather, tyre,
                                   pit, form, risk, aggro, pressure):
    """Analyze sensitivity of prediction to each strategy parameter."""
    results_dict = {}

    # Define ranges for each parameter
    ranges = {
        "pit": (1, 10),
        "form": (0, 100),
        "risk": (0, 100),
        "aggro": (0, 100),
        "pressure": (0, 100),
    }

    current_params = {
        "pit": pit,
        "form": form,
        "risk": risk,
        "aggro": aggro,
        "pressure": pressure,
    }

    for param_name, (min_val, max_val) in ranges.items():
        param_values = [min_val, min_val + (max_val - min_val) * 0.25,
                        min_val + (max_val - min_val) * 0.5,
                        min_val + (max_val - min_val) * 0.75, max_val]

        predictions = []

        for param_val in param_values:
            temp_params = {
                "pit": param_val if param_name == "pit" else pit,
                "form": param_val if param_name == "form" else form,
                "risk": param_val if param_name == "risk" else risk,
                "aggro": param_val if param_name == "aggro" else aggro,
                "pressure": param_val if param_name == "pressure" else pressure,
            }

            # Compute for grid position 1-20
            probs = []
            for gp in range(1, 21):
                d = base_input.copy()
                d["grid"] = gp
                sim_df = pd.DataFrame([d]).reindex(columns=feature_cols, fill_value=0)
                base_prob = model.predict_proba(sim_df)[0, 1] * 100 if model else 50
                delta, _ = strategy_adjustment(gp, weather, tyre, temp_params["pit"],
                                               temp_params["form"], temp_params["risk"],
                                               temp_params["aggro"], temp_params["pressure"])
                final_prob = float(np.clip(base_prob + delta, 0, 100))
                probs.append(final_prob)

            predictions.append(np.mean(probs))

        results_dict[param_name] = {
            "values": param_values,
            "predictions": predictions,
        }

    return results_dict

# =========================================
# SEASON PROJECTION
# =========================================

@st.cache_data
def season_projection(driver_standings, races, modern_results):
    """Project final championship standings based on current data."""
    if driver_standings.empty or races.empty:
        return None

    # Get latest round data
    latest_round = driver_standings["raceId"].max()
    current_standings = driver_standings[driver_standings["raceId"] == latest_round].copy()

    if current_standings.empty:
        return None

    current_standings = current_standings.merge(
        races[["raceId", "year"]], on="raceId", how="left"
    )

    # Get season info
    current_year = current_standings["year"].iloc[0]
    total_races_planned = len(races[races["year"] == current_year])
    races_completed = len(races[(races["year"] == current_year) & (races["raceId"] <= latest_round)])

    if races_completed == 0:
        return None

    # Calculate projection
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
# PDF REPORT GENERATION
# =========================================

def generate_pdf_report(report_data):
    """Generate PDF report from race prediction data."""
    from io import BytesIO

    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
    except ImportError:
        return None

    # Create PDF buffer
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom style
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
    elements.append(Paragraph("🏎️ F1 Race Simulation Report", title_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Prediction metrics
    metric_style = styles["Normal"]
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

    # Build PDF
    doc.build(elements)
    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()
