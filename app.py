"""
F1 Race Intelligence Engine — Main Application
================================================
Streamlit dashboard for ML-powered race outcome prediction and F1 analytics.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from utils import (
    encode_weather, encode_tyre, normalize_probability,
    official_points_for_position, expected_position, projected_points,
    strategy_adjustment, strategy_adjustment_vectorized,
    monte_carlo, monte_carlo_batch, load_all_datasets,
    driver_matchup_analysis, circuit_statistics, driver_trend_analysis,
    constructor_analysis, parameter_sensitivity_analysis,
    generate_pdf_report,
    get_driver_id, get_constructor_id, get_circuit_id,
    predict_dnf_probability, get_actual_dnf_rate,
    qualifying_impact_analysis, streak_analysis,
    constructor_dominance,
    MODERN_ERA_START, RACES_PER_SEASON, FINISHED_STATUS_IDS,
)

# ===================================================
# PAGE CONFIG
# ===================================================

st.set_page_config(
    page_title="F1 Race Intelligence Engine",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================================================
# F1-THEMED CSS
# ===================================================

st.markdown("""
<style>
/* ---- Global Theme ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Header Banner ---- */
.f1-header {
    background: linear-gradient(135deg, #E10600 0%, #1E1E1E 70%);
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(225, 6, 0, 0.25);
}
.f1-header h1 {
    color: white;
    margin: 0;
    font-size: 2.2em;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.f1-header p {
    color: rgba(255,255,255,0.75);
    margin: 4px 0 0 0;
    font-size: 1em;
}

/* ---- Metric Cards ---- */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border-left: 4px solid #E10600;
    padding: 16px 20px;
    border-radius: 10px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(225, 6, 0, 0.2);
}
div[data-testid="stMetric"] label {
    color: #a0a0b0 !important;
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700;
}

/* ---- Tab Styling ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0e1117;
    padding: 4px;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
    font-size: 0.85em;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #E10600, #ff2d2d) !important;
    color: white !important;
}

/* ---- Section Headers ---- */
.section-header {
    background: linear-gradient(90deg, #E10600, transparent);
    padding: 2px 0;
    border-radius: 4px;
    margin: 24px 0 12px 0;
}
.section-header h3 {
    color: white;
    padding: 8px 16px;
    margin: 0;
    font-size: 1.1em;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1117, #1a1a2e);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #E10600;
    font-size: 1.2em;
    border-bottom: 2px solid #E10600;
    padding-bottom: 8px;
}

/* ---- Success/Info/Warning ---- */
.stAlert {
    border-radius: 8px;
}

/* ---- Download Buttons ---- */
.stDownloadButton button {
    background: linear-gradient(135deg, #E10600, #ff2d2d) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(225, 6, 0, 0.4) !important;
}

/* ---- Dataframe ---- */
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ===================================================
# HEADER
# ===================================================

st.markdown("""
<div class="f1-header">
    <h1>🏎️ Formula 1 Race Intelligence Engine</h1>
    <p>ML-powered race outcome simulation & analytics • Modern Era (≥ 2010)</p>
</div>
""", unsafe_allow_html=True)

# ===================================================
# LOAD MODEL
# ===================================================

@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load("f1_model.pkl")
        feature_cols = joblib.load("f1_model_features.pkl")
        return model, feature_cols
    except FileNotFoundError:
        st.warning("⚠️ Model files not found. Please run the training notebook first.")
        return None, []
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, []


@st.cache_data
def load_model_comparison():
    try:
        return pd.read_csv("model_comparison.csv")
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading comparison data: {e}")
        return pd.DataFrame()


comparison_df = load_model_comparison()
model, feature_cols = load_model_and_features()

# ===================================================
# LOAD ALL DATASETS
# ===================================================

(drivers_df, constructors_df, circuits_df, races_df,
 results_df, driver_standings_df, constructor_standings_df,
 qualifying_df, status_df, pit_stops_df, lap_times_df,
 sprint_results_df) = load_all_datasets()

# ===================================================
# HELPERS
# ===================================================

def build_base_input(driver_id, constructor_id, circuit_id,
                     weather, tyre, pit, form, risk, aggro, pressure):
    """Build the feature input dict for ML model (DRY — used in Tab 0 and Tab 7)."""
    return {
        "circuitId": circuit_id,
        "constructorId": constructor_id,
        "driverId": driver_id,
        "weather_code": encode_weather(weather),
        "tyre_strategy_code": encode_tyre(tyre),
        "pit_crew_rating": pit,
        "recent_form": form,
        "reliability_risk": risk,
        "aggression_level": aggro,
        "teammate_pressure": pressure,
    }


def resolve_ids(driver, constructor, circuit):
    """Resolve driver/constructor/circuit names to IDs with error handling."""
    driver_id = get_driver_id(driver, drivers_df)
    constructor_id = get_constructor_id(constructor, constructors_df)
    circuit_id = get_circuit_id(circuit, circuits_df)

    if None in [driver_id, constructor_id, circuit_id]:
        st.error("⚠️ Could not find matching IDs for the selected driver/constructor/circuit combination.")
        st.stop()

    return driver_id, constructor_id, circuit_id

# ===================================================
# SIDEBAR — RACE PARAMETERS
# ===================================================

with st.sidebar:
    st.markdown("## 🏁 Race Parameters")

    driver_list = (drivers_df["forename"] + " " + drivers_df["surname"]).sort_values()
    constructor_list = constructors_df["name"].sort_values()
    circuit_list = circuits_df["name"].sort_values()

    driver = st.selectbox("Driver", driver_list)
    constructor = st.selectbox("Constructor", constructor_list)
    circuit = st.selectbox("Circuit", circuit_list)

    grid = st.slider("Starting Grid", 1, 20, 1)

    st.markdown("## ⚙️ Strategy")

    weather = st.selectbox("Weather", ["Dry", "Mixed", "Wet"])
    tyre = st.selectbox("Tyre Strategy", ["Conservative", "Balanced", "Aggressive"])

    pit = st.slider("Pit Crew Rating", 1, 10, 6)
    form = st.slider("Recent Form", 0, 100, 70)
    risk = st.slider("Reliability Risk", 0, 100, 15)
    aggro = st.slider("Aggression", 0, 100, 60)
    pressure = st.slider("Teammate Pressure", 0, 100, 35)

    st.divider()
    run_prediction = st.button("🏎️ Run Simulation", use_container_width=True, type="primary")

# ===================================================
# TABS
# ===================================================

tabs = st.tabs([
    "🏁 Race Sim",
    "🧠 Model Brain",
    "🗺️ Circuits",
    "👥 Driver Matchup",
    "🏆 Circuit Records",
    "📈 Driver Trends",
    "🏭 Constructor",
    "🎯 Strategy Lab",
    "🔄 Qualifying",
    "🔥 Streaks",
    "👑 Dominance",
])

# ===================================================
# TAB 0: RACE SIMULATION
# ===================================================

with tabs[0]:
    if run_prediction:
        with st.spinner("Running simulation..."):
            driver_id, constructor_id, circuit_id = resolve_ids(driver, constructor, circuit)

            base_input = build_base_input(
                driver_id, constructor_id, circuit_id,
                weather, tyre, pit, form, risk, aggro, pressure
            )

            # Vectorized grid simulation
            grids = np.arange(1, 21)
            sim_inputs = []
            for gp in grids:
                d = base_input.copy()
                d["grid"] = gp
                sim_inputs.append(d)

            sim_df = pd.DataFrame(sim_inputs).reindex(columns=feature_cols, fill_value=0)
            base_probs = model.predict_proba(sim_df)[:, 1] * 100 if model else np.full(20, 50)

            # Vectorized strategy adjustment
            deltas, _ = strategy_adjustment_vectorized(
                grids, weather, tyre, pit, form, risk, aggro, pressure
            )
            final_probs = np.clip(base_probs + deltas, 0, 100)

            # User grid result
            user_base = float(base_probs[grid - 1])
            user_delta = float(deltas[grid - 1])
            user_prob = float(final_probs[grid - 1])

            proj_pts = projected_points(user_prob)
            pos = expected_position(user_prob)
            mc = monte_carlo(user_prob)

            # Accurate DNF probability using status data
            actual_dnf_rate = get_actual_dnf_rate(constructor_id, results_df, status_df)
            dnf_prob = predict_dnf_probability(risk, actual_dnf_rate)

            constructor_pts = proj_pts * 2
            season_projection_pts = proj_pts * RACES_PER_SEASON

        # Metrics display
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("ML Podium Probability", f"{user_base:.1f}%")
            st.metric("Strategy Impact", f"{user_delta:+.1f}%")
            st.metric("Monte Carlo Chance", f"{mc}%")

        with col2:
            st.metric("Final Adjusted Probability", f"{user_prob:.1f}%")
            st.metric("Expected Finish", f"P{pos}")
            st.metric("DNF Probability", f"{dnf_prob}%")

        with col3:
            st.metric("Projected Points", f"{proj_pts} pts")
            st.metric("Constructor Points (est.)", f"{constructor_pts:.0f}")
            st.metric("Season Projection", f"{season_projection_pts:.0f} pts")

        # Strategy breakdown
        st.markdown('<div class="section-header"><h3>📊 Strategy Impact Breakdown</h3></div>',
                    unsafe_allow_html=True)
        _, effects = strategy_adjustment(grid, weather, tyre, pit, form, risk, aggro, pressure)
        effects_df = pd.DataFrame.from_dict(effects, orient="index", columns=["Impact (%)"])
        st.bar_chart(effects_df)

        # Grid sensitivity
        st.markdown('<div class="section-header"><h3>📈 Grid Position Sensitivity</h3></div>',
                    unsafe_allow_html=True)
        chart_df = pd.DataFrame({
            "ML Prediction": base_probs,
            "Strategy Adjusted": final_probs,
            "Projected Points": [projected_points(p) for p in final_probs]
        }, index=range(1, 21))
        chart_df.index.name = "Grid Position"
        st.line_chart(chart_df)

        # Download reports
        report = pd.DataFrame({
            "Driver": [driver], "Grid": [grid],
            "ML Prediction": [user_base], "Strategy Impact": [user_delta],
            "Final Prediction": [user_prob], "Projected Points": [proj_pts],
            "DNF Probability": [dnf_prob]
        })

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download Race Report (CSV)",
                report.to_csv(index=False), "race_report.csv"
            )
        with col2:
            pdf_data = generate_pdf_report({
                "Driver": driver, "Grid": grid,
                "ML Prediction": user_base, "Strategy Impact": user_delta,
                "Final Prediction": user_prob, "Projected Points": proj_pts,
                "DNF Probability": dnf_prob,
            })
            if pdf_data:
                st.download_button(
                    "📄 Download Race Report (PDF)",
                    pdf_data, "race_report.pdf", "application/pdf"
                )

        st.success("✅ Simulation complete!")
    else:
        st.info("👈 Select parameters in the sidebar and click **Run Simulation** to begin.")

# ===================================================
# TAB 1: MODEL BRAIN
# ===================================================

with tabs[1]:
    st.header("🧠 Model Intelligence & Comparison")

    if model is not None and feature_cols:
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)
        st.bar_chart(importance_df.set_index("Feature"))
    else:
        st.warning("Model not loaded. Cannot display feature importance.")

    st.divider()

    if not comparison_df.empty:
        st.subheader("Model Performance Comparison")
        st.dataframe(comparison_df, use_container_width=True)

        st.subheader("ROC-AUC Comparison")
        roc_chart = comparison_df.set_index("Model")[["ROC AUC"]]
        st.bar_chart(roc_chart)

        st.info("""
        Multiple models were evaluated including Logistic Regression, Random Forest, and Gradient Boosting.
        Random Forest was selected for highest ROC-AUC and strong generalization.
        Tree-based models effectively capture nonlinear relationships in F1 race data.
        """)
    else:
        st.warning("Model comparison results not found.")

# ===================================================
# TAB 2: CIRCUITS MAP
# ===================================================

with tabs[2]:
    st.header("🗺️ F1 Circuit Map")
    st.markdown("Global locations of all Formula 1 circuits (2010+)")

    if {"lat", "lng"}.issubset(circuits_df.columns):
        st.map(circuits_df, latitude="lat", longitude="lng")
    else:
        st.warning("Circuit location data not available.")

# ===================================================
# TAB 3: DRIVER MATCHUP (ENHANCED)
# ===================================================

with tabs[3]:
    st.header("👥 Driver Head-to-Head Comparison")

    col1, col2 = st.columns(2)
    with col1:
        driver1 = st.selectbox("Driver 1", driver_list, key="d1")
    with col2:
        driver2 = st.selectbox("Driver 2", driver_list, key="d2")

    if driver1 != driver2:
        driver1_id = get_driver_id(driver1, drivers_df)
        driver2_id = get_driver_id(driver2, drivers_df)

        if driver1_id is None or driver2_id is None:
            st.error("Could not find drivers.")
        else:
            matchup = driver_matchup_analysis(driver1_id, driver2_id, results_df, races_df)

            if matchup:
                st.subheader(f"{driver1} vs {driver2}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Races Together", matchup["races_together"])
                with col2:
                    st.metric(f"{driver1.split()[-1]} Podiums", matchup["d1_podiums"])
                with col3:
                    st.metric(f"{driver2.split()[-1]} Podiums", matchup["d2_podiums"])
                with col4:
                    st.metric("Head Record", f"{matchup['d1_wins']}-{matchup['d2_wins']}")

                comparison_data = pd.DataFrame({
                    "Driver": [driver1, driver2],
                    "Podium Rate (%)": [matchup["d1_podium_rate"], matchup["d2_podium_rate"]],
                    "Avg Finish": [matchup["d1_avg_finish"], matchup["d2_avg_finish"]],
                    "Total Points": [matchup["d1_points"], matchup["d2_points"]],
                })
                st.dataframe(comparison_data, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(comparison_data.set_index("Driver")[["Podium Rate (%)"]])
                with col2:
                    st.bar_chart(comparison_data.set_index("Driver")[["Total Points"]])

                # NEW: Race-by-race timeline
                timeline = matchup.get("timeline")
                if timeline is not None and not timeline.empty:
                    st.markdown('<div class="section-header"><h3>📈 Race-by-Race Timeline</h3></div>',
                                unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Cumulative Points")
                        pts_chart = timeline[["d1_cum_points", "d2_cum_points"]].copy()
                        pts_chart.columns = [driver1.split()[-1], driver2.split()[-1]]
                        pts_chart.index = range(1, len(pts_chart) + 1)
                        pts_chart.index.name = "Race #"
                        st.line_chart(pts_chart)

                    with col2:
                        st.subheader("Finish Position per Race")
                        pos_chart = timeline[["d1_pos", "d2_pos"]].copy()
                        pos_chart.columns = [driver1.split()[-1], driver2.split()[-1]]
                        pos_chart.index = range(1, len(pos_chart) + 1)
                        pos_chart.index.name = "Race #"
                        st.line_chart(pos_chart)

                    # Head-to-head win rate
                    d1_beat = timeline["d1_beat_d2"].sum()
                    total = len(timeline)
                    st.metric(
                        f"{driver1.split()[-1]} Beats {driver2.split()[-1]}",
                        f"{d1_beat}/{total} ({d1_beat/total*100:.0f}%)"
                    )
            else:
                st.warning("These drivers never competed in the same races.")
    else:
        st.info("Select two different drivers to compare.")

# ===================================================
# TAB 4: CIRCUIT RECORDS
# ===================================================

with tabs[4]:
    st.header("🏆 Circuit Records & Statistics")

    selected_circuit = st.selectbox("Select Circuit", circuit_list, key="circuit_records")
    circuit_id_rec = get_circuit_id(selected_circuit, circuits_df)

    if circuit_id_rec:
        circuit_stats = circuit_statistics(circuit_id_rec, results_df, races_df, drivers_df, constructors_df)

        if circuit_stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Races Held", circuit_stats["races_held"])
            with col2:
                st.metric("Avg Winner Grid", f"{circuit_stats['avg_winner_grid']:.1f}")
            with col3:
                st.metric("Top Winner",
                          circuit_stats["top_winners"].index[0] if len(circuit_stats["top_winners"]) > 0 else "N/A")
            with col4:
                st.metric("Top Team",
                          circuit_stats["top_constructors"].index[0] if len(circuit_stats["top_constructors"]) > 0 else "N/A")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Most Wins (Drivers)")
                st.dataframe(circuit_stats["top_winners"].to_frame("Wins"), use_container_width=True)
            with col2:
                st.subheader("Most Constructor Wins")
                st.dataframe(circuit_stats["top_constructors"].to_frame("Wins"), use_container_width=True)

            st.subheader("Most Podium Finishes (Drivers)")
            st.dataframe(circuit_stats["top_podium_drivers"].to_frame("Podiums"), use_container_width=True)
        else:
            st.warning("No data available for this circuit.")

# ===================================================
# TAB 5: DRIVER TRENDS
# ===================================================

with tabs[5]:
    st.header("📈 Driver Performance Trends")

    selected_driver = st.selectbox("Select Driver", driver_list, key="driver_trends")
    driver_id_trend = get_driver_id(selected_driver, drivers_df)

    if driver_id_trend:
        trend_data = driver_trend_analysis(driver_id_trend, results_df, races_df, drivers_df)

        if trend_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Races", trend_data["total_races"])
            with col2:
                recent_wins = trend_data["recent_season"][
                    trend_data["recent_season"]["positionOrder_int"] == 1
                ].shape[0]
                st.metric("Recent Wins (23 races)", recent_wins)
            with col3:
                recent_podiums = trend_data["recent_season"][
                    trend_data["recent_season"]["is_podium"]
                ].shape[0]
                st.metric("Recent Podiums (23 races)", recent_podiums)

            st.subheader("Performance Trend (Last 2 Seasons)")
            trend_chart_data = trend_data["recent_season"][
                ["rolling_avg_finish", "rolling_podium_rate", "points_per_race"]
            ].copy()
            trend_chart_data.columns = ["Avg Finish Pos", "Podium Rate (%)", "Points/Race"]
            st.line_chart(trend_chart_data)

            st.subheader("Circuit Affinity")
            col1, col2 = st.columns(2)

            with col1:
                st.write("🏁 Best Circuits")
                st.dataframe(
                    trend_data["best_circuits"][["name", "avg_position", "podium_rate"]].rename(
                        columns={"name": "Circuit", "avg_position": "Avg Pos", "podium_rate": "Podium %"}
                    ), use_container_width=True
                )
            with col2:
                st.write("🚫 Challenging Circuits")
                st.dataframe(
                    trend_data["worst_circuits"][["name", "avg_position", "podium_rate"]].rename(
                        columns={"name": "Circuit", "avg_position": "Avg Pos", "podium_rate": "Podium %"}
                    ), use_container_width=True
                )
        else:
            st.warning("No data available for this driver.")

# ===================================================
# TAB 6: CONSTRUCTOR INSIGHTS
# ===================================================

with tabs[6]:
    st.header("🏭 Constructor Performance Analysis")

    selected_constructor = st.selectbox("Select Constructor", constructor_list, key="constructor_analysis")
    constructor_id_analysis = get_constructor_id(selected_constructor, constructors_df)

    if constructor_id_analysis:
        const_data = constructor_analysis(
            constructor_id_analysis, results_df, races_df,
            driver_standings_df, constructor_standings_df
        )

        if const_data:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Races", const_data["total_races"])
            with col2:
                st.metric("Wins", const_data["wins"])
            with col3:
                st.metric("Podiums", const_data["podiums"])
            with col4:
                st.metric("DNF Rate", f"{const_data['dnf_rate']:.1f}%")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Win Rate", f"{const_data['win_rate']:.1f}%")
            with col2:
                st.metric("Podium Rate", f"{const_data['podium_rate']:.1f}%")

            st.subheader("Performance Trend (Recent)")
            trend_chart = const_data["recent_results"][["rolling_avg_points"]].copy()
            trend_chart.columns = ["Avg Points/Race"]
            st.line_chart(trend_chart)
        else:
            st.warning("No data available for this constructor.")

# ===================================================
# TAB 7: STRATEGY WHAT-IF ANALYSIS
# ===================================================

with tabs[7]:
    st.header("🎯 Strategy Sensitivity Analysis")

    if run_prediction:
        with st.spinner("Analyzing parameter sensitivity..."):
            driver_id, constructor_id, circuit_id = resolve_ids(driver, constructor, circuit)

            base_input = build_base_input(
                driver_id, constructor_id, circuit_id,
                weather, tyre, pit, form, risk, aggro, pressure
            )

            sensitivity = parameter_sensitivity_analysis(
                model, feature_cols, base_input, drivers_df, constructors_df, circuits_df,
                weather, tyre, pit, form, risk, aggro, pressure
            )

            param_names = {
                "pit": "Pit Crew (Rating)",
                "form": "Recent Form (%)",
                "risk": "Reliability Risk (%)",
                "aggro": "Aggression (%)",
                "pressure": "Teammate Pressure (%)",
            }

            for param, param_display in param_names.items():
                st.subheader(f"Impact of {param_display}")

                sensitivity_df = pd.DataFrame({
                    param_display: sensitivity[param]["values"],
                    "Avg Podium Prob": sensitivity[param]["predictions"]
                })
                st.line_chart(sensitivity_df.set_index(param_display))
    else:
        st.info("Run a simulation first, then this tab will show sensitivity analysis.")



# ===================================================
# TAB 8: QUALIFYING IMPACT
# ===================================================

with tabs[8]:
    st.header("🔄 Qualifying Position Impact")
    st.markdown("How qualifying performance affects race outcomes")

    if not qualifying_df.empty:
        try:
            with st.spinner("Analyzing qualifying data..."):
                qual_impact, circuit_qual = qualifying_impact_analysis(
                    drivers_df, constructors_df, circuits_df, results_df, qualifying_df, races_df
                )

            st.subheader("Performance by Qualifying Position")
            st.line_chart(qual_impact[["avg_finish_pos"]])

            st.subheader("Qualifying Advantage by Circuit")
            circuit_display = circuit_qual[["name", "avg_qual_to_finish_delta", "overtakes_per_race"]].head(15)
            circuit_display.columns = ["Circuit", "Avg Qual→Finish", "Overtakes/Race"]
            st.dataframe(circuit_display, use_container_width=True)

            st.info("""
            **Key Insights:**
            - Positive values = drivers typically finish ahead of qualifying position
            - Circuits vary in overtaking difficulty
            - Qualifying position is a strong predictor on street circuits
            """)
        except Exception as e:
            st.warning(f"Could not generate qualifying analysis: {e}")
    else:
        st.warning("Qualifying data not available.")

# ===================================================
# TAB 9: DRIVER STREAKS
# ===================================================

with tabs[9]:
    st.header("🔥 Driver Streaks & Form Analysis")
    st.markdown("Track consecutive achievements and current momentum")

    streak_driver = st.selectbox("Select Driver", driver_list, key="streak_driver")

    if streak_driver:
        try:
            streak_driver_id = get_driver_id(streak_driver, drivers_df)
            if streak_driver_id:
                streaks = streak_analysis(streak_driver_id, results_df, races_df)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Longest Podium Streak", streaks["longest_podium_streak"])
                with col2:
                    st.metric("Longest Win Streak", streaks["longest_win_streak"])
                with col3:
                    st.metric("Current Podium Streak", streaks["current_podium_streak"])
                with col4:
                    st.metric("Recent Podiums (Last 10)", streaks["recent_podiums_10"])

                if streaks["current_podium_streak"] >= 3:
                    st.success(f"🔥 {streak_driver} is on a hot streak!")
                elif streaks["current_podium_streak"] == 0:
                    st.warning(f"⚠️ {streak_driver} needs a turnaround — no active podium streak")
                else:
                    st.info(f"📊 {streak_driver} has a modest current streak")

                # Driver trend visualization
                trend_data = driver_trend_analysis(streak_driver_id, results_df, races_df, drivers_df)

                if trend_data:
                    st.subheader("Last 10 Races Performance")
                    recent = trend_data["recent_season"].tail(10)
                    if "positionOrder_int" in recent.columns:
                        performance_df = recent[["positionOrder_int", "is_podium"]].copy()
                        performance_df.columns = ["Finish Position", "Podium"]
                        st.line_chart(performance_df)
        except Exception as e:
            st.error(f"Error analyzing streaks: {e}")

# ===================================================
# TAB 10: CONSTRUCTOR DOMINANCE
# ===================================================

with tabs[10]:
    st.header("👑 Constructor Dominance Index")
    st.markdown("Measure team supremacy across seasons")

    dom_constructor = st.selectbox("Select Constructor", constructor_list, key="dom_constructor")
    dom_constructor_id = get_constructor_id(dom_constructor, constructors_df)

    if dom_constructor_id:
        with st.spinner("Calculating dominance metrics..."):
            dom_data = constructor_dominance(
                dom_constructor_id, results_df, races_df, constructors_df
            )

        if dom_data:
            # All-time metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Wins", dom_data["total_wins"])
            with col2:
                st.metric("Total Podiums", dom_data["total_podiums"])
            with col3:
                st.metric("1-2 Finishes", dom_data["total_one_two"])
            with col4:
                st.metric("Avg Dominance Score", dom_data["avg_dominance_score"])

            # Peak season
            peak = dom_data["peak_season"]
            if peak is not None:
                st.subheader(f"🏆 Peak Season: {int(peak['year'])}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Wins", int(peak["wins"]))
                with col2:
                    st.metric("Points Share", f"{peak['points_share']:.1f}%")
                with col3:
                    st.metric("Dominance Score", peak["dominance_score"])

            # Season-by-season chart
            seasons_df = dom_data["seasons"]
            if not seasons_df.empty:
                st.subheader("Dominance Score by Season")
                dom_chart = seasons_df.set_index("year")[["dominance_score"]].copy()
                dom_chart.columns = ["Dominance Score"]
                st.line_chart(dom_chart)

                st.subheader("Points Share by Season")
                share_chart = seasons_df.set_index("year")[["points_share"]].copy()
                share_chart.columns = ["Points Share (%)"]
                st.line_chart(share_chart)

                st.subheader("Season Breakdown")
                display = seasons_df[["year", "wins", "podiums", "one_two_finishes",
                                      "points_share", "dominance_score"]].copy()
                display.columns = ["Year", "Wins", "Podiums", "1-2 Finishes",
                                   "Points Share (%)", "Dominance"]
                st.dataframe(display, use_container_width=True)
        else:
            st.warning(f"No dominance data available for {dom_constructor}.")
