import joblib
import numpy as np
import pandas as pd
import streamlit as st
from utils import (
    encode_weather, encode_tyre, normalize_probability,
    official_points_for_position, expected_position, projected_points,
    strategy_adjustment, monte_carlo, load_all_datasets,
    driver_matchup_analysis, circuit_statistics, driver_trend_analysis,
    constructor_analysis, parameter_sensitivity_analysis,
    season_projection, generate_pdf_report
)

st.set_page_config(page_title="F1 Race Intelligence Engine", page_icon="🏎️", layout="wide")

st.title("🏎️ Formula 1 Race Intelligence Engine")
st.markdown("Machine Learning powered race outcome simulator (Modern F1 Era ≥ 2010)")

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #1e1e1e;
    border-left: 5px solid #E10600;
    padding: 15px;
    border-radius: 8px;
}
</style>
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
# LOAD ALL DATASETS (OPTIMIZED)
# ===================================================

(drivers_df, constructors_df, circuits_df, races_df,
 results_df, driver_standings_df, constructor_standings_df,
 qualifying_df) = load_all_datasets()

# ===================================================
# SIDEBAR - RACE PARAMETERS
# ===================================================

with st.sidebar:
    st.header("Race Parameters")

    driver_list = (drivers_df["forename"] + " " + drivers_df["surname"]).sort_values()
    constructor_list = constructors_df["name"].sort_values()
    circuit_list = circuits_df["name"].sort_values()

    driver = st.selectbox("Driver", driver_list)
    constructor = st.selectbox("Constructor", constructor_list)
    circuit = st.selectbox("Circuit", circuit_list)

    grid = st.slider("Starting Grid", 1, 20, 1)

    st.subheader("Strategy")

    weather = st.selectbox("Weather", ["Dry", "Mixed", "Wet"])
    tyre = st.selectbox("Tyre Strategy", ["Conservative", "Balanced", "Aggressive"])

    pit = st.slider("Pit Crew", 1, 10, 6)
    form = st.slider("Recent Form", 0, 100, 70)
    risk = st.slider("Reliability Risk", 0, 100, 15)
    aggro = st.slider("Aggression", 0, 100, 60)
    pressure = st.slider("Teammate Pressure", 0, 100, 35)

    run_prediction = st.button("Run Simulation", use_container_width=True)

# ===================================================
# TABS
# ===================================================

tabs = st.tabs([
    "Race Simulation", "Model Brain", "Circuits",
    "Driver Matchup", "Circuit Records", "Driver Insights",
    "Constructor Insights", "Strategy Analysis", "Season Outlook"
])

# ===================================================
# TAB 0: RACE SIMULATION (ORIGINAL)
# ===================================================

with tabs[0]:
    if run_prediction:
        try:
            driver_id = int(drivers_df[(drivers_df["forename"] + " " + drivers_df["surname"]) == driver]["driverId"].values[0])
            constructor_id = int(constructors_df[constructors_df["name"] == constructor]["constructorId"].values[0])
            circuit_id = int(circuits_df[circuits_df["name"] == circuit]["circuitId"].values[0])
        except (IndexError, ValueError):
            st.error("⚠️ Could not find matching IDs for the selected driver/constructor/circuit combination.")
            st.stop()

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

        # Vectorized grid simulation
        sim_inputs = []
        for gp in range(1, 21):
            d = base_input.copy()
            d["grid"] = gp
            sim_inputs.append(d)

        sim_df = pd.DataFrame(sim_inputs).reindex(columns=feature_cols, fill_value=0)
        base_probs = model.predict_proba(sim_df)[:, 1] * 100 if model else np.full(20, 50)

        final_probs = []
        strategy_deltas = []

        for i, gp in enumerate(range(1, 21)):
            delta, _ = strategy_adjustment(gp, weather, tyre, pit, form, risk, aggro, pressure)
            strategy_deltas.append(delta)
            final_probs.append(float(np.clip(base_probs[i] + delta, 0, 100)))

        # User grid result
        user_base = base_probs[grid - 1]
        user_delta = strategy_deltas[grid - 1]
        user_prob = final_probs[grid - 1]

        proj_pts = projected_points(user_prob)
        pos = expected_position(user_prob)
        mc = monte_carlo(user_prob)

        constructor_pts = proj_pts * 2
        season_projection_pts = proj_pts * 23

        # Metrics display
        col1, col2 = st.columns(2)

        with col1:
            st.metric("ML Predicted Podium Probability", f"{user_base:.1f}%")
            st.metric("Strategy Impact", f"{user_delta:+.1f}%")
            st.metric("Final Adjusted Podium Probability", f"{user_prob:.1f}%")
            st.metric("Expected Finish Position", f"P{pos}")
            st.metric("Monte Carlo Podium Chance", f"{mc}%")

        with col2:
            st.metric("Projected Championship Points", f"{proj_pts} pts")
            st.metric("Constructor Expected Points", f"{constructor_pts:.1f}")
            st.metric("Season Projection", f"{season_projection_pts:.1f}")

        # Strategy breakdown
        st.subheader("Strategy Impact Breakdown")
        _, effects = strategy_adjustment(grid, weather, tyre, pit, form, risk, aggro, pressure)
        effects_df = pd.DataFrame.from_dict(effects, orient="index", columns=["Impact"])
        st.bar_chart(effects_df)

        # Grid sensitivity
        st.subheader("Grid Position Sensitivity")
        chart_df = pd.DataFrame({
            "ML Prediction": base_probs,
            "Strategy Adjusted": final_probs,
            "Projected Points": [projected_points(p) for p in final_probs]
        }, index=range(1, 21))
        chart_df.index.name = "Grid Position"
        st.line_chart(chart_df)

        # Download report
        report = pd.DataFrame({
            "Driver": [driver],
            "Grid": [grid],
            "ML Prediction": [user_base],
            "Strategy Impact": [user_delta],
            "Final Prediction": [user_prob],
            "Projected Points": [proj_pts]
        })

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download Race Report (CSV)",
                report.to_csv(index=False),
                "race_report.csv"
            )

        with col2:
            pdf_data = generate_pdf_report({
                "Driver": driver,
                "Grid": grid,
                "ML Prediction": user_base,
                "Strategy Impact": user_delta,
                "Final Prediction": user_prob,
                "Projected Points": proj_pts
            })
            if pdf_data:
                st.download_button(
                    "📄 Download Race Report (PDF)",
                    pdf_data,
                    "race_report.pdf",
                    "application/pdf"
                )

        st.success("✅ Simulation complete!")

    else:
        st.info("Select parameters in the sidebar and click Run Simulation")

# ===================================================
# TAB 1: MODEL BRAIN (OPTIMIZED - LAZY LOAD)
# ===================================================

with tabs[1]:
    st.header("Model Intelligence & Comparison")

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
# TAB 3: DRIVER MATCHUP
# ===================================================

with tabs[3]:
    st.header("🏁 Driver Head-to-Head Comparison")

    col1, col2 = st.columns(2)
    with col1:
        driver1 = st.selectbox("Driver 1", driver_list, key="d1")
    with col2:
        driver2 = st.selectbox("Driver 2", driver_list, key="d2")

    if driver1 != driver2:
        driver1_id = int(drivers_df[(drivers_df["forename"] + " " + drivers_df["surname"]) == driver1]["driverId"].values[0])
        driver2_id = int(drivers_df[(drivers_df["forename"] + " " + drivers_df["surname"]) == driver2]["driverId"].values[0])

        matchup = driver_matchup_analysis(driver1_id, driver2_id, results_df, races_df)

        if matchup:
            st.subheader(f"{driver1} vs {driver2}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Races Together", matchup["races_together"])
            with col2:
                st.metric(f"{driver1} Podiums", matchup["d1_podiums"])
            with col3:
                st.metric(f"{driver2} Podiums", matchup["d2_podiums"])
            with col4:
                st.metric("Head Record", f"{matchup['d1_wins']}-{matchup['d2_wins']}")

            comparison_data = pd.DataFrame({
                "Driver": [driver1, driver2],
                "Podium Rate (%)": [matchup["d1_podium_rate"], matchup["d2_podium_rate"]],
                "Avg Finish Position": [matchup["d1_avg_finish"], matchup["d2_avg_finish"]],
                "Total Points": [matchup["d1_points"], matchup["d2_points"]],
            })

            st.dataframe(comparison_data, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(comparison_data.set_index("Driver")[["Podium Rate (%)"]])
            with col2:
                st.bar_chart(comparison_data.set_index("Driver")[["Total Points"]])
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

    circuit_id = int(circuits_df[circuits_df["name"] == selected_circuit]["circuitId"].values[0])

    circuit_stats = circuit_statistics(circuit_id, results_df, races_df, drivers_df, constructors_df)

    if circuit_stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Races Held", circuit_stats["races_held"])
        with col2:
            st.metric("Avg Winner Grid", f"{circuit_stats['avg_winner_grid']:.1f}")
        with col3:
            st.metric("Top Winner", circuit_stats["top_winners"].index[0] if len(circuit_stats["top_winners"]) > 0 else "N/A")
        with col4:
            st.metric("Top Team", circuit_stats["top_constructors"].index[0] if len(circuit_stats["top_constructors"]) > 0 else "N/A")

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
# TAB 5: DRIVER INSIGHTS
# ===================================================

with tabs[5]:
    st.header("📈 Driver Performance Trends")

    selected_driver = st.selectbox("Select Driver", driver_list, key="driver_trends")

    driver_id = int(drivers_df[(drivers_df["forename"] + " " + drivers_df["surname"]) == selected_driver]["driverId"].values[0])

    trend_data = driver_trend_analysis(driver_id, results_df, races_df, drivers_df)

    if trend_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Races", trend_data["total_races"])
        with col2:
            recent_wins = trend_data["recent_season"][trend_data["recent_season"]["positionOrder_int"] == 1].shape[0]
            st.metric("Recent Wins (23 races)", recent_wins)
        with col3:
            recent_podiums = trend_data["recent_season"][trend_data["recent_season"]["is_podium"]].shape[0]
            st.metric("Recent Podiums (23 races)", recent_podiums)

        st.subheader("Performance Trend (Last 2 Seasons)")
        trend_chart_data = trend_data["recent_season"][["rolling_avg_finish", "rolling_podium_rate", "points_per_race"]].copy()
        trend_chart_data.columns = ["Avg Finish Pos", "Podium Rate (%)", "Points/Race"]
        st.line_chart(trend_chart_data)

        st.subheader("Circuit Affinity")
        col1, col2 = st.columns(2)

        with col1:
            st.write("🏁 Best Circuits")
            st.dataframe(
                trend_data["best_circuits"][["name", "avg_position", "podium_rate"]].rename(
                    columns={"name": "Circuit", "avg_position": "Avg Pos", "podium_rate": "Podium %"}
                ),
                use_container_width=True
            )

        with col2:
            st.write("🚫 Challenging Circuits")
            st.dataframe(
                trend_data["worst_circuits"][["name", "avg_position", "podium_rate"]].rename(
                    columns={"name": "Circuit", "avg_position": "Avg Pos", "podium_rate": "Podium %"}
                ),
                use_container_width=True
            )
    else:
        st.warning("No data available for this driver.")

# ===================================================
# TAB 6: CONSTRUCTOR INSIGHTS
# ===================================================

with tabs[6]:
    st.header("🏭 Constructor Performance Analysis")

    selected_constructor = st.selectbox("Select Constructor", constructor_list, key="constructor_analysis")

    constructor_id = int(constructors_df[constructors_df["name"] == selected_constructor]["constructorId"].values[0])

    const_data = constructor_analysis(constructor_id, results_df, races_df, driver_standings_df, constructor_standings_df)

    if const_data:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Races", const_data["total_races"])
        with col2:
            st.metric("Wins", const_data["wins"])
        with col3:
            st.metric("Podiums", const_data["podiums"])
        with col4:
            st.metric("DNF Rate (%)", f"{const_data['dnf_rate']:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Win Rate (%)", f"{const_data['win_rate']:.1f}%")
        with col2:
            st.metric("Podium Rate (%)", f"{const_data['podium_rate']:.1f}%")

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
        st.info("Analyzing sensitivity of your current prediction to strategy parameters...")

        try:
            driver_id = int(drivers_df[(drivers_df["forename"] + " " + drivers_df["surname"]) == driver]["driverId"].values[0])
            constructor_id = int(constructors_df[constructors_df["name"] == constructor]["constructorId"].values[0])
            circuit_id = int(circuits_df[circuits_df["name"] == circuit]["circuitId"].values[0])
        except:
            st.warning("Run a simulation first to analyze strategy sensitivity.")
        else:
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
        st.info("Run a simulation first, then this tab will show how changing each strategy parameter affects your prediction.")

# ===================================================
# TAB 8: SEASON OUTLOOK
# ===================================================

with tabs[8]:
    st.header("🏆 Championship Projection")

    projection = season_projection(driver_standings_df, races_df, results_df)

    if projection:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Season Year", int(projection["current_year"]))
        with col2:
            st.metric(f"Races Completed", projection["races_completed"])
        with col3:
            st.metric(f"Races Remaining", projection["total_races_planned"] - projection["races_completed"])

        st.subheader("Projected Final Standings")

        display_cols = ["position", "points", "avg_points_per_race", "projected_final_points", "projected_position"]
        display_df = projection["standings"][display_cols].copy()
        display_df.columns = ["Current Pos", "Current Points", "Avg/Race", "Projected Final", "Projected Rank"]

        st.dataframe(display_df.head(10), use_container_width=True)

        st.subheader("Points Projection")
        projection_chart = projection["standings"][["points", "projected_final_points"]].head(10).copy()
        projection_chart.columns = ["Current Points", "Projected Final"]
        st.line_chart(projection_chart)

    else:
        st.warning("Season projection data not available. Check if the current season data has been updated.")
