import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIG & STYLING
# --------------------------------------------------
st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏎️", layout="wide")

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #1e1e1e;
    border-left: 5px solid #E10600;
    padding: 15px;
    border-radius: 8px;
}
div.stButton > button:first-child {
    background-color: #E10600;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏎️ Formula 1 ML Strategy & Podium Predictor")
st.markdown("Machine Learning powered podium probability engine (Modern F1 Era ≥ 2010).")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load("f1_model.pkl")
        feature_cols = joblib.load("f1_model_features.pkl")
        return model, feature_cols
    except Exception:
        return None, []

# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------
@st.cache_data
def load_and_filter_datasets():
    try:
        circuits = pd.read_csv("datasets/circuits.csv")
        drivers = pd.read_csv("datasets/drivers.csv")
        constructors = pd.read_csv("datasets/constructors.csv")
        races = pd.read_csv("datasets/races.csv")
        results = pd.read_csv("datasets/results.csv")

        modern_races = races[races["year"] >= 2010]
        modern_results = results[results["raceId"].isin(modern_races["raceId"].unique())]

        drivers_modern = drivers[drivers["driverId"].isin(modern_results["driverId"].unique())]
        constructors_modern = constructors[constructors["constructorId"].isin(modern_results["constructorId"].unique())]
        circuits_modern = circuits[circuits["circuitId"].isin(modern_races["circuitId"].unique())]

        return circuits_modern, drivers_modern, constructors_modern
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

model, feature_cols = load_model_and_features()
circuits_df, drivers_df, constructors_df = load_and_filter_datasets()

# --------------------------------------------------
# ENCODERS
# --------------------------------------------------
def encode_weather(weather):
    return {"Dry": 0, "Mixed": 1, "Wet": 2}.get(weather, 0)

def encode_tyre(strategy):
    return {"Conservative": 0, "Balanced": 1, "Aggressive": 2}.get(strategy, 0)

# --------------------------------------------------
# STRATEGY ADJUSTMENT
# --------------------------------------------------
def strategy_adjustment(grid, weather, tyre, pit, form, risk, aggro, team_pressure):

    delta = 0

    delta += {"Dry": 3, "Mixed": 0, "Wet": -5}[weather]
    delta += {"Conservative": -1.5, "Balanced": 1, "Aggressive": 3}[tyre]

    delta += (pit - 5) * 1.1
    delta += (form - 50) * 0.2
    delta -= risk * 0.18
    delta += (aggro - 50) * 0.08
    delta -= team_pressure * 0.07

    if weather in {"Mixed", "Wet"}:
        if grid >= 10:
            delta += 2
        elif grid <= 3:
            delta -= 1

    return delta

# --------------------------------------------------
# PROJECTED POINTS FUNCTION
# --------------------------------------------------
def projected_points(probability):

    avg_podium_points = 19.33
    return round((probability / 100) * avg_podium_points, 2)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.header("🏁 Race Parameters")

    driver_list = (drivers_df["forename"] + " " + drivers_df["surname"]).sort_values()
    constructor_list = constructors_df["name"].sort_values()
    circuit_list = circuits_df["name"].sort_values()

    driver = st.selectbox("Driver", driver_list)
    constructor = st.selectbox("Constructor", constructor_list)
    circuit = st.selectbox("Circuit", circuit_list)

    grid_position = st.slider("Starting Grid", 1, 20, 1)

    with st.expander("⚙️ Strategy Inputs", expanded=True):

        weather = st.selectbox("Weather", ["Dry", "Mixed", "Wet"])
        tyre_strategy = st.selectbox("Tyre Strategy", ["Conservative", "Balanced", "Aggressive"])

        pit_crew_rating = st.slider("Pit Crew", 1, 10, 6)
        recent_form = st.slider("Recent Form", 0, 100, 70)
        reliability_risk = st.slider("Reliability Risk", 0, 100, 15)
        aggression_level = st.slider("Aggression", 0, 100, 60)
        teammate_pressure = st.slider("Teammate Pressure", 0, 100, 35)

    predict_clicked = st.button("🚀 Predict Podium", use_container_width=True)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🏎️ Predictor", "🧠 Model Brain", "🌍 Circuit Map"])

# --------------------------------------------------
# TAB 1
# --------------------------------------------------
with tab1:

    if predict_clicked:

        driver_id = int(drivers_df[(drivers_df["forename"] + " " + drivers_df["surname"]) == driver]["driverId"].values[0])
        constructor_id = int(constructors_df[constructors_df["name"] == constructor]["constructorId"].values[0])
        circuit_id = int(circuits_df[circuits_df["name"] == circuit]["circuitId"].values[0])

        base_input = {
            "circuitId": circuit_id,
            "constructorId": constructor_id,
            "driverId": driver_id,
            "weather_code": encode_weather(weather),
            "tyre_strategy_code": encode_tyre(tyre_strategy),
            "pit_crew_rating": pit_crew_rating,
            "recent_form": recent_form,
            "reliability_risk": reliability_risk,
            "aggression_level": aggression_level,
            "teammate_pressure": teammate_pressure
        }

        # Batch predictions
        sim_inputs = []

        for gp in range(1, 21):
            d = base_input.copy()
            d["grid"] = gp
            sim_inputs.append(d)

        sim_df = pd.DataFrame(sim_inputs).reindex(columns=feature_cols, fill_value=0)

        base_probs = model.predict_proba(sim_df)[:, 1] * 100 if model else np.full(20, 50)

        final_probs = []

        for i, gp in enumerate(range(1, 21)):

            delta = strategy_adjustment(
                gp, weather, tyre_strategy, pit_crew_rating,
                recent_form, reliability_risk,
                aggression_level, teammate_pressure
            )

            final_probs.append(float(np.clip(base_probs[i] + delta, 0, 100)))

        user_prob = final_probs[grid_position - 1]
        user_base = base_probs[grid_position - 1]
        user_delta = user_prob - user_base

        proj_points = projected_points(user_prob)

        # ---------------- UI ----------------

        colA, colB = st.columns([1, 1.5])

        with colA:

            st.subheader(driver)
            st.markdown(f"**{constructor}** | Grid P{grid_position} | {circuit}")

            st.metric(
                "Podium Probability",
                f"{user_prob:.1f}%",
                f"{user_delta:+.1f}% Strategy Impact"
            )

            st.metric(
                "Projected Championship Points",
                f"{proj_points} pts"
            )

            st.caption(f"Raw ML Base Probability: {user_base:.1f}%")

        with colB:

            st.subheader("Grid Position Impact")

            sim_chart = pd.DataFrame({
                "Podium Probability (%)": final_probs,
                "Projected Points": [projected_points(p) for p in final_probs]
            }, index=range(1, 21))

            sim_chart.index.name = "Grid Position"

            st.line_chart(sim_chart)

    else:
        st.info("👈 Select parameters in sidebar and click Predict.")

# --------------------------------------------------
# TAB 2
# --------------------------------------------------
with tab2:

    st.header("Feature Importance")

    if model is not None and feature_cols:

        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance")

        st.bar_chart(importance_df.set_index("Feature"))

    else:
        st.warning("Model not loaded")

# --------------------------------------------------
# TAB 3
# --------------------------------------------------
with tab3:

    st.header("Modern Era F1 Circuits")

    if not circuits_df.empty and {"lat", "lng"}.issubset(circuits_df.columns):

        st.map(circuits_df, latitude="lat", longitude="lng")

    else:
        st.error("Circuit dataset missing coordinates")