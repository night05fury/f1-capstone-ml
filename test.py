# import time
# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st

# # --------------------------------------------------
# # PAGE CONFIG
# # --------------------------------------------------
# st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏎️", layout="wide")

# st.title("🏎️ Formula 1 ML Strategy & Podium Predictor")
# st.markdown("Machine Learning powered podium probability engine (Modern F1 Era ≥ 2010).")

# # --------------------------------------------------
# # LOAD MODEL + FEATURES
# # --------------------------------------------------
# @st.cache_resource
# def load_model_and_features():
#     try:
#         model = joblib.load("f1_model.pkl")
#         feature_cols = joblib.load("f1_model_features.pkl")
#         return model, feature_cols
#     except Exception:
#         return None, None

# model, feature_cols = load_model_and_features()

# # --------------------------------------------------
# # LOAD DATASETS
# # --------------------------------------------------
# @st.cache_data
# def load_datasets():
#     try:
#         circuits = pd.read_csv("datasets/circuits.csv")
#         drivers = pd.read_csv("datasets/drivers.csv")
#         constructors = pd.read_csv("datasets/constructors.csv")
#         races = pd.read_csv("datasets/races.csv")
#         results = pd.read_csv("datasets/results.csv")
#         return circuits, drivers, constructors, races, results
#     except Exception:
#         return (
#             pd.DataFrame(),
#             pd.DataFrame(),
#             pd.DataFrame(),
#             pd.DataFrame(),
#             pd.DataFrame(),
#         )

# circuits_df, drivers_df, constructors_df, races_df, results_df = load_datasets()

# # --------------------------------------------------
# # FILTER MODERN ERA (>= 2010)
# # --------------------------------------------------
# modern_races = races_df[races_df["year"] >= 2010]
# modern_race_ids = modern_races["raceId"].unique()

# modern_results = results_df[results_df["raceId"].isin(modern_race_ids)]

# modern_driver_ids = modern_results["driverId"].unique()
# modern_constructor_ids = modern_results["constructorId"].unique()
# modern_circuit_ids = modern_races["circuitId"].unique()

# drivers_modern = drivers_df[drivers_df["driverId"].isin(modern_driver_ids)]
# constructors_modern = constructors_df[
#     constructors_df["constructorId"].isin(modern_constructor_ids)
# ]
# circuits_modern = circuits_df[circuits_df["circuitId"].isin(modern_circuit_ids)]

# # --------------------------------------------------
# # ENCODERS
# # --------------------------------------------------
# def encode_weather(weather):
#     return {"Dry": 0, "Mixed": 1, "Wet": 2}.get(weather, 0)

# def encode_tyre(strategy):
#     return {"Conservative": 0, "Balanced": 1, "Aggressive": 2}.get(strategy, 0)

# # --------------------------------------------------
# # STRATEGY LAYER
# # --------------------------------------------------
# def strategy_adjustment(
#     grid_position,
#     weather,
#     tyre_strategy,
#     pit_crew_rating,
#     recent_form,
#     reliability_risk,
#     aggression_level,
#     teammate_pressure,
# ):
#     delta = 0.0

#     weather_delta = {"Dry": 3.0, "Mixed": 0.0, "Wet": -5.0}
#     tyre_delta = {"Conservative": -1.5, "Balanced": 1.0, "Aggressive": 3.0}

#     delta += weather_delta.get(weather, 0.0)
#     delta += tyre_delta.get(tyre_strategy, 0.0)
#     delta += (pit_crew_rating - 5.0) * 1.1
#     delta += (recent_form - 50.0) * 0.2
#     delta -= reliability_risk * 0.18
#     delta += (aggression_level - 50.0) * 0.08
#     delta -= teammate_pressure * 0.07

#     if weather in {"Mixed", "Wet"}:
#         if grid_position >= 10:
#             delta += 2.0
#         elif grid_position <= 3:
#             delta -= 1.0

#     return delta

# # --------------------------------------------------
# # TABS
# # --------------------------------------------------
# tab1, tab2, tab3 = st.tabs(
#     ["🏎️ The Predictor", "🧠 Model Brain (Explainability)", "🌍 Circuit Explorer"]
# )

# # ==================================================
# # TAB 1 — PREDICTION
# # ==================================================
# with tab1:

#     st.sidebar.header("Race Parameters")

#     driver_list = (
#         drivers_modern["forename"] + " " + drivers_modern["surname"]
#     ).sort_values()

#     constructor_list = constructors_modern["name"].sort_values()
#     circuit_list = circuits_modern["name"].sort_values()

#     driver = st.sidebar.selectbox("Select Driver", driver_list)
#     constructor = st.sidebar.selectbox("Constructor", constructor_list)
#     circuit = st.sidebar.selectbox("Circuit", circuit_list)

#     grid_position = st.sidebar.slider("Starting Grid Position", 1, 20, 1)

#     st.sidebar.subheader("Advanced Strategy Inputs")

#     weather = st.sidebar.selectbox("Weather", ["Dry", "Mixed", "Wet"])
#     tyre_strategy = st.sidebar.selectbox("Tyre Strategy", ["Conservative", "Balanced", "Aggressive"])
#     pit_crew_rating = st.sidebar.slider("Pit Crew Rating (1-10)", 1, 10, 6)
#     recent_form = st.sidebar.slider("Recent Form (0-100)", 0, 100, 70)
#     reliability_risk = st.sidebar.slider("Reliability Risk (0-100)", 0, 100, 15)
#     aggression_level = st.sidebar.slider("Aggression (0-100)", 0, 100, 60)
#     teammate_pressure = st.sidebar.slider("Teammate Pressure (0-100)", 0, 100, 35)

#     # ---- CENTERED BUTTON ----
#     st.markdown("###")
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col1:
#         predict_clicked = st.button("🚀 Predict Podium Probability", use_container_width=True)

#     if predict_clicked:

#         with st.spinner("Running ML prediction engine..."):
#             time.sleep(1)

#             driver_row = drivers_modern[
#                 (drivers_modern["forename"] + " " + drivers_modern["surname"]) == driver
#             ]
#             constructor_row = constructors_modern[
#                 constructors_modern["name"] == constructor
#             ]
#             circuit_row = circuits_modern[
#                 circuits_modern["name"] == circuit
#             ]

#             driver_id = int(driver_row["driverId"].values[0])
#             constructor_id = int(constructor_row["constructorId"].values[0])
#             circuit_id = int(circuit_row["circuitId"].values[0])

#             input_dict = {
#                 "grid": grid_position,
#                 "circuitId": circuit_id,
#                 "constructorId": constructor_id,
#                 "driverId": driver_id,
#                 "weather_code": encode_weather(weather),
#                 "tyre_strategy_code": encode_tyre(tyre_strategy),
#                 "pit_crew_rating": pit_crew_rating,
#                 "recent_form": recent_form,
#                 "reliability_risk": reliability_risk,
#                 "aggression_level": aggression_level,
#                 "teammate_pressure": teammate_pressure,
#             }

#             input_df = pd.DataFrame([input_dict])[feature_cols]

#             if model is not None:
#                 base_probability = model.predict_proba(input_df)[0][1] * 100
#             else:
#                 base_probability = 50.0

#             delta = strategy_adjustment(
#                 grid_position,
#                 weather,
#                 tyre_strategy,
#                 pit_crew_rating,
#                 recent_form,
#                 reliability_risk,
#                 aggression_level,
#                 teammate_pressure,
#             )

#             probability = float(np.clip(base_probability + delta, 0, 100))

#             colA, colB = st.columns(2)

#             with colA:
#                 st.subheader(f"Results for {driver}")
#                 st.metric("Podium Probability", f"{probability:.1f}%")
#                 st.caption(f"ML Base: {base_probability:.1f}% | Strategy Impact: {delta:+.1f}")

#             with colB:
#                 st.subheader("What-If Grid Simulation")

#                 sim_probs = []
#                 for gp in range(1, 21):
#                     sim_input = input_dict.copy()
#                     sim_input["grid"] = gp
#                     sim_df = pd.DataFrame([sim_input])[feature_cols]

#                     if model is not None:
#                         gp_base = model.predict_proba(sim_df)[0][1] * 100
#                     else:
#                         gp_base = 50.0

#                     gp_delta = strategy_adjustment(
#                         gp,
#                         weather,
#                         tyre_strategy,
#                         pit_crew_rating,
#                         recent_form,
#                         reliability_risk,
#                         aggression_level,
#                         teammate_pressure,
#                     )

#                     sim_probs.append(float(np.clip(gp_base + gp_delta, 0, 100)))

#                 sim_chart = pd.DataFrame(
#                     {"Grid Position": range(1, 21), "Podium Probability (%)": sim_probs}
#                 ).set_index("Grid Position")

#                 st.line_chart(sim_chart)

# # ==================================================
# # TAB 2 — EXPLAINABILITY
# # ==================================================
# with tab2:

#     st.header("Model Explainability")

#     if model is not None:
#         importance_df = pd.DataFrame(
#             {
#                 "Feature": feature_cols,
#                 "Importance": model.feature_importances_,
#             }
#         ).sort_values("Importance", ascending=True).set_index("Feature")

#         st.bar_chart(importance_df)
#     else:
#         st.warning("Model not loaded.")

# # ==================================================
# # TAB 3 — CIRCUIT MAP
# # ==================================================
# with tab3:

#     st.header("Global Modern-Era Circuits (≥ 2010)")

#     if not circuits_modern.empty and {"lat", "lng"}.issubset(circuits_modern.columns):
#         st.map(circuits_modern, latitude="lat", longitude="lng", zoom=1)
#     else:
#         st.error("Circuit dataset missing lat/lng columns.")

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIG & STYLING
# --------------------------------------------------
st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏎️", layout="wide")

# Custom F1-themed CSS for metrics and buttons
st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #1e1e1e;
        border-left: 5px solid #E10600; /* F1 Red */
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div.stButton > button:first-child {
        background-color: #E10600;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff1a1a;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏎️ Formula 1 ML Strategy & Podium Predictor")
st.markdown("Machine Learning powered podium probability engine (Modern F1 Era ≥ 2010).")

# --------------------------------------------------
# LOAD MODEL & DATA (OPTIMIZED CACHING)
# --------------------------------------------------
@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load("f1_model.pkl")
        feature_cols = joblib.load("f1_model_features.pkl")
        return model, feature_cols
    except Exception:
        return None, []

@st.cache_data
def load_and_filter_datasets():
    """Loads datasets and filters for modern era ONCE, keeping the app fast."""
    try:
        circuits = pd.read_csv("datasets/circuits.csv")
        drivers = pd.read_csv("datasets/drivers.csv")
        constructors = pd.read_csv("datasets/constructors.csv")
        races = pd.read_csv("datasets/races.csv")
        results = pd.read_csv("datasets/results.csv")

        # Filter >= 2010
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
# ENCODERS & STRATEGY LAYER
# --------------------------------------------------
def encode_weather(weather):
    return {"Dry": 0, "Mixed": 1, "Wet": 2}.get(weather, 0)

def encode_tyre(strategy):
    return {"Conservative": 0, "Balanced": 1, "Aggressive": 2}.get(strategy, 0)

def strategy_adjustment(grid, weather, tyre, pit, form, risk, aggro, team_pressure):
    delta = 0.0
    delta += {"Dry": 3.0, "Mixed": 0.0, "Wet": -5.0}.get(weather, 0.0)
    delta += {"Conservative": -1.5, "Balanced": 1.0, "Aggressive": 3.0}.get(tyre, 0.0)
    delta += (pit - 5.0) * 1.1
    delta += (form - 50.0) * 0.2
    delta -= risk * 0.18
    delta += (aggro - 50.0) * 0.08
    delta -= team_pressure * 0.07

    if weather in {"Mixed", "Wet"}:
        if grid >= 10: delta += 2.0
        elif grid <= 3: delta -= 1.0

    return delta

# --------------------------------------------------
# SIDEBAR UI
# --------------------------------------------------
with st.sidebar:
    st.header("🏁 Race Parameters")
    
    # Safe fallback if datasets fail to load
    d_list = (drivers_df["forename"] + " " + drivers_df["surname"]).sort_values() if not drivers_df.empty else ["Unknown"]
    c_list = constructors_df["name"].sort_values() if not constructors_df.empty else ["Unknown"]
    circ_list = circuits_df["name"].sort_values() if not circuits_df.empty else ["Unknown"]

    driver = st.selectbox("Select Driver", d_list)
    constructor = st.selectbox("Constructor", c_list)
    circuit = st.selectbox("Circuit", circ_list)
    grid_position = st.slider("Starting Grid Position", 1, 20, 1)

    with st.expander("⚙️ Advanced Strategy Inputs", expanded=True):
        weather = st.selectbox("Weather", ["Dry", "Mixed", "Wet"])
        tyre_strategy = st.selectbox("Tyre Strategy", ["Conservative", "Balanced", "Aggressive"])
        pit_crew_rating = st.slider("Pit Crew Rating", 1, 10, 6)
        recent_form = st.slider("Recent Form (0-100)", 0, 100, 70)
        reliability_risk = st.slider("Reliability Risk (0-100)", 0, 100, 15)
        aggression_level = st.slider("Aggression (0-100)", 0, 100, 60)
        teammate_pressure = st.slider("Teammate Pressure (0-100)", 0, 100, 35)

    predict_clicked = st.button("🚀 Predict Podium", use_container_width=True)

# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🏎️ The Predictor", "🧠 Model Brain", "🌍 Circuit Explorer"])

with tab1:
    if predict_clicked:
        if drivers_df.empty or model is None:
            st.error("⚠️ Data or Model missing. Please ensure datasets/ and .pkl files exist.")
            st.stop()

        with st.spinner("Crunching telemetry and ML weights..."):
            # ID Mapping
            driver_id = int(drivers_df[(drivers_df["forename"] + " " + drivers_df["surname"]) == driver]["driverId"].values[0])
            constructor_id = int(constructors_df[constructors_df["name"] == constructor]["constructorId"].values[0])
            circuit_id = int(circuits_df[circuits_df["name"] == circuit]["circuitId"].values[0])

            # Base Input Dictionary
            base_input = {
                "circuitId": circuit_id, "constructorId": constructor_id, "driverId": driver_id,
                "weather_code": encode_weather(weather), "tyre_strategy_code": encode_tyre(tyre_strategy),
                "pit_crew_rating": pit_crew_rating, "recent_form": recent_form,
                "reliability_risk": reliability_risk, "aggression_level": aggression_level,
                "teammate_pressure": teammate_pressure,
            }

            # 1. BATCH PREDICTION FOR WHAT-IF SIMULATION (MASSIVE SPEED BOOST)
            # Create a dataframe of 20 rows (Grid 1 to 20) in one go
            sim_inputs = []
            for gp in range(1, 21):
                d = base_input.copy()
                d["grid"] = gp
                sim_inputs.append(d)
            
            # Ensure columns match training exactly
            sim_df = pd.DataFrame(sim_inputs).reindex(columns=feature_cols, fill_value=0)
            
            # Predict all 20 positions at once
            base_probs = model.predict_proba(sim_df)[:, 1] * 100 if model else np.full(20, 50.0)

            # Apply Strategy Adjustments to all 20 positions
            final_probs = []
            for i, gp in enumerate(range(1, 21)):
                delta = strategy_adjustment(gp, weather, tyre_strategy, pit_crew_rating, recent_form, reliability_risk, aggression_level, teammate_pressure)
                final_probs.append(float(np.clip(base_probs[i] + delta, 0, 100)))

            # Extract the user's specific grid position result
            user_prob = final_probs[grid_position - 1]
            user_base = base_probs[grid_position - 1]
            user_delta = user_prob - user_base

            # ------------------------------
            # UI DISPLAY
            # ------------------------------
            colA, colB = st.columns([1, 1.5])

            with colA:
                with st.container(border=True):
                    st.subheader(f"{driver}")
                    st.markdown(f"**{constructor}** | Grid: P{grid_position} | {circuit}")
                    st.divider()
                    
                    # Using Streamlit's built-in metric delta feature
                    st.metric(
                        label="Podium Probability", 
                        value=f"{user_prob:.1f}%", 
                        delta=f"{user_delta:+.1f}% Strategy Impact",
                        delta_color="normal" if user_delta > 0 else "inverse"
                    )
                    
                    st.caption(f"Raw ML Base Probability: {user_base:.1f}%")

            with colB:
                with st.container(border=True):
                    st.subheader("Grid Position Impact (P1 - P20)")
                    sim_chart = pd.DataFrame({"Podium Probability (%)": final_probs}, index=range(1, 21))
                    sim_chart.index.name = "Grid Position"
                    # Add a red color to the chart to fit the theme
                    st.line_chart(sim_chart, color="#E10600")

    else:
        st.info("👈 Select your parameters in the sidebar and click **Predict Podium** to run the simulation.")

with tab2:
    st.header("🧠 Feature Importance")
    if model is not None and feature_cols:
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True).set_index("Feature")
        st.bar_chart(importance_df, color="#E10600")
    else:
        st.warning("Model not loaded. Cannot display feature importance.")

with tab3:
    st.header("🌍 Global Modern-Era Circuits")
    if not circuits_df.empty and {"lat", "lng"}.issubset(circuits_df.columns):
        st.map(circuits_df, latitude="lat", longitude="lng", zoom=1, color="#E10600")
    else:
        st.error("Circuit dataset missing lat/lng columns.")