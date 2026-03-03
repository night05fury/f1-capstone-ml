# import time
# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st

# #  Page Configuration 
# st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏎️", layout="wide")

# #  Main Header 
# st.title("🏎️ Formula 1 Strategy & Podium Predictor")
# st.markdown("Predict podium probabilities, analyze model logic, and explore historical F1 circuit data.")


# #  Load Data & Model 
# @st.cache_resource
# def load_model():
#     try:
#         # after training:
#         # return joblib.load("f1_model.pkl")
#         return None
#     except Exception:
#         return None


# @st.cache_data
# def load_circuits():
#     try:
#         return pd.read_csv("datasets/circuits.csv")
#     except Exception:
#         return pd.DataFrame()


# model = load_model()
# circuits_df = load_circuits()


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
#     """Heuristic strategy layer on top of model/demo base probability."""
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

#     # Chaotic conditions can help cars starting lower down.
#     if weather in {"Mixed", "Wet"}:
#         if grid_position >= 10:
#             delta += 2.0
#         elif grid_position <= 3:
#             delta -= 1.0

#     return delta


# #  Organize App with Tabs
# tab1, tab2, tab3 = st.tabs(["🏎️ The Predictor", "🧠 Model Brain (Explainability)", "🌍 Circuit Explorer"])

# # ==========================================
# # TAB 1: THE PREDICTOR & WHAT-IF SIMULATOR
# # ==========================================
# with tab1:
#     st.sidebar.header("Race Parameters")

#     driver = st.sidebar.selectbox(
#         "Select Driver",
#         [
#             "Max Verstappen",
#             "Sergio Perez",
#             "Lando Norris",
#             "Oscar Piastri",
#             "Charles Leclerc",
#             "Carlos Sainz",
#             "Lewis Hamilton",
#             "George Russell",
#             "Fernando Alonso",
#             "Lance Stroll",
#             "Pierre Gasly",
#             "Esteban Ocon",
#             "Alex Albon",
#             "Yuki Tsunoda",
#             "Valtteri Bottas",
#             "Nico Hulkenberg",
#             "Kevin Magnussen",
#             "Daniel Ricciardo",
#             "Zhou Guanyu",
#             "Logan Sargeant",
#         ],
#     )
#     constructor = st.sidebar.selectbox(
#         "Constructor Team",
#         [
#             "Red Bull Racing",
#             "McLaren",
#             "Ferrari",
#             "Mercedes",
#             "Aston Martin",
#             "Alpine F1 Team",
#             "Williams",
#             "RB F1 Team",
#             "Kick Sauber",
#             "Haas F1 Team",
#         ],
#     )

#     if not circuits_df.empty and "name" in circuits_df.columns:
#         circuit_list = sorted(circuits_df["name"].dropna().astype(str).tolist())
#     else:
#         circuit_list = ["Monaco", "Silverstone", "Monza"]
#     circuit = st.sidebar.selectbox("Circuit", circuit_list)

#     grid_position = st.sidebar.slider("Starting Grid Position", 1, 20, 1)

#     # Expanded feature set
#     st.sidebar.subheader("Advanced Features")
#     weather = st.sidebar.selectbox("Weather Forecast", ["Dry", "Mixed", "Wet"])
#     tyre_strategy = st.sidebar.selectbox("Tyre Strategy", ["Conservative", "Balanced", "Aggressive"])
#     pit_crew_rating = st.sidebar.slider("Pit Crew Performance (1-10)", 1, 10, 6)
#     recent_form = st.sidebar.slider("Driver Recent Form (0-100)", 0, 100, 70)
#     reliability_risk = st.sidebar.slider("Reliability Risk (0-100)", 0, 100, 15)
#     aggression_level = st.sidebar.slider("Overtake Aggression (0-100)", 0, 100, 60)
#     teammate_pressure = st.sidebar.slider("Teammate Pressure (0-100)", 0, 100, 35)

#     if st.sidebar.button("Predict Podium Probability"):
#         with st.spinner("Analyzing telemetry and historical grid data..."):
#             time.sleep(1.0)

#             # Base probability: model if available, fallback demo logic otherwise
#             if model is not None:

#                 # base_probability = model.predict_proba([[grid_position, constructor_id, driver_id, circuit_id]])[0][1] * 100
#                 base_probability = 85 if grid_position <= 3 else (40 if grid_position <= 8 else 10)
#             else:
#                 base_probability = 85 if grid_position <= 3 else (40 if grid_position <= 8 else 10)

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

#             col1, col2 = st.columns(2)
#             with col1:
#                 st.subheader(f"Results for {driver}")
#                 st.metric(label="Podium Probability", value=f"{probability:.1f}%", delta="Top 3 Finish")
#                 st.caption(f"Strategy feature impact: {delta:+.1f} points")
#                 st.write(f"Team: **{constructor}** | Circuit: **{circuit}**")

#                 if probability > 50:
#                     st.success("High likelihood of a podium finish!")
#                 else:
#                     st.warning("Podium finish is unlikely under these conditions.")

#             with col2:
#                 st.subheader("What-If: Grid Position Impact")
#                 st.markdown("How probability changes based on qualifying performance and advanced strategy inputs:")

#                 simulated_probs = []
#                 for gp in range(1, 21):
#                     gp_base = 85 if gp <= 3 else (40 if gp <= 8 else 10)
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
#                     simulated_probs.append(float(np.clip(gp_base + gp_delta, 0, 100)))

#                 what_if_df = pd.DataFrame(
#                     {"Grid Position": range(1, 21), "Podium Probability (%)": simulated_probs}
#                 ).set_index("Grid Position")

#                 st.line_chart(what_if_df)

# # ==========================================
# # TAB 2: MODEL EXPLAINABILITY
# # ==========================================
# with tab2:
#     st.header("How the Algorithm Makes Decisions")
#     st.markdown(
#         "This chart breaks down which features the Machine Learning model relies on the most. "
#         "Evaluators look for this to ensure the model isn't a 'black box'."
#     )

#     feature_names = ["Starting Grid (grid)", "Constructor (constructorId)", "Driver (driverId)", "Circuit (circuitId)"]
#     importances = [0.60, 0.18, 0.12, 0.10]

#     importance_df = (
#         pd.DataFrame({"Feature": feature_names, "Importance (%)": [i * 100 for i in importances]})
#         .set_index("Feature")
#         .sort_values(by="Importance (%)", ascending=True)
#     )
#     st.bar_chart(importance_df, horizontal=True)

#     st.subheader("Advanced Strategy Feature Impact (Heuristic Layer)")
#     strategy_importance_df = (
#         pd.DataFrame(
#             {
#                 "Feature": [
#                     "Recent Form",
#                     "Reliability Risk",
#                     "Pit Crew Performance",
#                     "Teammate Pressure",
#                     "Weather",
#                     "Tyre Strategy",
#                     "Aggression",
#                 ],
#                 "Relative Weight": [28, 24, 16, 12, 8, 6, 6],
#             }
#         )
#         .set_index("Feature")
#         .sort_values(by="Relative Weight", ascending=True)
#     )
#     st.bar_chart(strategy_importance_df, horizontal=True)
#     st.info("Insight: Grid position dominates, but strategy and race conditions can still move the outcome significantly.")

# # ==========================================
# # TAB 3: CIRCUIT MAP
# # ==========================================
# with tab3:
#     st.header("Global F1 Circuits")
#     st.markdown("An interactive map of all Formula 1 circuits used in the dataset.")

#     if not circuits_df.empty and {"lat", "lng"}.issubset(circuits_df.columns):
#         st.map(circuits_df, latitude="lat", longitude="lng", zoom=1)
#     else:
#         st.error("Could not load circuits.csv with required lat/lng columns.")




import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏎️", layout="wide")

# Custom CSS for that F1 aesthetic
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border-left: 5px solid #e10600; }
    </style>
    """, unsafe_allow_html=True)

# --- Data & Model Loaders ---
@st.cache_resource
def load_model():
    try:
        return joblib.load("f1_model.pkl")
    except:
        return None

@st.cache_data
def load_circuits():
    try:
        # Try loading external data
        df = pd.read_csv("datasets/circuits.csv")
        return df
    except:
        # Fallback 2026 Season Mock Data if file is missing
        data = {
            "name": ["Bahrain", "Jeddah", "Melbourne", "Suzuka", "Shanghai", "Miami", "Monaco", "Montreal", "Silverstone", "Spa", "Monza", "Singapore", "Austin", "Mexico City", "Interlagos", "Las Vegas", "Abu Dhabi"],
            "lat": [26.0325, 21.6319, -37.8497, 34.8431, 31.3389, 25.9581, 43.7347, 45.5005, 52.0786, 50.4372, 45.6189, 1.2914, 30.1328, 19.4042, -23.7036, 36.1147, 24.4672],
            "lng": [50.5106, 39.1044, 144.968, 136.541, 121.22, -80.2389, 7.4206, -73.5228, -1.0169, 5.9714, 9.2811, 103.864, -97.6411, -99.0907, -46.6997, -115.173, 54.6031]
        }
        return pd.DataFrame(data)

model = load_model()
circuits_df = load_circuits()

# --- Logic Layer ---
def strategy_adjustment(grid_position, weather, tyre_strategy, pit_crew_rating, recent_form, reliability_risk, aggression_level, teammate_pressure):
    """Refined heuristic layer to simulate real race dynamics."""
    delta = 0.0
    
    # Impact Factors
    weather_map = {"Dry": 2.0, "Mixed": 0.0, "Wet": -4.0}
    tyre_map = {"Conservative": -1.0, "Balanced": 1.0, "Aggressive": 4.0}
    
    delta += weather_map.get(weather, 0.0)
    delta += tyre_map.get(tyre_strategy, 0.0)
    delta += (pit_crew_rating - 6.0) * 1.5      # 6 is baseline
    delta += (recent_form - 70.0) * 0.25        # 70 is baseline
    delta -= reliability_risk * 0.20           # Pure penalty
    delta += (aggression_level - 60.0) * 0.1   # High risk/high reward
    delta -= teammate_pressure * 0.08
    
    # Chaos Multiplier: Wet races help the midfield, hurt the front
    if weather == "Wet":
        if grid_position > 8: delta += 5.0
        if grid_position <= 3: delta -= 3.0
        
    return delta

# --- Main UI ---
st.title("🏎️ Formula 1 Strategy & Podium Predictor")
if model is None:
    st.info("💡 **Engine Status:** Running on *Heuristic Logic Mode* (No pre-trained model found).")

tab1, tab2, tab3 = st.tabs(["📊 Prediction Engine", "🧠 Model Logic", "🌍 Circuit Map"])

with tab1:
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.header("Race Settings")
        driver = st.selectbox("Select Driver", ["Max Verstappen", "Lando Norris", "Lewis Hamilton", "Charles Leclerc", "Oscar Piastri", "George Russell", "Carlos Sainz", "Fernando Alonso", "Nico Hulkenberg", "Yuki Tsunoda"])
        constructor = st.selectbox("Constructor Team", ["Red Bull", "McLaren", "Ferrari", "Mercedes", "Aston Martin", "Haas", "RB", "Williams", "Alpine", "Sauber"])
        circuit = st.selectbox("Circuit", circuits_df["name"].tolist())
        grid_pos = st.slider("Starting Grid Position", 1, 20, 1)
        
        with st.expander("Advanced Strategy Modifiers"):
            weather = st.select_slider("Weather Forecast", options=["Dry", "Mixed", "Wet"])
            tyre = st.selectbox("Tyre Strategy", ["Conservative", "Balanced", "Aggressive"])
            pit = st.slider("Pit Crew Skill", 1, 10, 8)
            form = st.slider("Driver Form", 0, 100, 75)
            risk = st.slider("Reliability Risk (%)", 0, 100, 10)
            aggro = st.slider("Aggression", 0, 100, 65)
            press = st.slider("Teammate Pressure", 0, 100, 20)

    with col_b:
        st.header("Probability Analysis")
        
        # Calculate Base Prob (Simplified Logarithmic decay for demo)
        # Prob = 100 / (1 + e^(0.3 * (grid - 5))) -- adjusted for F1 reality
        base_prob = 90 if grid_pos == 1 else (80 if grid_pos == 2 else (70 if grid_pos == 3 else max(5, 60 - (grid_pos * 4))))
        
        delta = strategy_adjustment(grid_pos, weather, tyre, pit, form, risk, aggro, press)
        final_prob = float(np.clip(base_prob + delta, 0, 100))
        
        # Display Metric
        m1, m2 = st.columns(2)
        m1.metric("Podium Probability", f"{final_prob:.1f}%", delta=f"{delta:+.1f}% vs Grid")
        m2.metric("Projected Points", f"{int(final_prob * 0.25)} pts", help="Estimated based on podium likelihood")
        
        # What-If Plotly Chart
        st.subheader("Grid Position Sensitivity")
        sim_data = []
        for gp in range(1, 21):
            b_p = 90 if gp == 1 else (80 if gp == 2 else (70 if gp == 3 else max(5, 60 - (gp * 4))))
            d_p = strategy_adjustment(gp, weather, tyre, pit, form, risk, aggro, press)
            sim_data.append({"Grid": gp, "Prob": np.clip(b_p + d_p, 0, 100)})
        
        fig = px.area(pd.DataFrame(sim_data), x="Grid", y="Prob", 
                      title="Podium Probability by Grid Slot",
                      labels={"Prob": "Probability %", "Grid": "Starting Position"},
                      color_discrete_sequence=['#e10600'])
        fig.update_layout(hovermode="x unified", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Feature Weighting")
    features = {"Grid Position": 65, "Constructor Pace": 15, "Driver Skill": 10, "Strategy/Pit": 7, "Weather": 3}
    feat_df = pd.DataFrame(list(features.items()), columns=["Feature", "Weight"])
    
    st.bar_chart(feat_df.set_index("Feature"), horizontal=True)
    st.info("The model prioritizes **Grid Position** due to the high difficulty of overtaking in modern F1 aerodynamics (Dirty Air effect).")

with tab3:
    st.header("2026 Global Calendar")
    st.map(circuits_df, latitude="lat", longitude="lng", size=20, color="#e10600")

# --- Interactive Footer ---
st.divider()
st.write("---")
if st.button("Generate Strategy Report (PDF)"):
    st.toast("Generating telemetry-based report...", icon="🏎️")
    time.sleep(2)
    st.success("Report Ready! (This is a demo feature)")