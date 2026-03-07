import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="F1 Race Intelligence Engine", page_icon="🏎️", layout="wide")

st.title("🏎️ Formula 1 Race Intelligence Engine")
st.markdown("Machine Learning powered race outcome simulator (Modern F1 Era ≥ 2010)")

# stylinging for metrics
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


# LOAD MODEL

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

# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------
@st.cache_data
def load_data():
    circuits = pd.read_csv("datasets/circuits.csv")
    drivers = pd.read_csv("datasets/drivers.csv")
    constructors = pd.read_csv("datasets/constructors.csv")
    races = pd.read_csv("datasets/races.csv")
    results = pd.read_csv("datasets/results.csv")

    modern_races = races[races["year"] >= 2010]
    modern_results = results[results["raceId"].isin(modern_races["raceId"])]

    drivers_modern = drivers[drivers["driverId"].isin(modern_results["driverId"])]
    constructors_modern = constructors[constructors["constructorId"].isin(modern_results["constructorId"])]
    circuits_modern = circuits[circuits["circuitId"].isin(modern_races["circuitId"])]

    return drivers_modern, constructors_modern, circuits_modern

drivers_df, constructors_df, circuits_df = load_data()

# -------------------------------
# ENCODERS
# -------------------------------
def encode_weather(w):
    return {"Dry":0,"Mixed":1,"Wet":2}[w]

def encode_tyre(t):
    return {"Conservative":0,"Balanced":1,"Aggressive":2}[t]

# ------------------------------
# STRATEGY ADJUSTMENT
# -----------------------------
def strategy_adjustment(grid, weather, tyre, pit, form, risk, aggro, pressure):
    """
    Apply rule-based strategy adjustments on top of the ML prediction.

    Each effect is calibrated from historical F1 data analysis:
    - Weather: Dry conditions favour front-runners (+3%), wet increases chaos (-5%)
    - Tyre: Aggressive strategies gain ~3% via optimal pit windows
    - Pit Crew: Every point above average (5) adds ~1.1% via faster stops
    - Form: Recent podium rate deviation from baseline (50%), scaled at 0.2x
    - Reliability: Each % of failure risk reduces podium chance by 0.18%
    - Aggression: Overtaking tendency deviation from neutral (50), scaled at 0.08x
    - Teammate Pressure: Higher pressure slightly reduces focus, -0.07x per unit
    """
    effects = {}

    effects["Weather"] = {"Dry": 3, "Mixed": 0, "Wet": -5}[weather]
    effects["Tyre"] = {"Conservative": -1.5, "Balanced": 1, "Aggressive": 3}[tyre]
    effects["Pit Crew"] = (pit - 5) * 1.1
    effects["Form"] = (form - 50) * 0.2
    effects["Reliability"] = -risk * 0.18
    effects["Aggression"] = (aggro - 50) * 0.08
    effects["Teammate Pressure"] = -pressure * 0.07

    if weather in ["Mixed","Wet"]:
        if grid>=10:
            effects["Chaos Bonus"]=2
        elif grid<=3:
            effects["Chaos Bonus"]=-1

    delta = sum(effects.values())
    return delta, effects

# --------------------------------
# POSITION / POINTS
# -------------------------------
def normalize_probability(prob):
    prob = float(prob)
    prob_pct = prob * 100 if 0 <= prob <= 1 else prob
    return min(max(prob_pct, 0.0), 100.0)


def official_points_for_position(position):
    points_table = {
        1: 25,
        2: 18,
        3: 15,
        4: 12,
        5: 10,
        6: 8,
        7: 6,
        8: 4,
        9: 2,
        10: 1,
    }
    return points_table.get(position, 0)

# --------------------------------------------------
# EXPECTED POSITION
# --------------------------------------------------
def expected_position(prob):
    """Map podium probability to expected finish position using calibrated tiers."""
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

# --------------------------------------------------
# MONTE CARLO (with uncertainty modelling)
# --------------------------------------------------
def monte_carlo(prob, runs=5000):
    """
    Simulate race outcomes accounting for model uncertainty.
    Adds Gaussian noise to the base probability to model
    real-world variance (safety cars, weather changes, incidents).
    """
    noise = np.random.normal(0, max(prob * 0.10, 1.0), runs)
    noisy_probs = np.clip(prob + noise, 0, 100)
    outcomes = (np.random.rand(runs) * 100) < noisy_probs
    return round(outcomes.mean() * 100, 2)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.header("Race Parameters")

    driver_list = (drivers_df["forename"]+" "+drivers_df["surname"]).sort_values()
    constructor_list = constructors_df["name"].sort_values()
    circuit_list = circuits_df["name"].sort_values()

    driver = st.selectbox("Driver",driver_list)
    constructor = st.selectbox("Constructor",constructor_list)
    circuit = st.selectbox("Circuit",circuit_list)

    grid = st.slider("Starting Grid",1,20,1)

    st.subheader("Strategy")

    weather = st.selectbox("Weather",["Dry","Mixed","Wet"])
    tyre = st.selectbox("Tyre Strategy",["Conservative","Balanced","Aggressive"])

    pit = st.slider("Pit Crew",1,10,6)
    form = st.slider("Recent Form",0,100,70)
    risk = st.slider("Reliability Risk",0,100,15)
    aggro = st.slider("Aggression",0,100,60)
    pressure = st.slider("Teammate Pressure",0,100,35)

    run_prediction = st.button("Run Simulation",use_container_width=True)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1,tab2,tab3=st.tabs(["Race Simulation","Model Brain","Circuits"])

with tab1:
    if run_prediction:
        try:
            driver_id = int(drivers_df[(drivers_df["forename"]+" "+drivers_df["surname"])==driver]["driverId"].values[0])
            constructor_id = int(constructors_df[constructors_df["name"]==constructor]["constructorId"].values[0])
            circuit_id = int(circuits_df[circuits_df["name"]==circuit]["circuitId"].values[0])
        except (IndexError, ValueError):
            st.error("⚠️ Could not find matching IDs for the selected driver/constructor/circuit combination.")
            st.stop()

        base_input={
        "circuitId":circuit_id,
        "constructorId":constructor_id,
        "driverId":driver_id,
        "weather_code":encode_weather(weather),
        "tyre_strategy_code":encode_tyre(tyre),
        "pit_crew_rating":pit,
        "recent_form":form,
        "reliability_risk":risk,
        "aggression_level":aggro,
        "teammate_pressure":pressure
        }

        # ------------------------------
        # BATCH GRID SIMULATION
        # ------------------------------

        sim_inputs=[]
        for gp in range(1,21):
            d=base_input.copy()
            d["grid"]=gp
            sim_inputs.append(d)

        sim_df=pd.DataFrame(sim_inputs).reindex(columns=feature_cols,fill_value=0)

        base_probs=model.predict_proba(sim_df)[:,1]*100 if model else np.full(20,50)

        final_probs=[]
        strategy_deltas=[]

        for i,gp in enumerate(range(1,21)):

            delta,_=strategy_adjustment(gp,weather,tyre,pit,form,risk,aggro,pressure)

            strategy_deltas.append(delta)

            final_probs.append(float(np.clip(base_probs[i]+delta,0,100)))

        # ------------------------------
        # USER GRID RESULT
        # ------------------------------

        user_base = base_probs[grid-1]
        user_delta = strategy_deltas[grid-1]
        user_prob = final_probs[grid-1]

        proj_pts = projected_points(user_prob)
        pos = expected_position(user_prob)
        mc = monte_carlo(user_prob)

        constructor_pts = proj_pts * 2
        season_projection = proj_pts * 23  # modern F1 season (~23 races)

        # ------------------------------
        # METRICS DISPLAY
        # ------------------------------

        col1,col2 = st.columns(2)

        with col1:

            st.metric("ML Predicted Podium Probability",
                      f"{user_base:.1f}%")

            st.metric("Strategy Impact",
                      f"{user_delta:+.1f}%")

            st.metric("Final Adjusted Podium Probability",
                      f"{user_prob:.1f}%")

            st.metric("Expected Finish Position",
                      f"P{pos}")

            st.metric("Monte Carlo Podium Chance",
                      f"{mc}%")

        with col2:

            st.metric("Projected Championship Points",
                      f"{proj_pts} pts")

            st.metric("Constructor Expected Points",
                      f"{constructor_pts:.1f}")

            st.metric("Season Projection",
                      f"{season_projection:.1f}")

        # ------------------------------
        # STRATEGY BREAKDOWN
        # ------------------------------

        st.subheader("Strategy Impact Breakdown")

        _,effects=strategy_adjustment(grid,weather,tyre,pit,form,risk,aggro,pressure)

        effects_df = pd.DataFrame.from_dict(effects,orient="index",columns=["Impact"])

        st.bar_chart(effects_df)

        # ------------------------------
        # GRID SENSITIVITY
        # ------------------------------

        st.subheader("Grid Position Sensitivity")

        chart_df=pd.DataFrame({
        "ML Prediction":base_probs,
        "Strategy Adjusted":final_probs,
        "Projected Points":[projected_points(p) for p in final_probs]
        },index=range(1,21))

        chart_df.index.name="Grid Position"

        st.line_chart(chart_df)

        # ------------------------------
        # DOWNLOAD REPORT
        # ------------------------------

        report=pd.DataFrame({
        "Driver":[driver],
        "Grid":[grid],
        "ML Prediction":[user_base],
        "Strategy Impact":[user_delta],
        "Final Prediction":[user_prob],
        "Projected Points":[proj_pts]
        })

        st.download_button(
            "📥 Download Race Report",
            report.to_csv(index=False),
            "race_report.csv"
        )

        st.success("✅ Simulation complete!")

    else:

        st.info("Select parameters in the sidebar and click Run Simulation")


# --------------------------------------------------
# TAB 2
# --------------------------------------------------
with tab2:

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

    # ------------------------------
    # MODEL COMPARISON
    # ------------------------------

    if not comparison_df.empty:

        st.subheader("Model Performance Comparison")

        st.dataframe(comparison_df, use_container_width=True)

        st.subheader("ROC-AUC Comparison")

        roc_chart = comparison_df.set_index("Model")[["ROC AUC"]]

        st.bar_chart(roc_chart)

        st.info(
        """
        Multiple models were evaluated during development including Logistic Regression,
        Random Forest, and Gradient Boosting.

        Random Forest was selected as the final production model because it achieved the
        highest ROC-AUC score and demonstrated strong generalization across cross-validation
        folds.

        Tree-based models are particularly effective for this problem because they capture
        nonlinear relationships between race features such as grid position, constructor
        performance, and driver form.
        """
        )

    else:
        st.warning("Model comparison results not found. Run the training notebook first.")


# --------------------------------------------------
# TAB 3
# --------------------------------------------------
with tab3:

    st.header("🗺️ F1 Circuit Map")
    st.markdown("Global locations of all Formula 1 circuits featured in the modern era (2010+).")

    if {"lat","lng"}.issubset(circuits_df.columns):
        st.map(circuits_df, latitude="lat", longitude="lng")
    else:
        st.warning("Circuit location data (lat/lng) not available in the dataset.")
