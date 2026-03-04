import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="F1 Race Intelligence Engine", page_icon="🏎️", layout="wide")

st.title("🏎️ Formula 1 Race Intelligence Engine")
st.markdown("Machine Learning powered race outcome simulator (Modern F1 Era ≥ 2010)")

# --------------------------------------------------
# STYLE
# --------------------------------------------------
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

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load("f1_model.pkl")
        feature_cols = joblib.load("f1_model_features.pkl")
        return model, feature_cols
    except:
        return None, []

model, feature_cols = load_model_and_features()

# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------
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

# --------------------------------------------------
# ENCODERS
# --------------------------------------------------
def encode_weather(w):
    return {"Dry":0,"Mixed":1,"Wet":2}[w]

def encode_tyre(t):
    return {"Conservative":0,"Balanced":1,"Aggressive":2}[t]

# --------------------------------------------------
# STRATEGY ADJUSTMENT
# --------------------------------------------------
def strategy_adjustment(grid, weather, tyre, pit, form, risk, aggro, pressure):

    effects = {}

    effects["Weather"] = {"Dry":3,"Mixed":0,"Wet":-5}[weather]
    effects["Tyre"] = {"Conservative":-1.5,"Balanced":1,"Aggressive":3}[tyre]
    effects["Pit Crew"] = (pit-5)*1.1
    effects["Form"] = (form-50)*0.2
    effects["Reliability"] = -risk*0.18
    effects["Aggression"] = (aggro-50)*0.08
    effects["Teammate Pressure"] = -pressure*0.07

    if weather in ["Mixed","Wet"]:
        if grid>=10:
            effects["Chaos Bonus"]=2
        elif grid<=3:
            effects["Chaos Bonus"]=-1

    delta = sum(effects.values())
    return delta, effects

# --------------------------------------------------
# PROJECTED POINTS
# --------------------------------------------------
def projected_points(prob):
    avg_podium_points = 19.33
    return round((prob/100)*avg_podium_points,2)

# --------------------------------------------------
# EXPECTED POSITION
# --------------------------------------------------
def expected_position(prob):

    if prob >=80: return 2
    if prob >=60: return 3
    if prob >=40: return 5
    if prob >=25: return 7
    if prob >=10: return 10
    return 14

# --------------------------------------------------
# MONTE CARLO
# --------------------------------------------------
def monte_carlo(prob, runs=1000):
    outcomes = np.random.rand(runs) < (prob/100)
    return round(outcomes.mean()*100,2)

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

    driver2 = st.selectbox("Compare Driver",driver_list)

    run_prediction = st.button("Run Simulation",use_container_width=True)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1,tab2,tab3=st.tabs(["Race Simulation","Model Brain","Circuits"])

# --------------------------------------------------
# TAB 1
# --------------------------------------------------
with tab1:

    if run_prediction:

        driver_id = int(drivers_df[(drivers_df["forename"]+" "+drivers_df["surname"])==driver]["driverId"].values[0])
        constructor_id = int(constructors_df[constructors_df["name"]==constructor]["constructorId"].values[0])
        circuit_id = int(circuits_df[circuits_df["name"]==circuit]["circuitId"].values[0])

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

        sim_inputs=[]
        for gp in range(1,21):
            d=base_input.copy()
            d["grid"]=gp
            sim_inputs.append(d)

        sim_df=pd.DataFrame(sim_inputs).reindex(columns=feature_cols,fill_value=0)

        base_probs=model.predict_proba(sim_df)[:,1]*100 if model else np.full(20,50)

        final_probs=[]
        for i,gp in enumerate(range(1,21)):

            delta,effects=strategy_adjustment(gp,weather,tyre,pit,form,risk,aggro,pressure)

            final_probs.append(float(np.clip(base_probs[i]+delta,0,100)))

        user_prob=final_probs[grid-1]
        user_base=base_probs[grid-1]

        proj_pts=projected_points(user_prob)
        pos=expected_position(user_prob)
        mc=monte_carlo(user_prob)

        constructor_pts=proj_pts*2
        season_projection=proj_pts*8

        col1,col2=st.columns(2)

        with col1:

            st.metric("Podium Probability",f"{user_prob:.1f}%")
            st.metric("Projected Points",f"{proj_pts} pts")
            st.metric("Expected Finish",f"P{pos}")

            st.metric("Monte Carlo Podium Chance",f"{mc}%")

        with col2:

            st.metric("Constructor Expected Points",f"{constructor_pts:.1f}")
            st.metric("Season Projection",f"{season_projection:.1f}")

        st.subheader("Strategy Impact Breakdown")

        _,effects=strategy_adjustment(grid,weather,tyre,pit,form,risk,aggro,pressure)
        st.bar_chart(pd.DataFrame.from_dict(effects,orient="index",columns=["Impact"]))

        st.subheader("Grid Sensitivity")

        chart_df=pd.DataFrame({
        "Podium Probability":final_probs,
        "Projected Points":[projected_points(p) for p in final_probs]
        },index=range(1,21))

        st.line_chart(chart_df)

        report=pd.DataFrame({
        "Driver":[driver],
        "Grid":[grid],
        "Podium Probability":[user_prob],
        "Projected Points":[proj_pts]
        })

        st.download_button("Download Race Report",report.to_csv(index=False),"race_report.csv")

    else:
        st.info("Select parameters in the sidebar and click Run Simulation")

# --------------------------------------------------
# TAB 2
# --------------------------------------------------
with tab2:

    if model:

        importance=pd.DataFrame({
        "Feature":feature_cols,
        "Importance":model.feature_importances_
        }).sort_values("Importance")

        st.bar_chart(importance.set_index("Feature"))

# --------------------------------------------------
# TAB 3
# --------------------------------------------------
with tab3:

    if {"lat","lng"}.issubset(circuits_df.columns):

        st.map(circuits_df,latitude="lat",longitude="lng")