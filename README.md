# 🏎️ Formula 1 Race Intelligence Engine

An ML-powered Formula 1 race outcome prediction, simulation, and historical analytics dashboard built with **Streamlit** and **Python**. This engine analyzes F1 race data from the modern era (2010–present) to project race results, model team/driver performance, and analyze qualifying and strategy sensitivity.

---

## 🚀 Key Features (11 Dashboard Tabs)

The application is structured into 11 dedicated analysis tabs:

1. **🏁 Race Sim**: Input driver, constructor, circuit, grid position, weather, tyre strategy, aggression, and reliability risk to run simulations. Predicts podium probabilities, expected finishing positions, DNF percentages, and runs a Monte Carlo simulation (5,000 runs) to account for race-day uncertainty.
2. **🧠 Model Brain**: Displays ML feature importances and ROC-AUC comparisons between model options (Random Forest vs. other classifiers).
3. **🗺️ Circuits Map**: An interactive map showing the global geographical distribution of all modern F1 circuits.
4. **👥 Driver Matchup**: A head-to-head comparison tool between any two drivers who competed together, showing win records, cumulative points, and race-by-race finish timelines.
5. **🏆 Circuit Records**: Historical statistics per track, including most successful drivers, top teams, average winner grid slots, and podium records.
6. **📈 Driver Trends**: Displays multi-season driver performance trends (last 2 seasons rolling averages, points per race) and driver circuit affinity (best/worst circuits).
7. **🏭 Constructor Insights**: Constructor-specific analytics tracking career wins, podiums, win rates, DNF rates, and points momentum.
8. **🎯 Strategy Lab (What-If)**: Parameter sensitivity analysis plotting how changing individual strategy variables (e.g., aggression, teammate pressure, pit crew rating, reliability risk) impacts podium probabilities.
9. **🔄 Qualifying Position Impact**: Analyzes how starting position correlates with race results across different circuits and shows track-specific overtaking difficulty (qualifying-to-finish deltas).
10. **🔥 Driver Streaks**: Track driver momentum including longest/current podium streaks, win streaks, DNF reliability streaks, and recent form.
11. **👑 Constructor Dominance**: Computes a customized dominance index and points share for constructors across different seasons to measure team supremacy.

---

## 🛠️ Technology Stack

- **Backend Logic / ML**: Python, Scikit-learn (Random Forest Classifier), NumPy, Pandas, Joblib
- **Frontend Dashboard**: Streamlit, Altair Charts, Vanilla CSS Injection
- **PDF Generation**: ReportLab (used to compile and export custom race simulation reports)
- **Data Source**: Historical F1 CSV datasets (races, drivers, results, qualifying, lap times, pit stops, status, etc.)

---

## 🔧 Installation & Setup

Follow these steps to run the F1 Race Intelligence Engine locally:

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Create and Activate a Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the ML Model
If you do not have the `f1_model.pkl` and `f1_model_features.pkl` files, run the training notebook:
```bash
jupyter notebook Model_pred.ipynb
```
*Run all cells in the notebook to preprocess the datasets, train the Random Forest model, evaluate its performance, and export the `.pkl` files.*

### 5. Launch the Dashboard
Start the Streamlit application:
```bash
streamlit run app.py
```
*The app will automatically compile and launch in your default web browser (typically at `http://localhost:8501`).*

---

## ⚡ Performance & Code Quality Enhancements

The codebase includes several notable optimizations to ensure smooth rendering and rapid calculation speeds:

- **Cached ID Lookups**: Dataframe queries for driver, constructor, and circuit IDs have been replaced with dictionary-based lookups (`O(1)` complexity), reducing dataframe operations from ~15+ per tab to just 1.
- **LRU Cache on Encoded Features**: Frequent helper functions like `encode_weather` and `encode_tyre` use Python's `@lru_cache` decorator for instantaneous dictionary access.
- **Vectorized Calculations**:
  - **Strategy Adjustment**: Pre-computes adjustments using NumPy vectors rather than serial looping.
  - **Sensitivity Analysis**: Reduced predictions from **500 down to just 20** per run by pre-calculating baseline predictions and vectorizing modifications, making the strategy tab **~25x faster**.
  - **Batch Simulations**: Vectorized inputs are sent to the model for concurrent predictions.
- **Accurate DNF Risk Modeling**: Uses `status.csv` data to check driver/constructor finish history (e.g., status codes for finished, lapped, crash, mechanical, etc.), rather than checking only if points were zero, giving precise reliability risk metrics.

---

> [!NOTE]
> All simulations and historical data are focused on the **Modern Era (2010–2024)** to ensure relevancy with contemporary F1 regulations, point structures, and driver lineups.
