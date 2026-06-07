"""
F1 Race Intelligence Engine — Archived Features
=================================================
These features were removed from the main app.py for a cleaner UI,
but are preserved here for future re-integration.

HOW TO RE-ENABLE:
1. Add the desired tab name back to the `tabs = st.tabs([...])` list in app.py
2. Copy the corresponding `with tabs[N]:` block below into app.py
3. Update the tab index `N` to match its position in the tabs list
4. Ensure the required imports are present in app.py (listed per feature below)

All utility functions remain in utils.py — no changes needed there.
"""


# =============================================================================
# FEATURE 1: SEASON OUTLOOK
# =============================================================================
#
# Required imports (already in utils.py):
#   from utils import season_projection
#
# Add to tabs list:
#   "📊 Season Outlook"
#
# --- Tab Code ---

def render_season_outlook(tabs, tab_index, driver_standings_df, races_df, results_df):
    """
    Championship projection tab.
    Projects final standings based on current season data.
    """
    import streamlit as st
    from utils import season_projection

    with tabs[tab_index]:
        st.header("📊 Championship Projection")

        projection = season_projection(driver_standings_df, races_df, results_df)

        if projection:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Season Year", int(projection["current_year"]))
            with col2:
                st.metric("Races Completed", projection["races_completed"])
            with col3:
                st.metric("Races Remaining",
                           projection["total_races_planned"] - projection["races_completed"])

            st.subheader("Projected Final Standings")
            display_cols = ["position", "points", "avg_points_per_race",
                            "projected_final_points", "projected_position"]
            display_df = projection["standings"][display_cols].copy()
            display_df.columns = ["Current Pos", "Current Points", "Avg/Race",
                                  "Projected Final", "Projected Rank"]
            st.dataframe(display_df.head(10), use_container_width=True)

            st.subheader("Points Projection")
            projection_chart = projection["standings"][
                ["points", "projected_final_points"]
            ].head(10).copy()
            projection_chart.columns = ["Current Points", "Projected Final"]
            st.line_chart(projection_chart)
        else:
            st.warning("Season projection data not available.")


# =============================================================================
# FEATURE 2: BATCH RACE SIMULATIONS
# =============================================================================
#
# Required imports (already in utils.py):
#   from utils import batch_race_simulation
#
# Add to tabs list:
#   "⚡ Batch Sim"
#
# --- Tab Code ---

def render_batch_simulations(tabs, tab_index, model, feature_cols,
                              driver_list, constructor_list, circuit_list,
                              weather, tyre, pit, form, risk, aggro, pressure,
                              drivers_df, constructors_df, circuits_df):
    """
    Batch race simulation tab.
    Simulates multiple drivers across identical conditions.
    """
    import streamlit as st
    from utils import batch_race_simulation

    with tabs[tab_index]:
        st.header("⚡ Batch Race Simulations")
        st.markdown("Simulate multiple drivers across identical conditions")

        col1, col2 = st.columns(2)
        with col1:
            selected_drivers = st.multiselect(
                "Select Drivers to Compare",
                driver_list.values,
                default=list(driver_list.values[:3]) if len(driver_list) >= 3 else list(driver_list.values)
            )
        with col2:
            batch_constructor = st.selectbox("Constructor", constructor_list, key="batch_constructor")

        batch_circuit = st.selectbox("Circuit", circuit_list, key="batch_circuit")
        batch_grid = st.slider("All Drivers Start From", 1, 20, 10, key="batch_grid")

        if st.button("⚡ Run Batch Simulation", use_container_width=True):
            if selected_drivers and model:
                with st.spinner("Running batch simulation..."):
                    try:
                        batch_results = batch_race_simulation(
                            model, feature_cols, selected_drivers, batch_constructor, batch_circuit,
                            [batch_grid], weather, tyre, pit, form, risk, aggro, pressure,
                            drivers_df, constructors_df, circuits_df
                        )

                        st.subheader("Batch Simulation Results")
                        st.dataframe(batch_results, use_container_width=True)

                        chart_data = batch_results[["Driver", "Final Prediction", "Projected Points"]].set_index("Driver")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.bar_chart(chart_data[["Final Prediction"]])
                        with col2:
                            st.bar_chart(chart_data[["Projected Points"]])

                        st.download_button(
                            "📥 Download Batch Results (CSV)",
                            batch_results.to_csv(index=False), "batch_simulation.csv"
                        )
                    except Exception as e:
                        st.error(f"Error in batch simulation: {e}")
            else:
                st.warning("Select at least one driver and ensure model is loaded.")


# =============================================================================
# FEATURE 3: LAP TIME ANALYSIS
# =============================================================================
#
# Required imports (already in utils.py):
#   from utils import lap_time_analysis, format_lap_time
#
# Required datasets (already loaded in load_all_datasets):
#   lap_times_df, pit_stops_df
#
# Add to tabs list:
#   "⏱️ Lap Analysis"
#
# --- Tab Code ---

def render_lap_analysis(tabs, tab_index, driver_list, circuit_list,
                         drivers_df, circuits_df, lap_times_df,
                         races_df, pit_stops_df):
    """
    Lap time analysis tab.
    Analyzes lap-by-lap performance, consistency, and tire degradation.
    """
    import streamlit as st
    from utils import lap_time_analysis, format_lap_time, get_driver_id, get_circuit_id

    with tabs[tab_index]:
        st.header("⏱️ Lap Time Analysis")
        st.markdown("Dive deep into lap-by-lap performance, consistency, and tire degradation")

        col1, col2 = st.columns(2)
        with col1:
            lap_driver = st.selectbox("Select Driver", driver_list, key="lap_driver")
        with col2:
            lap_circuit = st.selectbox("Select Circuit", circuit_list, key="lap_circuit")

        lap_driver_id = get_driver_id(lap_driver, drivers_df)
        lap_circuit_id = get_circuit_id(lap_circuit, circuits_df)

        if lap_driver_id and lap_circuit_id:
            with st.spinner("Analyzing lap times..."):
                lap_data = lap_time_analysis(
                    lap_driver_id, lap_circuit_id,
                    lap_times_df, races_df, pit_stops_df
                )

            if lap_data:
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Lap", format_lap_time(lap_data["avg_lap_ms"]))
                with col2:
                    st.metric("Best Lap", format_lap_time(lap_data["best_lap_ms"]))
                with col3:
                    st.metric("Consistency (σ)",
                              f"{lap_data['consistency_ms']/1000:.3f}s")
                with col4:
                    st.metric("Races Analyzed", lap_data["races_analyzed"])

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Laps", lap_data["total_laps"])
                with col2:
                    avg_deg = lap_data["avg_degradation_ms"]
                    st.metric("Avg Tire Degradation",
                              f"{'+' if avg_deg > 0 else ''}{avg_deg/1000:.3f}s/stint")

                # Lap progression chart
                st.subheader("Lap Time Progression (Average by Lap Number)")
                if not lap_data["lap_progression"].empty:
                    prog_df = lap_data["lap_progression"].to_frame("Avg Lap Time (ms)")
                    prog_df.index.name = "Lap"
                    st.line_chart(prog_df)

                # Stint data
                stint_df = lap_data["stint_data"]
                if not stint_df.empty:
                    st.subheader("Stint Analysis")
                    stint_summary = stint_df.groupby("stint").agg(
                        avg_laps=("laps", "mean"),
                        avg_time_ms=("avg_ms", "mean"),
                        avg_degradation=("degradation", "mean")
                    ).round(1)
                    stint_summary.columns = ["Avg Laps/Stint", "Avg Lap Time (ms)", "Avg Degradation (ms)"]
                    st.dataframe(stint_summary, use_container_width=True)

                    st.subheader("Degradation by Stint")
                    deg_chart = stint_summary[["Avg Degradation (ms)"]]
                    st.bar_chart(deg_chart)
            else:
                st.warning(f"No lap time data available for {lap_driver} at {lap_circuit}.")


# =============================================================================
# FEATURE 4: PIT STOP STRATEGY
# =============================================================================
#
# Required imports (already in utils.py):
#   from utils import pit_stop_analysis
#
# Required datasets (already loaded in load_all_datasets):
#   pit_stops_df
#
# Add to tabs list:
#   "⛽ Pit Strategy"
#
# --- Tab Code ---

def render_pit_strategy(tabs, tab_index, circuit_list, circuits_df,
                         pit_stops_df, results_df, races_df, constructors_df):
    """
    Pit stop strategy analysis tab.
    Analyzes pit stop timing, team performance, and strategy outcomes.
    """
    import streamlit as st
    from utils import pit_stop_analysis, get_circuit_id

    with tabs[tab_index]:
        st.header("⛽ Pit Stop Strategy Analysis")
        st.markdown("Pit stop timing, team performance, and strategy outcomes")

        pit_circuit = st.selectbox("Select Circuit", circuit_list, key="pit_circuit")
        pit_circuit_id = get_circuit_id(pit_circuit, circuits_df)

        if pit_circuit_id:
            with st.spinner("Analyzing pit strategies..."):
                pit_data = pit_stop_analysis(
                    pit_circuit_id, pit_stops_df, results_df,
                    races_df, constructors_df
                )

            if pit_data:
                # Key metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Stop Duration",
                              f"{pit_data['avg_stop_ms']/1000:.2f}s")
                with col2:
                    st.metric("Fastest Stop",
                              f"{pit_data['fastest_stop_ms']/1000:.2f}s")
                with col3:
                    st.metric("Total Stops Analyzed", pit_data["total_stops_analyzed"])

                # Strategy breakdown
                st.subheader("Strategy Distribution")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Stop Count Distribution**")
                    strat_df = pit_data["strategy_pct"].to_frame("Percentage (%)")
                    strat_df.index.name = "Number of Stops"
                    st.bar_chart(strat_df)

                with col2:
                    st.write("**Strategy vs Result**")
                    if not pit_data["strategy_results"].empty:
                        strat_results = pit_data["strategy_results"].copy()
                        strat_results.index.name = "Stops"
                        strat_results.columns = ["Avg Finish", "Avg Points", "Count"]
                        st.dataframe(strat_results, use_container_width=True)

                # Team performance
                st.subheader("Team Pit Crew Rankings")
                team_perf = pit_data["team_performance"].copy()
                team_perf["avg_duration_s"] = (team_perf["avg_duration_ms"] / 1000).round(2)
                team_perf["fastest_stop_s"] = (team_perf["fastest_stop_ms"] / 1000).round(2)
                display_cols = ["avg_duration_s", "fastest_stop_s", "total_stops"]
                display_df = team_perf[display_cols].copy()
                display_df.columns = ["Avg Duration (s)", "Fastest (s)", "Total Stops"]
                st.dataframe(display_df.head(15), use_container_width=True)

                # Pit window
                st.subheader("Pit Stop Timing Heatmap")
                pit_window = pit_data["pit_window"].copy()
                if not pit_window.empty:
                    pit_window = pit_window.set_index("lap")
                    pit_window.index.name = "Lap"
                    st.bar_chart(pit_window)
            else:
                st.warning(f"No pit stop data available for {pit_circuit}.")


# =============================================================================
# FEATURE 5: RACE INCIDENTS
# =============================================================================
#
# Required imports (already in utils.py):
#   from utils import retirement_analysis
#
# Required datasets (already loaded in load_all_datasets):
#   status_df
#
# Add to tabs list:
#   "🚨 Race Incidents"
#
# --- Tab Code ---

def render_race_incidents(tabs, tab_index, circuit_list, circuits_df,
                           results_df, races_df, status_df):
    """
    Race incidents and retirement analysis tab.
    Analyzes DNF causes, circuit danger ratings, and reliability trends.
    """
    import streamlit as st
    from utils import retirement_analysis, get_circuit_id

    with tabs[tab_index]:
        st.header("🚨 Race Incidents & Retirement Analysis")
        st.markdown("DNF causes, circuit danger ratings, and reliability trends")

        incident_circuit = st.selectbox("Select Circuit", circuit_list, key="incident_circuit")
        incident_circuit_id = get_circuit_id(incident_circuit, circuits_df)

        if incident_circuit_id:
            with st.spinner("Analyzing race incidents..."):
                incident_data = retirement_analysis(
                    incident_circuit_id, results_df, races_df, status_df
                )

            if incident_data:
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Entries", incident_data["total_entries"])
                with col2:
                    st.metric("DNF Count", incident_data["dnf_count"])
                with col3:
                    st.metric("DNF Rate", f"{incident_data['dnf_rate']:.1f}%")
                with col4:
                    danger = incident_data["danger_rating"]
                    emoji = "🟢" if danger < 3 else "🟡" if danger < 6 else "🔴"
                    st.metric("Danger Rating", f"{emoji} {danger}/10")

                # Cause breakdown
                st.subheader("DNF Cause Breakdown")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**By Category**")
                    cause_df = incident_data["cause_breakdown"].to_frame("Count")
                    cause_df.index.name = "Category"
                    st.bar_chart(cause_df)

                with col2:
                    st.write("**Top Specific Causes**")
                    top_causes = incident_data["top_causes"].to_frame("Count")
                    top_causes.index.name = "Cause"
                    st.dataframe(top_causes, use_container_width=True)

                # DNF rate per race
                st.subheader("DNF Rate Trend (Per Race)")
                race_dnf = incident_data["race_dnf_rates"]
                if not race_dnf.empty:
                    race_dnf_chart = race_dnf[["dnf_rate_pct"]].copy()
                    race_dnf_chart.columns = ["DNF Rate (%)"]
                    race_dnf_chart.index = range(1, len(race_dnf_chart) + 1)
                    race_dnf_chart.index.name = "Race #"
                    st.line_chart(race_dnf_chart)

                st.info(f"""
                **Circuit Analysis Summary:**
                - {incident_data['races_analyzed']} races analyzed at this circuit
                - {incident_data['dnf_rate']:.1f}% of all entries result in retirement
                - Danger rating of {danger}/10 based on collision and mechanical failure rates
                """)
            else:
                st.warning(f"No incident data available for {incident_circuit}.")
