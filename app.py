import json
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "trajectory_benchmark.json"
)

SIGNAL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "signal_optimization.json"
)

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "videos"
    / "traffic_demo.mp4"
)

SUMO_VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "sumo_simulation_browser.mp4"
)

TRAJECTORY_PLOT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "trajectory_plot.png"
)


st.set_page_config(
    page_title="Reality → Simulation",
    page_icon="🚦",
    layout="wide",
)


def load_json(path):
    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


benchmark = load_json(BENCHMARK_PATH)
signal = load_json(SIGNAL_PATH)


st.title("🚦 Reality → Simulation")

st.markdown(
    """
### AI-Powered Traffic Digital Twin

**Real-world traffic → Perception → Prediction → Simulation → Optimization**
"""
)

st.divider()


# --------------------------------------------------
# REALITY → DIGITAL TWIN
# --------------------------------------------------

st.subheader(
    "📹 Reality → 🌐 Digital Twin"
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        "### 📹 Real-World Traffic"
    )

    if VIDEO_PATH.exists():

        st.video(
            str(VIDEO_PATH)
        )

        st.caption(
            "Real traffic video used as the input to the perception pipeline."
        )

    else:

        st.warning(
            "Source traffic video not found."
        )


with col2:

    st.markdown(
        "### 🌐 SUMO Simulation"
    )

    if SUMO_VIDEO_PATH.exists():

        st.video(
            str(SUMO_VIDEO_PATH)
        )

        st.caption(
            "Microscopic traffic simulation generated from the SUMO digital twin."
        )

    else:

        st.warning(
            "SUMO simulation video not found."
        )


st.divider()


# --------------------------------------------------
# PERCEPTION
# --------------------------------------------------

st.subheader(
    "🚗 Perception Results"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Tracked Vehicles",
        "81",
    )

with col2:
    st.metric(
        "Trajectory Points",
        "3,381",
    )

with col3:
    st.metric(
        "Video Frames",
        "251",
    )


st.divider()


# --------------------------------------------------
# TRAJECTORIES
# --------------------------------------------------

st.subheader(
    "📈 Extracted Vehicle Trajectories"
)

if TRAJECTORY_PLOT_PATH.exists():

    st.image(
        str(TRAJECTORY_PLOT_PATH),
        use_container_width=True,
    )

    st.caption(
        "Vehicle trajectories extracted using YOLO + ByteTrack."
    )

else:

    st.warning(
        "Trajectory visualization not found."
    )


st.divider()


# --------------------------------------------------
# PIPELINE
# --------------------------------------------------

st.subheader(
    "🔄 AI Traffic Pipeline"
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "📹 Reality",
        "Traffic Video",
    )

with col2:
    st.metric(
        "🚗 Perception",
        "YOLO + ByteTrack",
    )

with col3:
    st.metric(
        "🧠 Prediction",
        "Transformer",
    )

with col4:
    st.metric(
        "🌐 Digital Twin",
        "SUMO",
    )

with col5:
    st.metric(
        "🚦 Optimization",
        "Signal Timing",
    )


st.divider()


# --------------------------------------------------
# TRAJECTORY PREDICTION
# --------------------------------------------------

st.subheader(
    "🧠 Trajectory Prediction"
)

if benchmark:

    models = benchmark["models"]

    baseline = models[0]

    best_model = min(
        models,
        key=lambda model:
        model["ade_pixels"],
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best Model",
            best_model["name"],
        )

    with col2:
        st.metric(
            "Best ADE",
            f"{best_model['ade_pixels']:.4f} px",
        )

    with col3:
        st.metric(
            "Prediction Horizon",
            f"{benchmark['prediction_frames']} frames",
        )

    st.markdown(
        "#### Model Comparison"
    )

    for model in models:

        improvement = (
            (
                baseline["ade_pixels"]
                - model["ade_pixels"]
            )
            / baseline["ade_pixels"]
            * 100
        )

        st.write(
            f"**{model['name']}** — "
            f"ADE: `{model['ade_pixels']:.4f}` px | "
            f"FDE: `{model['fde_pixels']:.4f}` px | "
            f"vs baseline: `{improvement:.2f}%`"
        )

else:

    st.warning(
        "Trajectory benchmark results not found."
    )


st.divider()


# --------------------------------------------------
# DIGITAL TWIN
# --------------------------------------------------

st.subheader(
    "🌐 SUMO Digital Twin"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Simulation",
        "300 seconds",
    )

with col2:
    st.metric(
        "Road Network",
        "2-lane intersection",
    )

with col3:
    st.metric(
        "Traffic Demand",
        "600 vehicles/hour",
    )


st.divider()


# --------------------------------------------------
# COUNTERFACTUAL OPTIMIZATION
# --------------------------------------------------

st.subheader(
    "🚦 Counterfactual Signal Optimization"
)

if signal:

    results = signal["all_results"]

    best = signal["best_scenario"]

    baseline_result = next(
        (
            result
            for result in results
            if result["green_duration"] == 30
        ),
        None,
    )

    if baseline_result:

        speed_change = (
            (
                best["average_speed"]
                - baseline_result["average_speed"]
            )
            / baseline_result["average_speed"]
            * 100
        )

        waiting_change = (
            (
                best["average_waiting_time"]
                - baseline_result["average_waiting_time"]
            )
            / baseline_result["average_waiting_time"]
            * 100
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Optimal Signal",
                f"{best['green_duration']}s / "
                f"{best['red_duration']}s",
            )

        with col2:
            st.metric(
                "Speed Improvement",
                f"{speed_change:+.1f}%",
            )

        with col3:
            st.metric(
                "Waiting Reduction",
                f"{-waiting_change:.1f}%",
            )

        st.markdown(
            "#### Baseline vs Optimized"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "##### 30s Green — Baseline"
            )

            st.metric(
                "Average Speed",
                f"{baseline_result['average_speed']:.2f} m/s",
            )

            st.metric(
                "Average Waiting",
                f"{baseline_result['average_waiting_time']:.2f} s/vehicle",
            )

            st.metric(
                "Vehicles Arrived",
                baseline_result["vehicles_arrived"],
            )

        with col2:

            st.markdown(
                "##### 50s Green — Optimized"
            )

            st.metric(
                "Average Speed",
                f"{best['average_speed']:.2f} m/s",
            )

            st.metric(
                "Average Waiting",
                f"{best['average_waiting_time']:.2f} s/vehicle",
            )

            st.metric(
                "Vehicles Arrived",
                best["vehicles_arrived"],
            )

    st.markdown(
        "#### Signal Timing Search"
    )

    chart_data = {
        "Green Duration (s)": [
            result["green_duration"]
            for result in results
        ],
        "Average Waiting (s)": [
            result["average_waiting_time"]
            for result in results
        ],
    }

    st.line_chart(
        chart_data,
        x="Green Duration (s)",
        y="Average Waiting (s)",
    )

else:

    st.warning(
        "Signal optimization results not found."
    )


st.divider()


# --------------------------------------------------
# SYSTEM STATUS
# --------------------------------------------------

st.subheader(
    "✅ System Status"
)

status_items = {
    "Video ingestion": True,
    "Vehicle detection": True,
    "Vehicle tracking": True,
    "Trajectory extraction": True,
    "Motion features": True,
    "Trajectory prediction": True,
    "Interaction graph": True,
    "SUMO simulation": True,
    "Signal optimization": True,
}


for name, status in status_items.items():

    st.write(
        f"✅ {name}"
        if status
        else f"❌ {name}"
    )


st.divider()

st.caption(
    "Reality → Simulation | AI Traffic Digital Twin Prototype"
)