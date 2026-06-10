import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Smart BP Monitoring Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.metric-box {
    border-radius: 10px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# MODEL TRAINING
# ----------------------------
@st.cache_resource
def train_model():

    np.random.seed(42)

    data = {
        "age": np.random.randint(20, 80, 500),
        "weight": np.random.randint(40, 100, 500),
        "height": np.random.randint(150, 190, 500),
        "bmi": np.random.uniform(18, 35, 500),
        "heart_rate": np.random.randint(60, 100, 500)
    }

    df = pd.DataFrame(data)

    df["systolic"] = (
        100
        + 0.5 * df["age"]
        + 0.2 * df["weight"]
        - 0.1 * df["height"]
        + 0.3 * df["heart_rate"]
    )

    df["diastolic"] = (
        60
        + 0.3 * df["age"]
        + 0.1 * df["weight"]
        - 0.2 * df["height"]
        + 0.4 * df["heart_rate"]
    )

    X = df[["age", "weight", "height", "bmi", "heart_rate"]]

    y_sys = df["systolic"]
    y_dia = df["diastolic"]

    model_sys = RandomForestRegressor(n_estimators=100, random_state=42)
    model_dia = RandomForestRegressor(n_estimators=100, random_state=42)

    model_sys.fit(X, y_sys)
    model_dia.fit(X, y_dia)

    return model_sys, model_dia, X


model_sys, model_dia, X = train_model()

# ----------------------------
# HEADER
# ----------------------------
st.title("🩺 Smart Blood Pressure Monitoring Dashboard")

st.markdown("""
Estimate blood pressure based on personal health indicators.
""")

# ----------------------------
# INPUT SECTION
# ----------------------------
with st.form("bp_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=25
        )

        height = st.number_input(
            "Height (cm)",
            min_value=120,
            max_value=220,
            value=170
        )

        heart_rate = st.number_input(
            "Heart Rate (bpm)",
            min_value=40,
            max_value=180,
            value=72
        )

    with col2:
        weight = st.number_input(
            "Weight (kg)",
            min_value=30,
            max_value=200,
            value=70
        )

        bmi = round(weight / ((height / 100) ** 2), 2)

        st.info(f"Calculated BMI: {bmi}")

    submit = st.form_submit_button("Predict Blood Pressure")

# ----------------------------
# HISTORY
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# PREDICTION
# ----------------------------
if submit:

    input_df = pd.DataFrame([{
        "age": age,
        "weight": weight,
        "height": height,
        "bmi": bmi,
        "heart_rate": heart_rate
    }])

    systolic = round(model_sys.predict(input_df)[0], 1)
    diastolic = round(model_dia.predict(input_df)[0], 1)

    if systolic < 120:
        status = "Normal"
    elif systolic < 130:
        status = "Elevated"
    elif systolic < 140:
        status = "Stage 1"
    else:
        status = "Stage 2"

    st.subheader("Prediction Results")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Systolic BP", f"{systolic} mmHg")

    with c2:
        st.metric("Diastolic BP", f"{diastolic} mmHg")

    with c3:
        st.metric("Status", status)

    st.subheader("Health Insights")

    if status == "Normal":
        st.success("Blood pressure is within the normal range.")
    elif status == "Elevated":
        st.warning("Blood pressure is slightly elevated.")
    else:
        st.error("Blood pressure is above the recommended range.")

    st.markdown("""
### Recommendations

✅ Exercise regularly

✅ Reduce sodium intake

✅ Stay hydrated

✅ Monitor blood pressure weekly
""")

    st.session_state.history.append({
        "Systolic": systolic,
        "Diastolic": diastolic
    })

# ----------------------------
# HISTORY CHART
# ----------------------------
if len(st.session_state.history) > 0:

    st.subheader("Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df, use_container_width=True)

    st.line_chart(history_df)

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model_sys.feature_importances_
})

st.subheader("Feature Importance")

st.bar_chart(
    importance.set_index("Feature")
)
