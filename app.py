import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --------------------
# PAGE CONFIG
# --------------------

st.set_page_config(
    page_title="AI Public Health Dashboard",
    page_icon="🏥",
    layout="wide"
)
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2966/2966486.png",
    width=80
)
st.sidebar.title("🏥 NPHIS")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "AI Prediction",
        "Dataset",
        "Map",
        "About"
    ]
)
st.sidebar.markdown("---")

st.sidebar.caption(
    "National Public Health Intelligence System"
)

# --------------------
# LOAD DATA
# --------------------

data = pd.read_csv("health_data.csv")
state_locations = {
    "Kerala": [10.8505, 76.2711],
    "Tamil Nadu": [11.1271, 78.6569],
    "Karnataka": [15.3173, 75.7139],
    "Maharashtra": [19.7515, 75.7139],
    "Delhi": [28.7041, 77.1025],
    "Gujarat": [22.2587, 71.1924],
    "Rajasthan": [27.0238, 74.2179],
    "Uttar Pradesh": [26.8467, 80.9462],
    "West Bengal": [22.9868, 87.8550],
    "Punjab": [31.1471, 75.3412],
    "Haryana": [29.0588, 76.0856],
    "Andhra Pradesh": [15.9129, 79.7400],
    "Telangana": [18.1124, 79.0193],
    "Madhya Pradesh": [22.9734, 78.6569],
    "Odisha": [20.9517, 85.0985]
}

# --------------------
# CREATE RISK LEVEL
# --------------------

def classify_risk(cases):
    if cases < 150:
        return "Low"
    elif cases <= 300:
        return "Medium"
    else:
        return "High"

data["Risk_Level"] = data["Cases"].apply(classify_risk)

# --------------------
# MACHINE LEARNING MODEL
# --------------------

encoder = LabelEncoder()

data["State_Encoded"] = encoder.fit_transform(data["State"])

X = data[["State_Encoded", "Month", "Cases"]]

y = data["Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# --------------------
# DASHBOARD PAGE
# --------------------

if page == "Dashboard":
    st.markdown("""
<h5 style='color:gray'>
Ministry of Health & Family Welfare
</h5>
""",
unsafe_allow_html=True)

    st.markdown("""
<div style="
background: linear-gradient(90deg,#0E4D92,#1B6CA8);
padding:20px;
border-radius:12px;
color:white;
">
<h1>🏥 National Public Health Intelligence System</h1>
<p>AI-powered disease surveillance and outbreak prediction platform</p>
</div>
""", unsafe_allow_html=True)
    from datetime import datetime

    st.caption(
    f"Last Updated: {datetime.now().strftime('%d %B %Y %H:%M')}"
)

    total_cases = data["Cases"].sum()

    high_risk = len(data[data["Risk_Level"] == "High"])

    medium_risk = len(data[data["Risk_Level"] == "Medium"])

    low_risk = len(data[data["Risk_Level"] == "Low"])

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
      st.markdown(f"""
    <div style="background:#0E4D92;padding:20px;height:140px;border-radius:10px;color:white;text-align:center">
    <div style="font-size:18px;font-weight:bold;">
Total Cases
</div>

<div style="font-size:38px;font-weight:bold;margin-top:15px;">
{total_cases}
</div>
    </div>
    """, unsafe_allow_html=True)

    with c2:
      st.markdown(f"""
    <div style="background:#C62828;padding:20px;height:140px;border-radius:10px;color:white;text-align:center">
    <div style="font-size:18px;font-weight:bold;">
High Risk
</div>

<div style="font-size:38px;font-weight:bold;margin-top:15px;">
{high_risk}
</div>
    </div>
    """, unsafe_allow_html=True)

    with c3:
      st.markdown(f"""
    <div style="background:#F9A825;padding:20px;height:140px;border-radius:10px;color:white;text-align:center">
    <div style="font-size:18px;font-weight:bold;">
Medium Risk
</div>

<div style="font-size:38px;font-weight:bold;margin-top:15px;">
{medium_risk}
</div>
    </div>
    """, unsafe_allow_html=True)

    with c4:
      st.markdown(f"""
    <div style="background:#2E7D32;padding:20px;height:140px;border-radius:10px;color:white;text-align:center">
    <div style="font-size:18px;font-weight:bold;">
Low Risk
</div>

<div style="font-size:38px;font-weight:bold;margin-top:15px;">
{low_risk}
</div>
    </div>
    """, unsafe_allow_html=True)
      
    with c5:
      st.markdown(f"""
    <div style="
    background:#6A1B9A;
    padding:20px;
    height:140px;
    border-radius:10px;
    color:white;
    text-align:center;
    ">

    <div style="
    font-size:18px;
    font-weight:bold;
    ">
    Accuracy
    </div>

    <div style="
    font-size:38px;
    font-weight:bold;
    margin-top:15px;
    ">
    {accuracy*100:.1f}%
    </div>

    </div>
    """, unsafe_allow_html=True)

   
    high_percentage = round((high_risk / len(data)) * 100, 1)

    st.metric(
    "High Risk Percentage",
    f"{high_percentage}%"
)
    st.success(
    "🟢 Surveillance Network Status: Operational Across All Reporting Regions"
)
    if high_risk > 8:
        st.error("🔴 National Health Status: Elevated Risk")
    elif high_risk > 4:
        st.warning("🟡 National Health Status: Moderate Risk")
    else:
        st.success("🟢 National Health Status: Stable")

    st.info(
    f"AI Model trained on {len(data)} surveillance records with {accuracy*100:.1f}% accuracy."
      )
    st.error(
    "🚨 National Health Alert: Continuous monitoring active across all regions."
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("📊 Cases by State")

        state_cases = data.groupby("State")["Cases"].sum()

        st.bar_chart(state_cases)

    with right:
        st.subheader("📈 Monthly Trend")

        month_cases = data.groupby("Month")["Cases"].sum()

        st.line_chart(month_cases)

    st.divider()

    st.subheader("⚠ Risk Distribution")

    risk_counts = data["Risk_Level"].value_counts()

    st.bar_chart(risk_counts)
    st.divider()

    st.subheader("🚨 Situation Summary")

    highest_state = state_cases.idxmax()

    highest_cases = state_cases.max()

    st.warning(f"""
Situation Assessment:

• Highest disease burden observed in {highest_state}

• Total reported cases: {highest_cases}

• Enhanced surveillance recommended

• No nationwide outbreak emergency declared
""")
    csv = data.to_csv(index=False)

    st.download_button(
    "📥 Download Surveillance Report",
    csv,
    "surveillance_report.csv",
    "text/csv"
)
    st.divider()

    st.subheader("🗺 Regional Surveillance Overview")

    state_summary = data.groupby("State").agg({
    "Cases": "sum"
}).reset_index()

    state_summary["Risk Level"] = state_summary["Cases"].apply(classify_risk)

    st.dataframe(
       state_summary,
       use_container_width=True
)
    st.markdown("---")

    st.markdown(
"""
<center>

National Public Health Intelligence System (NPHIS)

AI-Powered Disease Surveillance Platform

Developed using Streamlit, Pandas and Machine Learning

</center>
""",
      unsafe_allow_html=True
)
    st.caption(
    "Disclaimer: This dashboard is developed for academic and demonstration purposes only."
)
# --------------------
# AI PAGE
# --------------------

elif page == "AI Prediction":

    st.title("🤖 AI Outbreak Risk Assessment")

    state = st.selectbox(
        "Select State",
        data["State"].unique()
    )

    month = st.slider(
        "Month",
        1,
        12,
        6
    )

    cases = st.number_input(
        "Number of Cases",
        min_value=0,
        value=100
    )

    if st.button("Predict Risk"):

        state_encoded = encoder.transform([state])[0]

        input_data = pd.DataFrame({
            "State_Encoded": [state_encoded],
            "Month": [month],
            "Cases": [cases]
        })

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data).max() * 100

        st.metric(
    "Prediction Confidence",
    f"{probability:.1f}%"
)

        if prediction == "Low":

          st.success("🟢 LOW RISK")

          st.info("""
    AI Recommendation:

    • Continue routine surveillance
    • Maintain reporting mechanisms
    • Monitor trends monthly
    """)

        elif prediction == "Medium":

           st.warning("🟡 MEDIUM RISK")

           st.info("""
    AI Recommendation:

    • Increase monitoring frequency
    • Prepare district response teams
    • Monitor case growth weekly
    """)

        else:

          st.error("🔴 HIGH RISK")

          st.info("""
    AI Recommendation:

    • Deploy emergency surveillance teams
    • Increase testing capacity
    • Alert public health authorities
    • Initiate outbreak investigation
    """)

# --------------------
# DATASET PAGE
# --------------------

elif page == "Dataset":

    st.title("📋 Health Dataset")
    

    st.subheader("Dataset Statistics")

    c1, c2, c3 = st.columns(3)

    c1.metric("Records", len(data))
    c2.metric("States", data["State"].nunique())
    c3.metric("Months", data["Month"].nunique())

    search_state = st.selectbox(
        "Filter by State",
        ["All"] + list(data["State"].unique())
    )

    if search_state != "All":
        filtered_data = data[data["State"] == search_state]
    else:
        filtered_data = data

    st.divider()

    st.dataframe(
        filtered_data,
        use_container_width=True,
        height=500
    )
    csv = filtered_data.to_csv(index=False)

    st.download_button(
    "📥 Download Dataset",
    csv,
    "health_dataset.csv",
    "text/csv"
)
  
elif page == "Map":

    st.title("🗺 National Disease Surveillance Map")

    state_summary = data.groupby("State")["Cases"].sum().reset_index()

    state_summary["Latitude"] = state_summary["State"].map(
        lambda x: state_locations[x][0]
    )

    state_summary["Longitude"] = state_summary["State"].map(
        lambda x: state_locations[x][1]
    )

    fig = px.scatter_geo(
        state_summary,
        lat="Latitude",
        lon="Longitude",
        size="Cases",
        hover_name="State",
        hover_data=["Cases"],
        scope="asia",
        title="State-wise Disease Activity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        state_summary,
        use_container_width=True
    )
# --------------------
# ABOUT PAGE
# --------------------

elif page == "About":
    st.markdown("""
<div style="
background:#0E4D92;
padding:20px;
border-radius:10px;
color:white;
">
<h1>🏥 About NPHIS</h1>
<p>National Public Health Intelligence System</p>
</div>
""", unsafe_allow_html=True)
    st.markdown("""
### 🏥 National Public Health Intelligence System (NPHIS)

The National Public Health Intelligence System is an AI-powered
disease surveillance platform designed to monitor disease trends,
analyze outbreak risks, and support public health decision-making.

### 🔧 Technologies

- Streamlit
- Pandas
- Scikit-Learn
- Random Forest Classifier

### 🚀 Features

- Disease Monitoring Dashboard
- AI Risk Prediction
- Trend Analysis
- Risk Distribution Analytics
- Interactive Data Exploration

### 🎯 Objective

To support health agencies in identifying potential outbreaks
and improving response planning through AI-driven insights.
""")

    csv = data.to_csv(index=False)

    st.download_button(
        label="📥 Download Health Report",
        data=csv,
        file_name="health_report.csv",
        mime="text/csv"
    )