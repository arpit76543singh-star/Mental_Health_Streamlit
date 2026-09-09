import streamlit as st

from model import predict_mental_health


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Mental Health Score Predictor",
    page_icon="🧠",
    layout="centered"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .score-title {
        text-align: center;
        font-size: 20px;
        font-weight: 600;
    }

    .score-value {
        text-align: center;
        font-size: 55px;
        font-weight: 700;
        margin: 5px 0 15px 0;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🧠 Mental Health Score Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based student mental health score prediction'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter the student's lifestyle, academic and social-media information "
    "to generate a predicted mental health score."
)


# ==========================================
# STUDENT INFORMATION
# ==========================================

st.markdown(
    '<div class="section-title">👤 Student Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=20,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    country = st.selectbox(
        "Country",
        [
            "India",
            "USA",
            "Canada",
            "Australia",
            "UK",
            "Germany",
            "Mexico",
            "Turkey",
            "France",
            "Other"
        ]
    )

    academic_level = st.selectbox(
        "Academic Level",
        [
            "Undergraduate",
            "Graduate",
            "High School"
        ]
    )

    most_used_platform = st.selectbox(
        "Most Used Platform",
        [
            "Facebook",
            "LinkedIn",
            "Instagram",
            "Snapchat",
            "Twitter",
            "YouTube",
            "TikTok",
            "LINE",
            "KakaoTalk",
            "VKontakte",
            "WhatsApp",
            "WeChat"
        ]
    )


with col2:

    purpose_of_use = st.selectbox(
        "Purpose of Social Media Use",
        [
            "Networking",
            "Education",
            "Entertainment",
            "News"
        ]
    )

    avg_daily_usage_hours = st.number_input(
        "Average Daily Social Media Usage (hours)",
        min_value=0.0,
        max_value=24.0,
        value=3.0,
        step=0.1
    )

    daily_unlocks = st.number_input(
        "Daily Phone Unlocks",
        min_value=0,
        max_value=500,
        value=50,
        step=1
    )

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=4.0,
        step=0.1
    )

    physical_activity_hours = st.number_input(
        "Physical Activity Hours per Day",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

    sleep_hours_per_night = st.number_input(
        "Sleep Hours per Night",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.1
    )

    stress_level = st.selectbox(
        "Stress Level",
        [
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )


# ==========================================
# PREDICTION BUTTON
# ==========================================

st.divider()

predict_button = st.button(
    "🔮 Predict Mental Health Score",
    type="primary",
    use_container_width=True
)


# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    try:

        prediction = predict_mental_health(
            age=age,
            gender=gender,
            country=country,
            academic_level=academic_level,
            most_used_platform=most_used_platform,
            purpose_of_use=purpose_of_use,
            avg_daily_usage_hours=avg_daily_usage_hours,
            daily_unlocks=daily_unlocks,
            study_hours=study_hours,
            physical_activity_hours=physical_activity_hours,
            sleep_hours_per_night=sleep_hours_per_night,
            stress_level=stress_level
        )

        # ==================================
        # SCORE
        # ==================================

        st.success("Prediction completed successfully!")

        st.markdown(
            '<div class="score-title">'
            'Predicted Mental Health Score'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="score-value">{prediction:.2f} / 10</div>',
            unsafe_allow_html=True
        )


        # ==================================
        # PROGRESS BAR
        # ==================================

        st.progress(
            min(max(prediction / 10, 0.0), 1.0)
        )


        # ==================================
        # SCORE INTERPRETATION
        # ==================================

        if prediction < 4:

            st.warning(
                "Lower predicted score"
            )

            st.write(
                "The model produced a relatively lower score "
                "on the 0–10 prediction scale."
            )

        elif prediction < 7:

            st.info(
                "Moderate predicted score"
            )

            st.write(
                "The model produced a moderate score "
                "on the 0–10 prediction scale."
            )

        else:

            st.success(
                "Higher predicted score"
            )

            st.write(
                "The model produced a relatively higher score "
                "on the 0–10 prediction scale."
            )


        # ==================================
        # SCALE
        # ==================================

        st.markdown("### 📊 Score Scale")

        st.write(
            "**0–3.99:** Lower predicted score"
        )

        st.write(
            "**4–6.99:** Moderate predicted score"
        )

        st.write(
            "**7–10:** Higher predicted score"
        )


        # ==================================
        # INPUT SUMMARY
        # ==================================

        with st.expander("🔍 View Input Summary"):

            st.write(f"**Age:** {age}")
            st.write(f"**Gender:** {gender}")
            st.write(f"**Country:** {country}")
            st.write(f"**Academic Level:** {academic_level}")
            st.write(
                f"**Most Used Platform:** {most_used_platform}"
            )
            st.write(
                f"**Purpose:** {purpose_of_use}"
            )
            st.write(
                f"**Daily Social Media Usage:** "
                f"{avg_daily_usage_hours} hours"
            )
            st.write(
                f"**Daily Unlocks:** {daily_unlocks}"
            )
            st.write(
                f"**Study Hours:** {study_hours} hours"
            )
            st.write(
                f"**Physical Activity:** "
                f"{physical_activity_hours} hours"
            )
            st.write(
                f"**Sleep:** "
                f"{sleep_hours_per_night} hours"
            )
            st.write(
                f"**Stress Level:** {stress_level}"
            )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# ==========================================
# DISCLAIMER
# ==========================================

st.divider()

st.caption(
    "⚠️ Educational project only. "
    "The predicted score is generated by a machine learning model "
    "and should not be interpreted as a medical diagnosis, "
    "clinical assessment, or professional mental-health evaluation."
)