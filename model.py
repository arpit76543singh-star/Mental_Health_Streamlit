import joblib
import pandas as pd


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("Mental_Health_Model.pkl")


# ==========================================
# TOP COUNTRIES
# ==========================================

top_countries = [
    "India",
    "USA",
    "Canada",
    "Australia",
    "UK",
    "Germany",
    "Mexico",
    "Turkey",
    "France"
]


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_mental_health(
    age,
    gender,
    country,
    academic_level,
    most_used_platform,
    purpose_of_use,
    avg_daily_usage_hours,
    daily_unlocks,
    study_hours,
    physical_activity_hours,
    sleep_hours_per_night,
    stress_level
):

    # Group country
    grouped_country = (
        country
        if country in top_countries
        else "Other"
    )

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Country": country,
        "Academic_Level": academic_level,
        "Most_Used_Platform": most_used_platform,
        "Purpose_Of_Use": purpose_of_use,
        "Avg_Daily_Usage_Hours": avg_daily_usage_hours,
        "Daily_Unlocks": daily_unlocks,
        "Study_Hours": study_hours,
        "Physical_Activity_Hours": physical_activity_hours,
        "Sleep_Hours_Per_Night": sleep_hours_per_night,
        "Stress_Level": stress_level,
        "Grouped_country": grouped_country
    }])
# Prediction
    print("\nINPUT SENT TO MODEL:")
    print(input_data)

    prediction = model.predict(input_data)[0]

    print("PREDICTION:", prediction)

    return round(float(prediction), 2)