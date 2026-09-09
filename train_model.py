import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("Student Social Media And Mental Health Impact.csv")

print("Original shape:", df.shape)


# ==========================================
# 2. DATA CLEANING
# ==========================================

# Remove duplicate rows
df = df.drop_duplicates()

# Fix negative physical activity values
df["Physical_Activity_Hours"] = df["Physical_Activity_Hours"].clip(lower=0)

print("After cleaning:", df.shape)


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

# Group countries
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

df["Grouped_country"] = df["Country"].apply(
    lambda x: x if x in top_countries else "Other"
)


# ==========================================
# 4. FEATURES AND TARGET
# ==========================================

features = [
    "Age",
    "Gender",
    "Country",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Study_Hours",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
    "Stress_Level",
    "Grouped_country"
]

target = "Mental_Health_Score"

X = df[features]
y = df[target]


# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# ==========================================
# 6. FEATURE GROUPS
# ==========================================

skewed_features = [
    "Study_Hours"
]

numeric_features = [
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night"
]

ordinal_features = [
    "Stress_Level"
]

nominal_features = [
    "Gender",
    "Country",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Grouped_country"
]


# ==========================================
# 7. PREPROCESSING
# ==========================================

skewed_pipeline = Pipeline([
    (
        "log",
        FunctionTransformer(
            np.log1p,
            feature_names_out="one-to-one"
        )
    ),
    ("scaler", StandardScaler())
])


ordinal_pipeline = Pipeline([
    (
        "encoder",
        OrdinalEncoder(
            categories=[
                ["Low", "Medium", "High", "Very High"]
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    )
])


preprocessor = ColumnTransformer(
    transformers=[
        ("skewed", skewed_pipeline, skewed_features),
        ("numeric", StandardScaler(), numeric_features),
        ("ordinal", ordinal_pipeline, ordinal_features),
        (
            "nominal",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            nominal_features
        )
    ]
)


# ==========================================
# 8. RANDOM FOREST MODEL
# ==========================================

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "random forest",
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    )
])


# ==========================================
# 9. TRAIN MODEL
# ==========================================

print("\nTraining model...")

model_pipeline.fit(X_train, y_train)

print("Training completed.")


# ==========================================
# 10. EVALUATION
# ==========================================

y_pred = model_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance")
print("-------------------------")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")


# ==========================================
# 11. SAVE MODEL
# ==========================================

joblib.dump(
    model_pipeline,
    "Mental_Health_Model.pkl"
)

print("\nModel saved as:")
print("Mental_Health_Model.pkl")