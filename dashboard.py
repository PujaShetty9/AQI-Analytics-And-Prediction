import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import shap
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION AND DATA LOADING
# --------------------------------------------------

st.set_page_config(
    page_title="AQI Analytics Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AQI Analytics and Prediction Dashboard")
st.caption("Historical air-quality analysis, visualization, and model prediction")

DATA_FILES = [
    "dataset/cleaned/aqi_feature_engineered.csv",
    "dataset/raw/AQI_Dataset.csv",
    "dataset/raw/Geolocation_Cleaned.csv",
]


@st.cache_data
def load_data():
    for filename in DATA_FILES:
        path = Path(filename)
        if path.exists():
            return pd.read_csv(path), filename
    return None, None


df, data_filename = load_data()

if df is None:
    st.error(
        "Dataset not found. Place your existing AQI CSV file "
        "in the same folder as dashboard.py. "
        "Expected one of: " + ", ".join(DATA_FILES)
    )
    st.stop()

df.columns = [str(c).strip() for c in df.columns]


def find_col(possible_names):
    """Find a column while ignoring case and spaces."""
    lookup = {c.lower().replace(" ", "_"): c for c in df.columns}

    for name in possible_names:
        key = name.lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]

    return None


aqi_col = find_col(["aqi"])
date_col = find_col(["last_update", "date", "datetime", "timestamp"])
city_col = find_col(["city"])
state_col = find_col(["state"])
station_col = find_col(["station"])
pollutant_col = find_col(["pollutant_id", "pollutant"])
category_col = find_col(["aqi_bucket", "aqi_category", "category"])
temperature_col = find_col(["temperature_c", "temperature"])
humidity_col = find_col(["humidity_percent", "humidity"])
wind_col = find_col(["wind_speed_kmh", "wind_speed"])
pollutant_avg_col = find_col(["pollutant_avg", "avg_pollutant"])
pollutant_min_col = find_col(["pollutant_min"])
pollutant_max_col = find_col(["pollutant_max"])

if aqi_col is None:
    st.error("The dataset does not contain an AQI column.")
    st.write("Columns found:", df.columns.tolist())
    st.stop()


numeric_candidates = [
    aqi_col,
    temperature_col,
    humidity_col,
    wind_col,
    pollutant_avg_col,
    pollutant_min_col,
    pollutant_max_col
]

for col in numeric_candidates:
    if col:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

valid_aqi = df.dropna(subset=[aqi_col]).copy()


# --------------------------------------------------
# 2. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

filtered = valid_aqi.copy()

if state_col:
    states = sorted(filtered[state_col].dropna().astype(str).unique())

    chosen_states = st.sidebar.multiselect(
        "Select state",
        states,
        default=[]
    )

    if chosen_states:
        filtered = filtered[
            filtered[state_col].astype(str).isin(chosen_states)
        ]


if city_col:
    cities = sorted(filtered[city_col].dropna().astype(str).unique())

    chosen_cities = st.sidebar.multiselect(
        "Select city",
        cities,
        default=[]
    )

    if chosen_cities:
        filtered = filtered[
            filtered[city_col].astype(str).isin(chosen_cities)
        ]


if pollutant_col:
    pollutants = sorted(
        filtered[pollutant_col].dropna().astype(str).unique()
    )

    chosen_pollutants = st.sidebar.multiselect(
        "Select pollutant",
        pollutants,
        default=[]
    )

    if chosen_pollutants:
        filtered = filtered[
            filtered[pollutant_col].astype(str).isin(chosen_pollutants)
        ]


st.sidebar.caption(f"Dataset: {data_filename}")
st.sidebar.caption(f"Filtered records: {len(filtered):,}")

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()


# --------------------------------------------------
# 3. OVERVIEW
# --------------------------------------------------

st.header("📊 AQI Overview")

mean_aqi = filtered[aqi_col].mean()
max_aqi = filtered[aqi_col].max()
median_aqi = filtered[aqi_col].median()
record_count = len(filtered)

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total AQI Records", f"{record_count:,}")
k2.metric("Average AQI", f"{mean_aqi:.1f}")
k3.metric("Maximum AQI", f"{max_aqi:.1f}")
k4.metric("Median AQI", f"{median_aqi:.1f}")

st.caption(
    "These indicators summarize the selected historical records. "
    "They are descriptive statistics, not model evaluation metrics."
)


# --------------------------------------------------
# 4. AQI CATEGORY DISTRIBUTION
# --------------------------------------------------

st.subheader("AQI Category Distribution")

if category_col:

    category_counts = (
        filtered[category_col]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Records")
    )

    fig = px.pie(
        category_counts,
        names="Category",
        values="Records",
        hole=0.45,
        title="Distribution of AQI Categories"
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info(
        "No AQI category column was found. "
        "The category chart requires a category field in the dataset."
    )


# --------------------------------------------------
# 5. AQI DISTRIBUTION
# --------------------------------------------------

st.subheader("AQI Value Distribution")

fig = px.histogram(
    filtered,
    x=aqi_col,
    nbins=30,
    marginal="box",
    title="Distribution of AQI Values",
    labels={aqi_col: "AQI"}
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 6. LOCATION ANALYSIS
# --------------------------------------------------

st.header("📍 Location-Based Analysis")

if city_col:

    city_summary = (
        filtered.groupby(city_col, dropna=True)[aqi_col]
        .agg(["mean", "median", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": "Average AQI",
            "median": "Median AQI",
            "max": "Maximum AQI",
            "count": "Record Count"
        })
        .sort_values("Average AQI", ascending=False)
    )

    fig = px.bar(
        city_summary.head(20),
        x=city_col,
        y="Average AQI",
        color="Average AQI",
        title="Top 20 Cities by Average AQI",
        hover_data=["Maximum AQI", "Record Count"]
    )

    fig.update_layout(xaxis_tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("City-Level AQI Summary")

    st.dataframe(
        city_summary,
        use_container_width=True
    )

else:
    st.info("No city column was found in the dataset.")


if state_col:

    state_summary = (
        filtered.groupby(state_col, dropna=True)[aqi_col]
        .mean()
        .reset_index(name="Average AQI")
        .sort_values("Average AQI", ascending=False)
    )

    fig = px.bar(
        state_summary,
        x=state_col,
        y="Average AQI",
        title="Average AQI by State",
        color="Average AQI"
    )

    fig.update_layout(xaxis_tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 7. TIME-BASED ANALYSIS
# --------------------------------------------------

st.header("📅 Time-Based AQI Analysis")

if date_col and filtered[date_col].notna().any():

    time_df = filtered.dropna(subset=[date_col]).copy()

    time_df["Date"] = time_df[date_col].dt.date

    daily_aqi = (
        time_df.groupby("Date")[aqi_col]
        .mean()
        .reset_index(name="Average AQI")
    )

    fig = px.line(
        daily_aqi,
        x="Date",
        y="Average AQI",
        markers=True,
        title="Daily Average AQI Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

    if daily_aqi["Date"].nunique() <= 1:

        st.info(
            "The selected records contain only one distinct date. "
            "A meaningful time trend requires observations across multiple dates."
        )

else:

    st.info(
        "A usable date/time column was not found, "
        "so time-based analysis is unavailable."
    )


# --------------------------------------------------
# 8. POLLUTANT ANALYSIS
# --------------------------------------------------

st.header("🧪 Pollutant Analysis")

if pollutant_col:

    pollutant_summary = (
        filtered.groupby(pollutant_col, dropna=True)[aqi_col]
        .agg(["mean", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": "Average AQI",
            "max": "Maximum AQI",
            "count": "Records"
        })
        .sort_values("Average AQI", ascending=False)
    )

    fig = px.bar(
        pollutant_summary,
        x=pollutant_col,
        y="Average AQI",
        color="Average AQI",
        title="Average AQI by Pollutant"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        pollutant_summary,
        use_container_width=True
    )

else:
    st.info("No pollutant identifier column was found.")


if pollutant_avg_col:

    fig = px.histogram(
        filtered.dropna(subset=[pollutant_avg_col]),
        x=pollutant_avg_col,
        color=pollutant_col if pollutant_col else None,
        nbins=30,
        title="Distribution of Average Pollutant Measurements"
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 9. ENVIRONMENTAL FACTOR ANALYSIS
# --------------------------------------------------

st.header("🌡️ Environmental Factor Analysis")

environment_cols = [
    c for c in [
        temperature_col,
        humidity_col,
        wind_col
    ]
    if c is not None
]

if environment_cols:

    for col in environment_cols:

        chart_df = filtered.dropna(
            subset=[col, aqi_col]
        )

        if not chart_df.empty:

            fig = px.scatter(
                chart_df,
                x=col,
                y=aqi_col,
                color=pollutant_col if pollutant_col else None,
                opacity=0.65,
                title=f"AQI vs {col}",
                trendline="ols" if len(chart_df) >= 3 else None
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    corr_cols = [aqi_col] + environment_cols

    if pollutant_avg_col:
        corr_cols.append(pollutant_avg_col)

    corr = filtered[corr_cols].corr(
        numeric_only=True
    )

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Between AQI and Numeric Factors"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.info(
        "Temperature, humidity, or wind columns were not found. "
        "Check the dataset's column names."
    )


# --------------------------------------------------
# 10. DATA QUALITY
# --------------------------------------------------

st.header("🧹 Data Quality")

quality = pd.DataFrame({
    "Column": filtered.columns,
    "Missing Values": filtered.isna().sum().values,
    "Missing Percentage": (
        filtered.isna().mean().values * 100
    ).round(2),
    "Data Type": filtered.dtypes.astype(str).values
})

q1, q2, q3 = st.columns(3)

q1.metric(
    "Rows",
    f"{len(filtered):,}"
)

q2.metric(
    "Columns",
    f"{len(filtered.columns):,}"
)

q3.metric(
    "Total Missing Cells",
    f"{int(filtered.isna().sum().sum()):,}"
)

st.subheader("Missing Values by Column")

st.dataframe(
    quality.sort_values(
        "Missing Percentage",
        ascending=False
    ),
    use_container_width=True
)


# --------------------------------------------------
# 11. MODEL PREDICTION USING TRAINED MODEL
# --------------------------------------------------

st.header("🤖 AQI Prediction")

st.write(
    "Enter a new observation to generate a prediction using "
    "the trained AQI classification model."
)

with st.form("aqi_prediction_form"):

    left, right = st.columns(2)

    with left:

        pollutant_min = st.number_input(
            "Pollutant Minimum",
            min_value=0.0,
            value=10.0
        )

        pollutant_max = st.number_input(
            "Pollutant Maximum",
            min_value=0.0,
            value=50.0
        )

        pollutant_avg = st.number_input(
            "Pollutant Average",
            min_value=0.0,
            value=30.0
        )

        temperature_c = st.number_input(
            "Temperature (°C)",
            value=30.0
        )

    with right:

        humidity_percent = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

        wind_speed_kmh = st.number_input(
            "Wind Speed (km/h)",
            min_value=0.0,
            value=10.0
        )

        elevation = st.number_input(
            "Digital Elevation Model",
            value=10.0
        )

        pollutant_id = st.selectbox(
            "Pollutant ID",
            [
                "PM2.5",
                "PM10",
                "NO2",
                "SO2",
                "CO",
                "O3",
                "NH3"
            ]
        )

    predict_button = st.form_submit_button(
        "Predict AQI"
    )


if predict_button:

    if not pollutant_min <= pollutant_avg <= pollutant_max:

        st.error(
            "Check pollutant values: minimum must be less than "
            "or equal to average, and average less than or equal to maximum."
        )

    else:

        try:

            # Load the trained model
            trained_model = joblib.load(
                "trained_model.pkl"
            )

            # Create input data using the same feature names
            # used during model training
            input_data = pd.DataFrame([{
                "pollutant_min": pollutant_min,
                "pollutant_max": pollutant_max,
                "pollutant_avg": pollutant_avg,
                "temperature_c": temperature_c,
                "humidity_percent": humidity_percent,
                "wind_speed_kmh": wind_speed_kmh,
                "Digital Elevation Model": elevation,
                "pollutant_id": pollutant_id
            }])

            # Generate prediction
            prediction = trained_model.predict(
                input_data
            )[0]

            # Generate prediction probabilities
            probabilities = trained_model.predict_proba(
                input_data
            )[0]

            # Model classes:
            # 0 = Good
            # 1 = Not Good
            if prediction == 0:
                category = "Good"
            else:
                category = "Not Good"

            confidence = probabilities[
                int(prediction)
            ]

            st.success(
                "Prediction generated successfully."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Category",
                    category
                )

            with col2:

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2%}"
                )

            with st.expander(
                "Prediction Details"
            ):

                st.write(
                    "Model prediction:",
                    int(prediction)
                )

                st.write(
                    "Input values:"
                )

                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )

                st.write(
                    "Prediction probabilities:"
                )

                probability_df = pd.DataFrame({
                    "Class": [
                        "Good",
                        "Not Good"
                    ],
                    "Probability": [
                        probabilities[0],
                        probabilities[1]
                    ]
                })

                probability_df["Probability"] = (
                    probability_df["Probability"]
                    .map(lambda x: f"{x:.2%}")
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

        except FileNotFoundError:

            st.error(
                "trained_model.pkl was not found. "
                "Make sure the trained model is present in the repository."
            )

        except Exception as e:

            st.error(
                "Prediction could not be generated."
            )

            st.code(
                str(e)
            )


# --------------------------------------------------
# 12. MODEL PERFORMANCE
# --------------------------------------------------

st.header("📈 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Accuracy",
    "91.32%"
)

col2.metric(
    "ROC-AUC",
    "92.36%"
)

col3.metric(
    "Test Samples",
    "795"
)


st.subheader("Classification Report")

report_df = pd.DataFrame({
    "Class": [
        "Good",
        "Polluted",
        "Macro Average",
        "Weighted Average"
    ],
    "Precision": [
        0.90,
        0.97,
        0.94,
        0.92
    ],
    "Recall": [
        0.99,
        0.66,
        0.83,
        0.91
    ],
    "F1-Score": [
        0.95,
        0.79,
        0.87,
        0.91
    ],
    "Support": [
        603,
        192,
        795,
        795
    ]
})

st.dataframe(
    report_df,
    use_container_width=True,
    hide_index=True
)


st.subheader("Performance by Class")

chart_df = report_df[
    report_df["Class"].isin(
        ["Good", "Polluted"]
    )
].melt(
    id_vars="Class",
    value_vars=[
        "Precision",
        "Recall",
        "F1-Score"
    ],
    var_name="Metric",
    value_name="Score"
)

fig = px.bar(
    chart_df,
    x="Class",
    y="Score",
    color="Metric",
    barmode="group",
    text_auto=".2f",
    title="Precision, Recall and F1-Score by AQI Class",
    range_y=[0, 1.05]
)

fig.update_layout(
    yaxis_title="Score",
    xaxis_title="AQI Class",
    legend_title="Metric"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


st.info(
    "The model achieves 91.32% accuracy and 92.36% ROC-AUC. "
    "It identifies the Good class with 99% recall, while "
    "the Polluted class has 66% recall, meaning some polluted "
    "instances are missed."
)


# --------------------------------------------------
# 13. SHAP EXPLAINABILITY
# --------------------------------------------------

st.header("🔍 SHAP Explainability")

st.write(
    "SHAP shows how the model's input features contribute to its "
    "predictions. Larger absolute SHAP values indicate stronger "
    "influence on the model output."
)

try:

    # Load the existing trained pipeline
    trained_model = joblib.load(
        "trained_model.pkl"
    )

    # Features used by the trained model
    model_features = [
        "pollutant_min",
        "pollutant_max",
        "pollutant_avg",
        "temperature_c",
        "humidity_percent",
        "wind_speed_kmh",
        "Digital Elevation Model",
        "pollutant_id"
    ]

    # Check that required columns exist
    missing_features = [
        feature
        for feature in model_features
        if feature not in filtered.columns
    ]

    if missing_features:

        st.warning(
            "SHAP analysis cannot run because these model features "
            f"are missing from the dataset: {missing_features}"
        )

    else:

        shap_data = filtered[
            model_features
        ].dropna().copy()

        if len(shap_data) > 200:

            shap_data = shap_data.sample(
                200,
                random_state=42
            )

        if len(shap_data) < 2:

            st.warning(
                "Not enough valid records are available for SHAP analysis."
            )

        else:

            # The trained model is a preprocessing pipeline
            # followed by Random Forest.
            if hasattr(trained_model, "steps"):

                preprocessing = trained_model[:-1]
                classifier = trained_model[-1]

                transformed_data = preprocessing.transform(
                    shap_data
                )

                try:

                    feature_names = (
                        preprocessing.get_feature_names_out()
                    )

                except Exception:

                    feature_names = [
                        f"Feature {i + 1}"
                        for i in range(
                            transformed_data.shape[1]
                        )
                    ]

            else:

                st.warning(
                    "The saved model is not a preprocessing pipeline, "
                    "so SHAP analysis cannot be generated automatically."
                )

                transformed_data = None
                classifier = None

            if transformed_data is not None:

                explainer = shap.TreeExplainer(
                    classifier
                )

                shap_values = explainer.shap_values(
                    transformed_data
                )

                # Handle different SHAP versions
                if isinstance(shap_values, list):

                    if len(shap_values) > 1:
                        values = shap_values[1]
                    else:
                        values = shap_values[0]

                elif len(shap_values.shape) == 3:

                    values = shap_values[:, :, 1]

                else:

                    values = shap_values

                # Mean absolute SHAP value
                importance = pd.DataFrame({
                    "Feature": feature_names,
                    "Mean |SHAP Value|": (
                        abs(values).mean(axis=0)
                    )
                })

                importance = importance.sort_values(
                    "Mean |SHAP Value|",
                    ascending=False
                )

                fig = px.bar(
                    importance.head(15).sort_values(
                        "Mean |SHAP Value|"
                    ),
                    x="Mean |SHAP Value|",
                    y="Feature",
                    orientation="h",
                    title="Top Features by SHAP Importance"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.subheader(
                    "SHAP Feature Importance Table"
                )

                st.dataframe(
                    importance,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "SHAP values explain the model's contribution of "
                    "features to predictions. They indicate model behavior "
                    "and should not be interpreted as proof of causation."
                )

except Exception as e:

    st.error(
        "SHAP analysis could not be generated."
    )

    st.code(
        str(e)
    )


# --------------------------------------------------
# 14. RESPONSIBLE AI
# --------------------------------------------------

st.subheader(
    "Responsible AI Considerations"
)

st.markdown(
    """
- **Transparency:** predictions are generated by a trained model and should
  be interpreted alongside the input measurements.

- **Explainability:** SHAP values are provided to show which input features
  contribute most strongly to model predictions.

- **Reliability:** evaluate the model on held-out data and monitor errors
  and input quality.

- **Fairness:** compare prediction errors across relevant locations where
  sufficient observations exist. Location comparisons are not a substitute
  for demographic fairness analysis.

- **Privacy:** avoid collecting personal information that is not needed
  for AQI prediction.

- **Limitations:** model outputs are estimates and should not replace
  official air-quality readings or public-health guidance.
"""
)


# --------------------------------------------------
# 15. DATA PREVIEW AND DOWNLOAD
# --------------------------------------------------

with st.expander(
    "View filtered dataset"
):

    st.dataframe(
        filtered,
        use_container_width=True
    )

    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download filtered data",
        data=csv_data,
        file_name="aqi_filtered_data.csv",
        mime="text/csv"
    )