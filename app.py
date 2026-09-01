
# ============================================================
# FAKE NEWS DETECTION SYSTEM
# ============================================================
# Machine Learning Based Fake News Classification
#
# Technology:
# - Python
# - Pandas
# - Scikit-learn
# - TF-IDF
# - Linear SVM
# - Streamlit
#
# Final Model:
# Calibrated Linear SVM
# ============================================================

import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT CONSTANTS
# ============================================================

MODEL_PATH = "models/fake_news_calibrated_pipeline.pkl"

TOTAL_ARTICLES = 44689
FAKE_ARTICLES = 23478
REAL_ARTICLES = 21211


# ============================================================
# LOAD TRAINED DEPLOYMENT MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained model."
    )

    st.error(
        f"Error: {e}"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📰 Fake News Detection System")

st.markdown(
    """
    ### Machine Learning Based News Classification

    This application uses **TF-IDF text representation**
    and a **Calibrated Linear SVM classifier** to classify
    news articles as **FAKE** or **REAL**.

    The system supports:

    - Individual news classification
    - Prediction confidence
    - Class probability distribution
    - Batch CSV classification
    - Downloadable classification results
    - Model performance comparison
    """
)

st.divider()


# ============================================================
# SIDEBAR — PROJECT INFORMATION
# ============================================================

with st.sidebar:

    st.header("📌 Project Information")

    st.write("**Project:** Fake News Detection")

    st.write("**Task:** Binary Text Classification")

    st.write("**Dataset:** Fake.csv + True.csv")

    st.write("**Final Classifier:** Linear SVM")

    st.write("**Deployment Model:** Calibrated Linear SVM")

    st.write("**Feature Extraction:** TF-IDF")


    st.divider()


    st.subheader("📊 Dataset Statistics")

    st.write(
        f"Total Articles: **{TOTAL_ARTICLES:,}**"
    )

    st.write(
        f"Fake Articles: **{FAKE_ARTICLES:,}**"
    )

    st.write(
        f"Real Articles: **{REAL_ARTICLES:,}**"
    )


    st.divider()


    st.subheader("🔬 ML Pipeline")

    st.write("1. Dataset Collection")

    st.write("2. Data Cleaning")

    st.write("3. Duplicate Removal")

    st.write("4. Text Preprocessing")

    st.write("5. TF-IDF Vectorization")

    st.write("6. Model Training")

    st.write("7. Model Comparison")

    st.write("8. Linear SVM Selection")

    st.write("9. Probability Calibration")

    st.write("10. Streamlit Deployment")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("🏆 Selected Model Performance")


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Selected Model",
        "Linear SVM"
    )


with c2:

    st.metric(
        "Accuracy",
        "99.77%"
    )


with c3:

    st.metric(
        "Precision",
        "99.67%"
    )


with c4:

    st.metric(
        "F1 Score",
        "99.75%"
    )


st.caption(
    "Linear SVM achieved the highest overall performance "
    "among the evaluated machine learning models."
)


# ============================================================
# DEPLOYMENT MODEL INFORMATION
# ============================================================

st.info(
    """
    **Deployment Model:** A calibrated version of the selected
    Linear SVM is used to generate probability-based
    confidence scores for predictions.
    """
)


st.divider()


# ============================================================
# SECTION 1 — SINGLE ARTICLE PREDICTION
# ============================================================

st.header("🔍 Analyze a News Article")

st.write(
    "Paste a complete news article or news headline below."
)


news_text = st.text_area(
    "News Article",
    height=250,
    placeholder="Paste the news article here..."
)


analyze_button = st.button(
    "🔍 Analyze News",
    use_container_width=True
)


if analyze_button:

    if not news_text.strip():

        st.warning(
            "⚠️ Please enter a news article."
        )

    elif len(news_text.strip()) < 30:

        st.warning(
            "⚠️ Please enter at least 30 characters "
            "for a more meaningful classification."
        )

    else:

        try:

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                [news_text]
            )[0]


            # ------------------------------------------------
            # PROBABILITY PREDICTION
            # ------------------------------------------------

            probabilities = model.predict_proba(
                [news_text]
            )[0]


            # ------------------------------------------------
            # IDENTIFY CLASS PROBABILITIES
            # ------------------------------------------------

            classes = list(model.classes_)

            fake_index = classes.index(0)

            real_index = classes.index(1)

            fake_probability = float(
                probabilities[fake_index]
            )

            real_probability = float(
                probabilities[real_index]
            )


            # ------------------------------------------------
            # FINAL LABEL
            # ------------------------------------------------

            if prediction == 0:

                label = "FAKE"

                confidence = fake_probability

            else:

                label = "REAL"

                confidence = real_probability


            # ------------------------------------------------
            # DISPLAY MAIN RESULT
            # ------------------------------------------------

            if label == "FAKE":

                st.error(
                    f"🚨 Prediction: {label}"
                )

            else:

                st.success(
                    f"✅ Prediction: {label}"
                )


            # ------------------------------------------------
            # RESULT CARDS
            # ------------------------------------------------

            st.subheader("📊 Prediction Result")


            r1, r2, r3 = st.columns(3)


            with r1:

                st.metric(
                    "Classification",
                    label
                )


            with r2:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )


            with r3:

                if confidence >= 0.90:

                    confidence_level = "Very High"

                elif confidence >= 0.75:

                    confidence_level = "High"

                elif confidence >= 0.60:

                    confidence_level = "Moderate"

                else:

                    confidence_level = "Low"


                st.metric(
                    "Confidence Level",
                    confidence_level
                )


            # ------------------------------------------------
            # CONFIDENCE BAR
            # ------------------------------------------------

            st.write(
                "**Prediction Confidence**"
            )

            st.progress(
                min(max(float(confidence), 0.0), 1.0)
            )


            # ------------------------------------------------
            # CLASS PROBABILITIES
            # ------------------------------------------------

            st.subheader(
                "📈 Class Probability Distribution"
            )


            p1, p2 = st.columns(2)


            with p1:

                st.metric(
                    "🔴 Fake Probability",
                    f"{fake_probability * 100:.2f}%"
                )


            with p2:

                st.metric(
                    "🟢 Real Probability",
                    f"{real_probability * 100:.2f}%"
                )


            # ------------------------------------------------
            # PROBABILITY DATAFRAME
            # ------------------------------------------------

            probability_df = pd.DataFrame(
                {
                    "Probability (%)": [
                        fake_probability * 100,
                        real_probability * 100
                    ]
                },
                index=[
                    "Fake",
                    "Real"
                ]
            )


            # ------------------------------------------------
            # PROBABILITY VISUALIZATION
            # ------------------------------------------------

            st.bar_chart(
                probability_df
            )


            # ------------------------------------------------
            # DISCLAIMER FOR PREDICTION
            # ------------------------------------------------

            st.info(
                """
                The confidence score represents the calibrated
                model probability associated with the prediction.

                It does not guarantee that an article is factually
                true or false. Important information should always
                be verified using reliable independent sources.
                """
            )


        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.error(
                f"Error: {e}"
            )


# ============================================================
# SECTION 2 — BATCH CSV PREDICTION
# ============================================================

st.divider()

st.header("📁 Batch News Classification")

st.write(
    """
    Upload a CSV file containing multiple news articles.

    The application automatically detects common text
    columns such as **text**, **content**, **article**,
    **news**, or **title**.
    """
)


uploaded_csv = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)


if uploaded_csv is not None:

    try:

        # ------------------------------------------------
        # READ CSV
        # ------------------------------------------------

        batch_df = pd.read_csv(
            uploaded_csv
        )


        st.subheader(
            "📄 Uploaded Data"
        )


        st.write(
            f"Rows: **{len(batch_df):,}**"
        )


        st.dataframe(
            batch_df.head(10),
            use_container_width=True
        )


        # ------------------------------------------------
        # AUTOMATIC TEXT COLUMN DETECTION
        # ------------------------------------------------

        preferred_columns = [
            "text",
            "content",
            "article",
            "news",
            "title"
        ]


        text_column = None


        column_mapping = {
            str(column).strip().lower(): column
            for column in batch_df.columns
        }


        for preferred in preferred_columns:

            if preferred in column_mapping:

                text_column = column_mapping[
                    preferred
                ]

                break


        # ------------------------------------------------
        # MANUAL COLUMN SELECTION
        # ------------------------------------------------

        if text_column is None:

            st.warning(
                "No standard text column was detected. "
                "Please select the column containing news text."
            )


            text_column = st.selectbox(
                "Select News Text Column",
                batch_df.columns
            )


        else:

            st.success(
                f"Text column detected: **{text_column}**"
            )


        # ------------------------------------------------
        # BATCH CLASSIFICATION BUTTON
        # ------------------------------------------------

        classify_button = st.button(
            "🚀 Classify All News",
            use_container_width=True
        )


        if classify_button:

            if len(batch_df) == 0:

                st.warning(
                    "The uploaded CSV contains no records."
                )

            else:

                # ----------------------------------------
                # PREPARE TEXT DATA
                # ----------------------------------------

                texts = (
                    batch_df[text_column]
                    .fillna("")
                    .astype(str)
                    .tolist()
                )


                # ----------------------------------------
                # MODEL PREDICTION
                # ----------------------------------------

                predictions = model.predict(
                    texts
                )


                # ----------------------------------------
                # MODEL PROBABILITIES
                # ----------------------------------------

                probabilities = model.predict_proba(
                    texts
                )


                # ----------------------------------------
                # CLASS INFORMATION
                # ----------------------------------------

                classes = list(model.classes_)

                fake_index = classes.index(0)

                real_index = classes.index(1)


                # ----------------------------------------
                # CREATE OUTPUT DATAFRAME
                # ----------------------------------------

                output_df = batch_df.copy()


                output_df["Prediction"] = [
                    "FAKE" if prediction == 0
                    else "REAL"
                    for prediction in predictions
                ]


                output_df["Confidence"] = [
                    round(
                        max(probability) * 100,
                        2
                    )
                    for probability in probabilities
                ]


                output_df["Fake Probability"] = [
                    round(
                        probability[fake_index] * 100,
                        2
                    )
                    for probability in probabilities
                ]


                output_df["Real Probability"] = [
                    round(
                        probability[real_index] * 100,
                        2
                    )
                    for probability in probabilities
                ]


                # ----------------------------------------
                # CLASSIFICATION SUMMARY
                # ----------------------------------------

                st.subheader(
                    "📊 Batch Classification Summary"
                )


                fake_count = (
                    output_df["Prediction"] == "FAKE"
                ).sum()


                real_count = (
                    output_df["Prediction"] == "REAL"
                ).sum()


                b1, b2, b3 = st.columns(3)


                with b1:

                    st.metric(
                        "Total Articles",
                        f"{len(output_df):,}"
                    )


                with b2:

                    st.metric(
                        "Fake",
                        f"{fake_count:,}"
                    )


                with b3:

                    st.metric(
                        "Real",
                        f"{real_count:,}"
                    )


                # ----------------------------------------
                # DISPLAY RESULTS
                # ----------------------------------------

                st.subheader(
                    "📋 Classification Results"
                )


                st.dataframe(
                    output_df,
                    use_container_width=True
                )


                # ----------------------------------------
                # DOWNLOAD RESULTS
                # ----------------------------------------

                csv_output = (
                    output_df
                    .to_csv(index=False)
                    .encode("utf-8")
                )


                st.download_button(
                    label="⬇️ Download Classification Results",
                    data=csv_output,
                    file_name="fake_news_batch_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )


    except Exception as e:

        st.error(
            "Unable to process the uploaded CSV file."
        )

        st.error(
            f"Error: {e}"
        )


# ============================================================
# SECTION 3 — MODEL PERFORMANCE COMPARISON
# ============================================================

st.divider()

st.header("📊 Model Performance Comparison")


comparison = pd.DataFrame(
    {
        "Model": [
            "Linear SVM",
            "Logistic Regression",
            "Multinomial Naive Bayes"
        ],

        "Accuracy": [
            99.77,
            98.88,
            96.18
        ],

        "Precision": [
            99.67,
            98.48,
            94.87
        ],

        "Recall": [
            99.83,
            99.17,
            97.22
        ],

        "F1 Score": [
            99.75,
            98.83,
            96.03
        ]
    }
)


# ------------------------------------------------------------
# MODEL COMPARISON TABLE
# ------------------------------------------------------------

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# ACCURACY VISUALIZATION
# ------------------------------------------------------------

st.subheader(
    "📈 Accuracy Comparison"
)


accuracy_chart = comparison.set_index(
    "Model"
)[
    ["Accuracy"]
]


st.bar_chart(
    accuracy_chart
)


# ------------------------------------------------------------
# COMPLETE METRICS VISUALIZATION
# ------------------------------------------------------------

st.subheader(
    "📊 Complete Model Evaluation"
)


metric_chart = comparison.set_index(
    "Model"
)[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
]


st.bar_chart(
    metric_chart
)


# ============================================================
# SECTION 4 — WHY LINEAR SVM?
# ============================================================

st.subheader(
    "🏆 Why Was Linear SVM Selected?"
)


st.success(
    """
    Linear SVM was selected because it achieved the highest
    overall performance among the evaluated models.

    • Accuracy  : 99.77%
    • Precision : 99.67%
    • Recall    : 99.83%
    • F1 Score  : 99.75%

    Logistic Regression and Multinomial Naive Bayes achieved
    lower overall performance.

    Therefore, Linear SVM was selected as the final base
    classifier. A calibrated version of this classifier is
    used for deployment to provide probability-based
    confidence scores.
    """
)


# ============================================================
# SECTION 5 — PROJECT METHODOLOGY
# ============================================================

st.divider()

st.header(
    "⚙️ Project Methodology"
)


m1, m2, m3 = st.columns(3)


with m1:

    st.markdown(
        "### 1️⃣ Data Processing"
    )

    st.write(
        "Fake and real news datasets were combined. "
        "Missing values were checked, duplicate records "
        "were removed, and the text was cleaned."
    )


with m2:

    st.markdown(
        "### 2️⃣ Feature Extraction"
    )

    st.write(
        "TF-IDF converted the cleaned news text into "
        "numerical features suitable for machine learning."
    )


with m3:

    st.markdown(
        "### 3️⃣ Classification"
    )

    st.write(
        "Logistic Regression, Linear SVM and Multinomial "
        "Naive Bayes were trained and evaluated. Linear SVM "
        "was selected as the best-performing model."
    )


# ============================================================
# SECTION 6 — DEPLOYMENT PIPELINE
# ============================================================

st.divider()

st.header(
    "🚀 Deployment Pipeline"
)


deployment_steps = pd.DataFrame(
    {
        "Stage": [
            "Input News",
            "Text Processing",
            "TF-IDF Transformation",
            "Linear SVM Classification",
            "Probability Calibration",
            "Final Prediction"
        ],

        "Description": [
            "User enters or uploads news content.",
            "The trained pipeline processes the text.",
            "TF-IDF converts text into numerical features.",
            "Linear SVM classifies the news.",
            "Calibration generates probability estimates.",
            "System displays FAKE or REAL with confidence."
        ]
    }
)


st.dataframe(
    deployment_steps,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    ⚠️ DISCLAIMER

    This application is a machine-learning-based text
    classification system. It identifies linguistic and
    statistical patterns learned from the training dataset.

    It should not be considered a definitive fact-checking
    or truth-verification system.

    Important information should always be verified using
    reliable and independent sources.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Fake News Detection System | "
    "TF-IDF + Calibrated Linear SVM | "
    "Machine Learning / NLP Project"
)


st.caption(
    "Academic Project — Streamlit Deployment"
)
