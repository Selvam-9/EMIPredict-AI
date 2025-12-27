import streamlit as st
import os
import joblib
import pandas as pd

# MUST be the first Streamlit command
st.set_page_config(page_title="EMIPredict AI", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.write("🔍 BASE_DIR:", BASE_DIR)
st.write("📂 Files in BASE_DIR:", os.listdir(BASE_DIR))

@st.cache_resource
def load_classification_model():
    return joblib.load(os.path.join(BASE_DIR, "classification_model.pkl"))

@st.cache_resource
def load_regression_model():
    return joblib.load(os.path.join(BASE_DIR, "regression_model.pkl"))

clf_model = load_classification_model()
reg_model = load_regression_model()


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("EMIPredict AI")
st.subheader("Intelligent Financial Risk Assessment Platform")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "EMI Prediction", "Model Performance", "About"])

# --------------------------------------------------
# Home
# --------------------------------------------------
if page == "Home":
    st.header("Welcome to EMIPredict AI")
    st.write(
        "This application helps assess EMI eligibility and "
        "predict safe monthly EMI amounts using machine learning."
    )

# --------------------------------------------------
# EMI Prediction
# --------------------------------------------------

elif page == "EMI Prediction":
    st.header("EMI Eligibility & EMI Amount Prediction")

    with st.form("emi_form"):
        col1, col2 = st.columns(2)

        with col1:
            monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0, step=1000)
            current_emi = st.number_input("Current EMI Amount (₹)", min_value=0, step=500)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=10)

        with col2:
            monthly_expenses = st.number_input("Total Monthly Expenses (₹)", min_value=0, step=1000)
            bank_balance = st.number_input("Bank Balance (₹)", min_value=0, step=5000)
            emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0, step=5000)
            employment_type = st.selectbox("Employment Type",["Private", "Government", "Self-employed"])

        submit = st.form_submit_button("Predict EMI")

    if submit:
        # --------------------------------------------------
        # Feature Engineering
        # --------------------------------------------------
                
        emi_ratio = current_emi / monthly_salary if monthly_salary > 0 else 0
        buffer_months = bank_balance / monthly_expenses if monthly_expenses > 0 else 0

        emp_private = 1 if employment_type == "Private" else 0
        emp_self_employed = 1 if employment_type == "Self-employed" else 0

        st.subheader("Computed Financial Indicators")
        col1, col2 = st.columns(2)
        col1.metric("EMI Ratio", round(emi_ratio, 3))
        col2.metric("Buffer Months", round(buffer_months, 2))

        # --------------------------------------------------
        # Risk Explanation
        # --------------------------------------------------
        st.subheader("Risk Explanation")

        reasons = []

        if buffer_months < 1:
            reasons.append("Very low financial buffer (less than 1 month of expenses).")

        if buffer_months < 3:
            reasons.append("Low savings buffer increases repayment risk.")

        elif buffer_months < 6:
            reasons.append("Moderate savings buffer; financial resilience is limited.")

        else:
            reasons.append("Strong savings buffer supports EMI repayment stability.")

        if emi_ratio > 0.4:
            reasons.append("High existing EMI burden relative to income.")

        if credit_score < 650:
            reasons.append("Low credit score increases default risk.")

        for r in reasons:
            st.write("•", r)
        else:
            st.write("• Overall financial indicators are stable.")

        # --------------------------------------------------
        # Classifaction Prediction
        # --------------------------------------------------    

        clf_input_df = pd.DataFrame([{
            "monthly_salary": monthly_salary,
            "current_emi_amount": current_emi,
            "credit_score": credit_score,
            "emi_ratio": emi_ratio,
            "buffer_months": buffer_months,
            "emp_Private": emp_private,
            "emp_Self-employed": emp_self_employed
        }])
        pred_class = clf_model.predict(clf_input_df)[0]
        
        class_map = {
            0: "Eligible",
            1: "High_Risk",
            2: "Not_Eligible"
        }

        original_prediction = class_map[pred_class]
        eligibility = original_prediction

        # -------------------------------
        # Business-rule softening
        # -------------------------------
        if eligibility == "Not_Eligible":
            if (
                emi_ratio <= 0.20 and
                buffer_months >= 6 and
                credit_score >= 700
            ):
                eligibility = "High_Risk"

        # -------------------------------
        # Final Decision
        # -------------------------------
        st.subheader("Final Decision")
        
        if eligibility == "Eligible":
            st.success("✅ Eligible for EMI")
        elif eligibility == "High_Risk":
            st.warning("⚠️ High Risk – Conditional Approval")
        else:
            st.error("❌ Not Eligible for EMI")
            st.info(
                "Maximum EMI amount is not shown because the applicant "
                "is currently not eligible for EMI approval."
            )

        if original_prediction != eligibility:
            st.info(
                "Maximum EMI amount is not shown because the applicant "
                "is currently not eligible for EMI approval."
            )
        # -------------------------------
        # Regression (Gated)
        # -------------------------------    
        if eligibility in ["Eligible", "High_Risk"]:
            reg_input_df = pd.DataFrame([{
                "monthly_salary": monthly_salary,
                "current_emi_amount": current_emi,
                "credit_score": credit_score,
                "emi_ratio": emi_ratio,
                "buffer_months": buffer_months,
                "emp_Private": emp_private,
                "emp_Self-employed": emp_self_employed
            }])

            predicted_max_emi = reg_model.predict(reg_input_df)[0]

            st.subheader("Recommended Maximum EMI")
            st.metric(
                "Max Safe Monthly EMI (₹)",
                f"{int(predicted_max_emi):,}"
            )

            if eligibility == "High_Risk":
                st.caption(
                    "Recommendation is conservative due to elevated financial risk."
                )
    st.caption(
    "⚠️ Disclaimer: EMI recommendations are generated using machine learning models "
    "and historical data. Final loan decisions should involve human review.")
# ------------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------------
elif page == "Model Performance":
    st.header("Model Performance Summary")

    st.markdown("### 🔍 Classification Models (EMI Eligibility)")

    clf_perf = pd.DataFrame({
        "Model": [
            "Logistic Regression (Baseline)",
            "Random Forest (Final)",
            "XGBoost (Tuned)"
        ],
        "Accuracy": [
            "0.79",
            "0.80",
            "0.80"
        ],
        "F1 Macro": [
            "0.39",
            "0.44",
            "0.44"
        ],
        "Remarks": [
            "Baseline interpretable model",
            "Best balance across classes",
            "Similar performance, higher complexity"
        ]
    })

    st.dataframe(clf_perf, use_container_width=True)

    st.info(
        "Random Forest was selected as the final classification model "
        "due to better macro F1-score, handling class imbalance more effectively."
    )

    st.markdown("---")
    st.markdown("### 📈 Regression Models (Maximum EMI Prediction)")

    reg_perf = pd.DataFrame({
        "Model": [
            "Linear Regression (Baseline)",
            "Random Forest Regressor (Final)",
            "XGBoost Regressor"
        ],
        "MAE (₹)": [
            "3997",
            "679",
            "813"
        ],
        "RMSE (₹)": [
            "5739",
            "1499",
            "1583"
        ],
        "R² Score": [
            "0.44",
            "0.96",
            "0.95"
        ],
        "Remarks": [
            "Baseline model",
            "Best accuracy & lowest error",
            "Comparable but slightly weaker"
        ]
    })

    st.dataframe(reg_perf, use_container_width=True)

    st.success(
        "Random Forest Regressor was selected due to lowest RMSE "
        "and highest R², making it suitable for EMI recommendation."
    )

    st.markdown("---")
    st.markdown("### 🧪 Experiment Tracking")

    st.write(
        "All experiments, hyperparameters, and metrics were tracked using MLflow. "
        "This ensures reproducibility, model comparison, and version control."
    )
    st.caption(
    "Model metrics are reported on the hold-out test set. "
    "Macro F1-score is emphasized for classification due to class imbalance."
)


# ------------------------------------------------------------------
# ABOUT
# ------------------------------------------------------------------
elif page == "About":
    st.header("About EMIPredict AI")

    st.write("""
    **EMIPredict AI** is an intelligent financial risk assessment platform 
    designed to assist EMI eligibility evaluation and safe EMI amount recommendation.

    ### Key Features
    - Dual machine learning models:
        - Classification for EMI eligibility
        - Regression for maximum EMI recommendation
    - Advanced feature engineering using financial ratios
    - MLflow-based experiment tracking and model management
    - Interactive Streamlit application for real-time assessment

    ### Technology Stack
    - Python, Pandas, Scikit-learn
    - Random Forest, Logistic Regression, XGBoost
    - MLflow for experiment tracking
    - Streamlit for web deployment

    ### Business Objective
    The system aims to support responsible lending by combining 
    data-driven insights with business rules to minimize financial risk.
    """)








