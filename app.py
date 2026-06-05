import streamlit as st
import pandas as pd
import joblib
import os
import re
from datetime import date

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Health Prediction System",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("health_model.pkl")

# ==========================================
# DATA FILE
# ==========================================

DATA_FILE = "patient_records.csv"

if not os.path.exists(DATA_FILE):
    empty_df = pd.DataFrame(columns=[
        "ID",
        "Full Name",
        "Date of Birth",
        "Email",
        "Glucose",
        "Haemoglobin",
        "Cholesterol",
        "Remarks"
    ])
    empty_df.to_csv(DATA_FILE, index=False)

# ==========================================
# FUNCTIONS
# ==========================================

def load_data():
    try:
        df = pd.read_csv(DATA_FILE)

        if not df.empty:
            df["ID"] = df["ID"].astype(int)

        return df

    except Exception:
        return pd.DataFrame(columns=[
            "ID",
            "Full Name",
            "Date of Birth",
            "Email",
            "Glucose",
            "Haemoglobin",
            "Cholesterol",
            "Remarks"
        ])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email)

def predict_risk(glucose, haemoglobin, cholesterol):

    input_data = pd.DataFrame({
        "Glucose": [glucose],
        "Haemoglobin": [haemoglobin],
        "Cholesterol": [cholesterol]
    })

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    return prediction, probabilities

# ==========================================
# TITLE
# ==========================================

st.title("🏥 Health Prediction System")

st.markdown("""
### Features

- Create Patient Records
- Read Patient Records
- Update Patient Records
- Delete Patient Records
- AI/ML Health Risk Prediction
- CSV-Based Persistent Storage
""")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Select Operation",
    ["Create", "Read", "Update", "Delete"]
)

# ==========================================
# CREATE
# ==========================================

if menu == "Create":

    st.header("➕ Add Patient")

    name = st.text_input("Full Name")

    dob = st.date_input(
        "Date of Birth",
        min_value=date(1900, 1, 1),
        max_value=date.today()
    )

    email = st.text_input("Email Address")

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        value=100.0
    )

    haemoglobin = st.number_input(
        "Haemoglobin",
        min_value=0.0,
        value=13.5
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=0.0,
        value=180.0
    )

    if st.button("Predict & Save"):

        if not name.strip():
            st.error("Full Name is required")

        elif not validate_email(email):
            st.error("Invalid Email Address")

        else:

            prediction, probabilities = predict_risk(
                glucose,
                haemoglobin,
                cholesterol
            )

            df = load_data()

            if df.empty:
                new_id = 1
            else:
                new_id = int(df["ID"].max()) + 1

            new_record = pd.DataFrame({
                "ID": [new_id],
                "Full Name": [name],
                "Date of Birth": [dob],
                "Email": [email],
                "Glucose": [glucose],
                "Haemoglobin": [haemoglobin],
                "Cholesterol": [cholesterol],
                "Remarks": [prediction]
            })

            df = pd.concat(
                [df, new_record],
                ignore_index=True
            )

            save_data(df)

            st.success("Patient Record Saved Successfully")

            if prediction == "Healthy":
                st.success(f"Prediction: {prediction}")
            elif prediction == "Moderate Risk":
                st.warning(f"Prediction: {prediction}")
            else:
                st.error(f"Prediction: {prediction}")

            prob_df = pd.DataFrame({
                "Condition": model.classes_,
                "Probability": probabilities
            })

            st.subheader("Prediction Confidence")
            st.dataframe(prob_df)
            st.bar_chart(prob_df.set_index("Condition"))

# ==========================================
# READ
# ==========================================

elif menu == "Read":

    st.header("📋 Patient Records")

    df = load_data()

    if df.empty:
        st.warning("No records found")

    else:

        search = st.text_input(
            "Search by Patient Name"
        )

        if search:
            df = df[
                df["Full Name"]
                .astype(str)
                .str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        st.dataframe(
            df,
            use_container_width=True
        )

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            file_name="patient_records.csv",
            mime="text/csv"
        )

# ==========================================
# UPDATE
# ==========================================

elif menu == "Update":

    st.header("✏️ Update Patient Record")

    df = load_data()

    if df.empty:
        st.warning("No records available")

    else:

        patient_id = st.selectbox(
            "Select Patient ID",
            df["ID"].tolist()
        )

        row = df[df["ID"] == patient_id].iloc[0]

        name = st.text_input(
            "Full Name",
            value=str(row["Full Name"])
        )

        dob = st.date_input(
            "Date of Birth",
            value=pd.to_datetime(
                row["Date of Birth"]
            ).date()
        )

        email = st.text_input(
            "Email Address",
            value=str(row["Email"])
        )

        glucose = st.number_input(
            "Glucose",
            value=float(row["Glucose"])
        )

        haemoglobin = st.number_input(
            "Haemoglobin",
            value=float(row["Haemoglobin"])
        )

        cholesterol = st.number_input(
            "Cholesterol",
            value=float(row["Cholesterol"])
        )

        if st.button("Update Record"):

            if not validate_email(email):
                st.error("Invalid Email Address")

            else:

                prediction, _ = predict_risk(
                    glucose,
                    haemoglobin,
                    cholesterol
                )

                idx = df[
                    df["ID"] == patient_id
                ].index[0]

                df.loc[idx, "Full Name"] = name
                df.loc[idx, "Date of Birth"] = str(dob)
                df.loc[idx, "Email"] = email
                df.loc[idx, "Glucose"] = glucose
                df.loc[idx, "Haemoglobin"] = haemoglobin
                df.loc[idx, "Cholesterol"] = cholesterol
                df.loc[idx, "Remarks"] = prediction

                save_data(df)

                st.success(
                    "Record Updated Successfully"
                )

                st.rerun()

# ==========================================
# DELETE
# ==========================================

elif menu == "Delete":

    st.header("🗑️ Delete Patient Record")

    df = load_data()

    if df.empty:
        st.warning("No records available")

    else:

        patient_id = st.selectbox(
            "Select Patient ID",
            df["ID"].tolist()
        )

        patient = df[
            df["ID"] == patient_id
        ]

        st.subheader("Selected Record")

        st.dataframe(
            patient,
            use_container_width=True
        )

        confirm = st.checkbox(
            "I confirm deletion"
        )

        if st.button("Delete Record"):

            if not confirm:
                st.warning(
                    "Please confirm deletion first"
                )

            else:

                df = df[
                    df["ID"] != patient_id
                ]

                save_data(df)

                st.success(
                    "Record Deleted Successfully"
                )

                st.rerun()
                