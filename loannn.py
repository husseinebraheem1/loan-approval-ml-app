import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import joblib




data = pd.read_csv('Loan approval prediction.csv')




def home():
    st.markdown(
        "<h1 style='color:#FF6F61; text-align:center;'>💳 Welcome to the Loan Data Analysis Application</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style='background-color:#FDEBD0; padding:15px; border-radius:10px;'>
        <h3 style='color:#884EA0;'>🎯 Application Objectives:</h3>
        <ul style='color:#2C3E50; font-size:16px;'>
            <li>🔍 <b>Explore</b> the dataset using colorful EDA to reveal hidden patterns.</li>
            <li>📈 <b>Enhance</b> chances of loan approval through data analysis.</li>
            <li>✅ <b>Ensure</b> loans are granted to deserving applicants.</li>
            <li>🤖 <b>Predict</b> approvals using Machine Learning.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style='background-color:#D6EAF8; padding:15px; border-radius:10px; margin-top:15px;'>
        <h3 style='color:#1F618D;'>💡 Why Exploratory Data Analysis (EDA)?</h3>
        <p style='color:#154360;'>EDA helps uncover relationships between variables and identify key factors influencing loan approvals.</p>
        
        <h3 style='color:#1F618D;'>🤖 Why Prediction?</h3>
        <p style='color:#154360;'>Machine Learning models predict approval likelihood, enabling banks to make data-driven decisions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style='background-color:#F9EBEA; padding:15px; border-radius:10px; margin-top:15px;'>
        <h3 style='color:#C0392B;'>📌 Navigation</h3>
        <p style='color:#641E16;'>
        Choose from three main sections in the sidebar:
        <ul>
            <li>🏠 <b>Home</b>: Overview and Introduction.</li>
            <li>📊 <b>EDA</b>: Perform data exploration.</li>
            <li>🤖 <b>Machine Learning</b>: Build and evaluate models.</li>
        </ul>
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<p style='text-align:center; color:#7D3C98; font-size:18px;'>🚀 Keep exploring to discover more about boosting loan approval chances!</p>", unsafe_allow_html=True)


def eda():
    sns.set_style("dark", {"axes.facecolor": "#1e1e2f"})


def eda():
    st.title("🎯 Exploratory Data Analysis (EDA) Dashboard")
    st.markdown("""
    **Welcome to the colorful world of Data Exploration!** 🌈  
    Here, you'll **explore** 📊, **compare** 🔍, and **visualize** 🎨  
    your loan dataset in the most engaging way possible.
    """)

    data = pd.read_csv('Loan approval prediction.csv')

    st.subheader("📋 Dataset Preview")
    st.dataframe(data.head().style.background_gradient(cmap="Blues"))

    columns = ['person_age', 'person_income', 'person_home_ownership',
       'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
       'cb_person_cred_hist_length', 'loan_status']

    selected_column = st.selectbox("🎨 Select a column for EDA", columns)
    st.success(f"Now analyzing: {selected_column}")

    fig, ax = plt.subplots()
    ax.hist(data[selected_column].dropna(), bins=20, edgecolor='black', color='skyblue')
    ax.set_title(f"📊 Histogram of {selected_column}", fontsize=14, color="darkblue")
    st.pyplot(fig)

    if pd.api.types.is_numeric_dtype(data[selected_column]):
        fig, ax = plt.subplots()
        data.boxplot(column=selected_column, ax=ax, color="purple")
        ax.set_title(f"📦 Boxplot of {selected_column}", fontsize=14, color="purple")
        st.pyplot(fig)

    else:
        st.warning(f"'{selected_column}' is not numerical. Skipping boxplot.")

    st.subheader("🔀 Compare Two Columns")
    selected_column1 = st.selectbox("Select the first column", columns, key="col1")
    selected_column2 = st.selectbox("Select the second column", columns, key="col2")

    if selected_column1 == selected_column2:
        st.error("❌ Please select two different columns.")
    else:
        if pd.api.types.is_numeric_dtype(data[selected_column1]) and pd.api.types.is_numeric_dtype(data[selected_column2]):
            fig, ax = plt.subplots()
            ax.scatter(data[selected_column1], data[selected_column2], edgecolor='black', color='teal')
            ax.set_xlabel(selected_column1)
            ax.set_ylabel(selected_column2)
            ax.set_title(f"🔵 Scatter plot: {selected_column1} vs {selected_column2}")
            st.pyplot(fig)
        elif not pd.api.types.is_numeric_dtype(data[selected_column1]) and not pd.api.types.is_numeric_dtype(data[selected_column2]):
            fig, ax = plt.subplots()
            sns.countplot(data=data, x=selected_column1, hue=selected_column2, ax=ax, palette="coolwarm")
            ax.set_title(f"📊 Countplot: {selected_column1} vs {selected_column2}")
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            sns.boxplot(data=data, x=selected_column1, y=selected_column2, ax=ax, palette="viridis")
            ax.set_title(f"📦 Boxplot: {selected_column1} vs {selected_column2}")
            st.pyplot(fig)

    st.subheader("✅ Loan Status Analysis")
    loan_status_options = ['Accepted', 'Rejected']
    status_choice = st.selectbox("Select Loan Status:", loan_status_options)

    filtered_data = data[data['loan_status'] == (1 if status_choice == 'Accepted' else 0)]
    st.dataframe(filtered_data.head().style.background_gradient(cmap="YlGn"))

    selected_column = st.selectbox("Select column to visualize:", columns, key="status_col")
    if pd.api.types.is_numeric_dtype(filtered_data[selected_column]):
        fig, ax = plt.subplots()
        sns.histplot(filtered_data[selected_column], kde=True, ax=ax, color="orange")
        ax.set_title(f"📊 Histogram of {selected_column} ({status_choice} loans)")
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_data, x=selected_column, ax=ax, palette="Set2")
        ax.set_title(f"📊 Countplot of {selected_column} ({status_choice} loans)")
        st.pyplot(fig)

    st.markdown("✨ **Exploration Complete!** Data never looked so good. 🚀")



        
    






# Machine Learning

def machine_learning():
    import os
    import joblib
    import pandas as pd
    import streamlit as st
    from sklearn.preprocessing import LabelEncoder

    st.title("Machine Learning")
    st.write("In this section, you can train and evaluate machine learning models on the bank data.")
    st.write("""
    ### Enter Applicant's Information:
    Provide the necessary details below to predict whether the loan will be approved or not.
    """)

    model = joblib.load("xgb_loan_model.pkl")
    # Applicant input fields
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Annual Income", min_value=0, value=50000)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    person_emp_length = st.number_input("Years of Employment", min_value=0, value=5)
    loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, value=0.2)
    cb_person_default_on_file = st.selectbox("Default on File", ["Y", "N"])
    cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0, value=5)

    if st.button("Predict Loan Approval"):
        # Prepare input as DataFrame
        input_df = pd.DataFrame([[
            person_age, person_income, person_home_ownership,
            person_emp_length, loan_intent, loan_grade, loan_amnt,
            loan_int_rate, loan_percent_income, cb_person_default_on_file,
            cb_person_cred_hist_length
        ]], columns=[
            'person_age', 'person_income', 'person_home_ownership',
            'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
            'cb_person_cred_hist_length'
        ])

        # Apply the same Label Encoding as in training
        columns_to_encode = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
        train_data = pd.read_csv("Loan approval prediction.csv")
        for column in columns_to_encode:
            le = LabelEncoder()
            le.fit(train_data[column])
            input_df[column] = le.transform(input_df[column])

        # Predict (no scaler)
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")


    
# Main Application
def main():
    st.sidebar.markdown(
    "<h2 style='color:#4CAF50;'>🚀 Explore Pages</h2>", 
    unsafe_allow_html=True
)

    page = st.sidebar.radio("Select a page", ["Home", "EDA", "Machine Learning"])

    if page == "Home":
        home()
    elif page == "EDA":
        eda()
    elif page == "Machine Learning":
        machine_learning()

if __name__ == "__main__":
    main()

    




