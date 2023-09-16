import streamlit as st
import pandas as pd
import joblib
import warnings
import pycountry

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the Model
model = joblib.load("final_model/income_model.pkl")

# Load the preprocessor
preprocessor = joblib.load("final_model/preprocessor.joblib")

# Define the main function
def main():
    # Set page title and layout
    st.set_page_config(page_title="Income Inequality Prediction App", layout="wide")

    st.title("Income Inequality Prediction App")

    # Dropdown for selecting a country
    # Get a list of all countries
    all_countries = list(pycountry.countries)
    # Extract country names from the country objects
    country_names = [country.name for country in all_countries]
    selected_country = st.selectbox("Select a Country", country_names, index=0)

    # Define form inputs and initial values
    age = st.slider("Age", 0, 92, step=1)
    
    citizenship_options = [
        "Foreign born - Not a citizen of U.S.",
        "Foreign born - U.S. citizen by naturalization",
        "Native-born abroad of American parent(s)",
        "Native-born in Puerto Rico or U.S. Outlying"]
    
    citizenship = st.selectbox("Citizenship", citizenship_options, index=0)

    gender = st.selectbox("Gender", ["Female", "Male"], index=0)

    tax_status = st.selectbox("Tax Status", [
        "Nonfiler",
        "Joint both under 65",
        "Single",
        "Joint both 65+",
        "Head of household",
        "Joint one under 65 & one 65+"], index=0)
    
    employment_stat = st.slider("Employment Status", 0, 2, step=1)

    industry_code = st.slider("Industry Code", 0, 52, step=1)

    wage_per_hour = st.slider("Wage Per hour", 0, 10000, step=1)

    mig_year = st.slider("Migration Year", 94, 96, step=1)

    stocks_status = st.slider("Stocks Status", 0, 10000, step=1)

    # Define the form
    with st.form("Income_inequality_form"):
        # Add submit button inside the form
        submit_button = st.form_submit_button("Submit")

        # If submit button is clicked
        if submit_button:
            # Create a DataFrame with the selected features
            input_data = pd.DataFrame({
                "age": [age],
                "selected_country": [selected_country],
                "citizenship": [citizenship],
                "gender": [gender],
                "tax status": [tax_status],
                "employment stat": [employment_stat],
                "industry_code": [industry_code],
                "wage_per_hour": [wage_per_hour],
                "mig_year": [mig_year],
                "stocks_status": [stocks_status]
            })

            # Encode the categorical variables if required
            categorical_cols = input_data.select_dtypes(include=["object"]).columns
            for col in categorical_cols:
                input_data[col] = input_data[col].astype("category").cat.codes

            # Make the prediction
            prediction = model.predict(input_data)

            # Map the prediction to human-readable labels
            income_mapping = {0: "Below Limit", 1: "Above Limit"}
            predicted_income_limit = income_mapping.get(prediction[0], "Unknown")

            # Show the prediction
            st.subheader("Prediction:")
            st.write("The predicted income limit of the individual is:", predicted_income_limit)

# Run the main function
if __name__ == "__main__":
    main()
