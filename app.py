
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("property_price_model.pkl")
preprocessor = joblib.load("property_preprocessor.pkl")

st.set_page_config(
    page_title="Property Price Prediction",
    page_icon="🏠"
)

st.title("Property Sale Price Prediction")
st.write("Enter the property details below.")

tower_number = st.selectbox(
    "Tower Number",
    list(range(1, 21))
)

unit_category = st.selectbox(
    "Unit Category",
    ["1 BHK", "2 BHK", "3 BHK", "4 BHK"]
)

listing_status = st.selectbox(
    "Listing Status",
    ["Available", "Sold", "Under Construction"]
)

floor_area = st.number_input(
    "Floor Area (sq ft)",
    min_value=100.0,
    value=1000.0
)


if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "tower_number": tower_number,
        "unit_category": unit_category,
        "listing_status": listing_status,
        "floor_area_sqft": floor_area
    }])

    try:
        processed_data = preprocessor.transform(input_data)
        predicted_price = model.predict(processed_data)[0]

        st.success(
            f"Predicted Property Price: {predicted_price:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

