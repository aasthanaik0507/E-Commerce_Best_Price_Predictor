import streamlit as st
import pickle
import pandas as pd
import datetime
import calendar
import numpy as np

def load_model(model_file):
    return pickle.load(open(model_file, 'rb'))

def get_week_info(day):
    # Convert NumPy int64 to Python integer
    day = int(day)

    # Create a datetime object for the given day in the current year
    date_object = datetime.datetime(datetime.datetime.now().year, 1, 1) + datetime.timedelta(day - 1)

    # Get the ISO week number and weekday international standardized organisation 
    week_number = date_object.isocalendar()[1]
    # weekday = date_object.weekday()

    # Get the month and year
    month = date_object.month
    # year = date_object.year

    # Get the month name
    month_name = calendar.month_name[month]

    return week_number, month_name

def predict_price(dataset_file, model_file):
    model = load_model(model_file)

    # Load the dataset based on the selected product
    df = pd.read_csv(dataset_file)

    # Preparation of the data
    X = df['Days'].values.reshape(-1, 1)
    y = df['Price'].values.reshape(-1, 1)
    description = df['Description']
    rating = df['Rating']
    comment = df['Comment']
    available_on = df['AvailableOn']

    # Find the best price and corresponding day for the selected product
    best_price_index = y.argmin()
    best_day = X[best_price_index][0]
    best_price = y[best_price_index][0]
    best_description = description[best_price_index]
    best_available_on = available_on[best_price_index]

    # Check if the description is NaN, if so, fallback to the first line description
    if pd.isnull(best_description):
        best_description = description.iloc[0]

    best_review = comment[best_price_index]
    best_rating = rating[best_price_index]

    # Get week and month information for the best day
    week_number, month_name = get_week_info(best_day)

    return best_price, week_number, month_name, best_description, best_review, best_rating, best_available_on

def main():
    st.title("Ecommerce Best Price Predictor")

    products = {
        "Perfect Homes Atiu Metal 4 Seater Dining Set": {
            "dataset_file": "Dinning set - flipkart_reviews.csv",
            "model_file": "diningset.sav"
        },
        "Poco M4": {
            "dataset_file": "mobile.csv",
            "model_file": "mobile.sav"
        },
        "LG 7kg Washing Machine": {
            "dataset_file": "washingmachine.csv",
            "model_file": "washingmachine.sav"
        },
        "LG SmartTV": {
            "dataset_file":"lgtv.csv",
            "model_file": "tv.sav"
        },
        "Asus Tuf F15 Gaming Laptop": {
            "dataset_file":"laptop.csv",
            "model_file": "tv.sav"
        }
    }

    product = st.selectbox("Select a product:", list(products.keys()))

    if st.button("Predict"):
        dataset_file = products[product]["dataset_file"]
        model_file = products[product]["model_file"]
        best_price, week_number, month_name, best_description, best_review, best_rating, best_available_on = predict_price(dataset_file, model_file)

        st.write(f"Best Price for {product}: ₹ {best_price}")
        # st.write("Best Day: ", best_day)
        st.write("Week of the Year: ", week_number)
        st.write("Month: ", month_name)
        st.write("Description: ", best_description)
        st.write("Top Review: ", best_review)
        st.write("Rating: ", best_rating)
        st.write("Available On: ", best_available_on)

if __name__ == "__main__":
    main()
