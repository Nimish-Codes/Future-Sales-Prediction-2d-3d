# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic sales data
np.random.seed(40)

# Generate dates for the last two years
dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')

# Generate synthetic sales data
sales_data = pd.DataFrame({
    'Date': dates,
    'Sales': np.random.randint(5, 200, size=len(dates)) + np.sin(np.arange(len(dates))) * 20,
    'Quantity': np.random.randint(20, 100, size=len(dates)),
    'Product': np.random.randint(1, 5, size=len(dates))
})

# Feature engineering: Extracting useful features
sales_data['Year'] = sales_data['Date'].dt.year
sales_data['Month'] = sales_data['Date'].dt.month
sales_data['Day'] = sales_data['Date'].dt.day

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    sales_data[['Year', 'Month', 'Day']],
    sales_data['Sales'],
    test_size=0.2,
    random_state=42
)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Streamlit App
st.title('Sales Prediction Model and Forecasting')
st.write('This app evaluates a linear regression model for sales prediction and forecasts future sales.')

# Display sample data
st.subheader('Sample Data:')
st.write(sales_data.head(7))

# Display feature engineering results
st.subheader('Feature Engineering:')
st.write(sales_data[['Date', 'Year', 'Month', 'Day']].head(7))

# Evaluate the model
st.subheader('Model Evaluation:')
mse = mean_squared_error(y_test, y_pred)
st.write(f'Mean Squared Error: {mse}')

# Visualize the results
st.subheader('Sales Prediction Visualization:')
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_test.index, y_test, color='black', label='Actual Sales')
ax.plot(X_test.index.values, model.predict(X_test), color='blue', linewidth=3, label='Predicted Sales')
ax.set_title('Sales Prediction')
ax.set_xlabel('Index')
ax.set_ylabel('Sales')
ax.legend()
st.pyplot(fig)

# Forecast future sales
st.subheader('Forecast Future Sales:')
st.write('Enter a future date to forecast sales:')
future_date = st.date_input('Select a future date:', min_value=datetime.today() + timedelta(days=1))

# Extract features from the future date
future_year = future_date.year
future_month = future_date.month
future_day = future_date.day

# Make predictions for the future date
future_sales_prediction = model.predict([[future_year, future_month, future_day]])[0]

# Display the forecasted sales
st.write(f'Forecasted Sales for {future_date}: {future_sales_prediction}')
