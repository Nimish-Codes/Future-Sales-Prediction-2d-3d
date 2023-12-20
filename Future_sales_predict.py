# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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

# Make predictions on the test set
y_pred = model.predict(X_test)

# Streamlit App
st.title('Sales Prediction Model Evaluation')
st.write('This app evaluates a linear regression model for sales prediction.')

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
ax.plot(X_test.index.values, y_pred, color='blue', linewidth=3, label='Predicted Sales')
ax.set_title('Sales Prediction')
ax.set_xlabel('Index')
ax.set_ylabel('Sales')
ax.legend()
st.pyplot(fig)

# Visualize the results in 3D
st.subheader('3D Sales Prediction Visualization:')
fig_3d = plt.figure(figsize=(12, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.scatter(X_test['Year'], X_test['Month'], y_test, color='black', label='Actual Sales')
ax_3d.scatter(X_test['Year'], X_test['Month'], y_pred, color='blue', label='Predicted Sales')
ax_3d.set_xlabel('Year')
ax_3d.set_ylabel('Month')
ax_3d.set_zlabel('Sales')
ax_3d.set_title('3D Sales Prediction')
ax_3d.legend()
st.pyplot(fig_3d)

# Provide information about using the trained model for forecasting
st.subheader('Forecasting Future Sales:')
st.write('Now you can use the trained model to forecast future sales by providing future dates as input features.')
