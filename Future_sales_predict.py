import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Generate synthetic sales data
np.random.seed(40)
dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
sales_data = pd.DataFrame({
    'Date': dates,
    'Sales': np.random.randint(5, 200, size=len(dates)) + np.sin(np.arange(len(dates))) * 20,
    'Quantity': np.random.randint(20, 100, size=len(dates)),
    'Product': np.random.randint(1, 5, size=len(dates))
})

# Feature engineering: Extract useful features
sales_data['Year'] = sales_data['Date'].dt.year
sales_data['Month'] = sales_data['Date'].dt.month
sales_data['Day'] = sales_data['Date'].dt.day

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    sales_data[['Year', 'Month', 'Day']], sales_data['Sales'], test_size=0.2, random_state=42
)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test set and evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Display results
st.title("Sales Prediction and Forecast")

# Show code
st.code(open('sales_forecast.py').read(), language='python')

# Show head of sales data
st.header("Sales Data")
st.dataframe(sales_data.head(7))

# Show Mean Squared Error
st.metric(label="Mean Squared Error", value=mse)

# Show 2D sales prediction plot
st.plotly_chart(plt.figure(figsize=(10, 6)))
plt.scatter(X_test.index, y_test, color='black', label='Actual Sales')
plt.plot(X_test.index, y_pred, color='blue', linewidth=3, label='Predicted Sales')
plt.title('Sales Prediction')
plt.xlabel('Index')
plt.ylabel('Sales')
plt.legend()
st.pyplot()

# Show 3D sales prediction plot
st.plotly_chart(fig)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test['Year'], X_test['Month'], y_test, color='black', label='Actual Sales')
ax.scatter(X_test['Year'], X_test['Month'], y_pred, color='blue', label='Predicted Sales')
ax.set_xlabel('Year')
ax.set_ylabel('Month')
ax.set_zlabel('Sales')
ax.set_title('3D Sales Prediction')
plt.legend()
st.pyplot()

# Forecast message
st.info("Now you can use this trained model to forecast future sales by providing future dates as input features!")

