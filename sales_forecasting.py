import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('sales_data.csv')
print("Original Data:")
print(df)

# Convert Date to numeric value for regression
df['Date'] = pd.to_datetime(df['Date'])
df['DateOrdinal'] = df['Date'].map(pd.Timestamp.toordinal)

# Define features and target
X = df[['DateOrdinal']]
y = df['Revenue']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Plot Actual vs Predicted
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.title('Sales Forecasting using Linear Regression')
plt.legend()
plt.show()