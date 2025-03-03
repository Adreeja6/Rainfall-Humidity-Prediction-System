import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Create 'model' directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Load data from CSV
df = pd.read_csv('C:\Users\avijit\Desktop')  # Replace with your CSV file

# Splitting data for Random Forest model (Rainfall prediction)
X = df[['windspeed', 'winddirection', 'cloud', 'pressure']]
y_rainfall = df['rainfall']
X_train, X_test, y_train, y_test = train_test_split(X, y_rainfall, test_size=0.2, random_state=42)

# Train Random Forest for Rainfall
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Saving the Random Forest model
with open('model/rainfall_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Splitting data for Linear Regression model (Humidity prediction)
y_humidity = df['humidity']
X_train, X_test, y_train, y_test = train_test_split(X, y_humidity, test_size=0.2, random_state=42)

# Train Linear Regression for Humidity
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Saving the Linear Regression model
with open('model/humidity_lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
