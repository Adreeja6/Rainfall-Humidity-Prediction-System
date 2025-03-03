from flask import Flask, render_template, request, jsonify 
import joblib
import numpy as np

# Load models
loaded_lr_model = joblib.load('models/lr_model.pkl')
loaded_model = joblib.load('models/model.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    windspeed = float(request.form['windspeed'])
    winddirection = float(request.form['winddirection'])
    cloud = float(request.form['cloud'])
    pressure = float(request.form['pressure'])

    # Prepare the input data for the Random Forest model
    features = np.array([[windspeed, winddirection, cloud, pressure]])

    # Predict Rainfall using Random Forest
    rainfall_prediction = loaded_model.predict(features)[0]  # Extract single prediction
    
    # Convert 0/1 to "No"/"Yes"
    rainfall = "Yes" if rainfall_prediction == 1 else "No"

    # Include rainfall as the fifth feature for the Linear Regression model
    features_with_rainfall = np.array([[windspeed, winddirection, cloud, pressure, rainfall_prediction]])

    # Predict Humidity using Linear Regression
    humidity_prediction = loaded_lr_model.predict(features_with_rainfall)
    humidity = round(humidity_prediction[0], 2)

    return render_template('index.html', rainfall=rainfall, humidity=humidity)

if __name__ == '__main__':
    app.run(debug=True)
