from flask import Flask, request, jsonify
import requests
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

app = Flask(__name__)

# Define the API endpoint for demand prediction
@app.route('/predict_demand', methods=['GET'])
def predict_demand():
    # Load the LSTM model    
    model = load_model('houston2021 (5).h5', compile=False)

    # Define the WeatherAPI endpoint and parameters  
    url = "https://api.weatherapi.com/v1/forecast.json"
    api_key = "79188c0f65b94c27b1592026231302"
    location = "Houston"

    # Send the GET request to the WeatherAPI  
    response = requests.get(url, params={"key": api_key, "q": location, "days": 1}) 

    # Parse the JSON response   
    data = json.loads(response.text)

    # Extract the hourly weather forecast data   
    forecast = data['forecast']['forecastday'][0]['hour']

    # Preprocess the forecast data for model prediction  
    forecast_df = pd.DataFrame(forecast)
    forecast_df['time'] = pd.to_datetime(forecast_df['time'])
    forecast_df = forecast_df.set_index('time')
    forecast_df = forecast_df.resample('1H').ffill()
    forecast_df['conditions'] = pd.factorize([var['text'] for var in forecast_df['condition']])[0]
    forecast_df = forecast_df[['temp_c', 'humidity','gust_kph', 'wind_kph','conditions']]
    sc = MinMaxScaler(feature_range=(1,2))
    forecast_arr_scaled = sc.fit_transform(forecast_df)

    # Reshape the input array to have 2 dimensions
    x_test_scaled = np.reshape(forecast_arr_scaled, (forecast_arr_scaled.shape[0], forecast_arr_scaled.shape[1]))

    # Reshape the input array to have 3 dimensions for model prediction
    x_test_scaled = np.reshape(x_test_scaled, (1, x_test_scaled.shape[0], x_test_scaled.shape[1]))

    preds = model.predict(x_test_scaled[0])

    # Define a new scaler with the desired feature range
    scaler = MinMaxScaler(feature_range=(1,3))

    # Scale the predicted demand value between 1 and 3
    scaled_demand_pred = scaler.fit_transform(np.reshape(preds, (preds.shape[0], 1)))

    # Get the original demand prediction value from the scaled value
    original_demand_pred = scaler.inverse_transform(scaled_demand_pred)[-1][0]

    # Return the original demand prediction as a string
    return jsonify({'demand_pred': str(original_demand_pred)})



if __name__ == '__main__':
    app.run(debug=True)
