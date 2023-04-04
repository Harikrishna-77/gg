import requests
import json
import numpy as np
from flask import Flask, request, jsonify
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import load_model
 

app = Flask(__name__) 

print('a')

# Define the API endpoint for demand prediction
@app.route('/predict_demand', methods=['GET'])
def predict_demand():    
  # Load the LSTM model
  model = load_model('houston2021.h5')
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
  forecast_df['dummy'] = 0
  forecast_df['time'] = pd.to_datetime(forecast_df['time'])
  forecast_df = forecast_df.set_index('time')
  forecast_df = forecast_df.resample('1H').ffill()
  forecast_df['conditions'] = pd.factorize([var['text'] for var in forecast_df['condition']])[0]
  forecast_df = forecast_df[['temp_c', 'humidity','gust_kph', 'wind_kph','conditions']]
  sc = MinMaxScaler(feature_range=(0,1))
  forecast_arr_scaled = sc.fit_transform(forecast_df)
  x_test_scaled = forecast_arr_scaled[-336:, :]
  x_test_scaled = np.reshape(x_test_scaled, (x_test_scaled.shape[0], 1, x_test_scaled.shape[1]))
  preds = model.predict(x_test_scaled)
  demand_pred = sc.inverse_transform(preds)[-1][0]
  return jsonify({'demand_pred': str(demand_pred)}) 
    
if __name__ == '__main__':    
    app.run(debug=True) 
