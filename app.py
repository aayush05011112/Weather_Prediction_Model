from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from model import predict_weather
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        input_data = request.form.to_dict()
        
        # Convert string values to appropriate types
        for key in input_data:
            if key not in ['month', 'day']:  # These are handled separately
                try:
                    input_data[key] = float(input_data[key])
                except ValueError:
                    input_data[key] = 0  # Default value if conversion fails
        
        # Add month and day if not provided
        if 'month' not in input_data:
            input_data['month'] = 6  # Default month (June)
        else:
            input_data['month'] = int(input_data['month'])
            
        if 'day' not in input_data:
            input_data['day'] = 15  # Default day
        else:
            input_data['day'] = int(input_data['day'])
        
        # Make prediction
        prediction = predict_weather(input_data)
        
        result = 'Rain' if prediction == 1 else 'No rain'
        return jsonify({'prediction': result, 'status': 'success'})
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

if __name__ == '__main__':
    app.run(debug=True)