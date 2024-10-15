from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd
import re

# Load your pre-trained model (modify the path to your model file)
model = joblib.load('C:\\Users\\abdul\\Desktop\\deployment\\trained_model.pkl')  # Update with your model path

# Load the dataset that will be used for predictions
data = pd.read_csv('C:\\Users\\abdul\\Downloads\\processed_data.csv')  # Update with your dataset path

# Normalize location names by removing extra spaces and converting to lowercase
def normalize_location(location):
    return re.sub(r'\s+', ' ', location).strip().lower()

# Initialize the Flask app
app = Flask(__name__)

# Function to get data for a specific location and timestamp
def get_data_for_location(location, timestamp):
    # Normalize the location for filtering
    normalized_location = normalize_location(location)
    
    # Normalize dataset locations for comparison
    data['normalized_location'] = data['location'].apply(normalize_location)
    
    # Filter the data based on location and timestamp
    filtered_data = data[(data['normalized_location'] == normalized_location) & (data['timestamp'] == timestamp)]
    
    # Debug: Print number of rows retrieved after filtering
    print(f"Filtered data has {len(filtered_data)} rows for location {location} and timestamp {timestamp}")

    # Select numeric columns
    numeric_columns = filtered_data.select_dtypes(include=[np.number]).columns[:22]

    # If no data is found, relax filtering condition to include all data for the location
    if len(filtered_data) == 0:
        print(f"No exact match found for location {location} and timestamp {timestamp}, attempting relaxed filtering...")
        filtered_data = data[data['normalized_location'] == normalized_location]
        print(f"Filtered data after relaxed filtering for location {location} has {len(filtered_data)} rows.")

    # If still no data, return padded data with mean values of dataset
    if len(filtered_data) == 0:
        print(f"No data found for location {location}. Using padded mean values.")
        column_means = data[numeric_columns].mean().values
        data_subset = np.tile(column_means, (10, 1))  # Create 10 rows of mean values
    else:
        # Pad or slice to match the input shape (at least 10 rows needed)
        if len(filtered_data) < 10:
            data_subset = filtered_data[numeric_columns].values
            column_means = filtered_data[numeric_columns].mean().values
            padding = np.tile(column_means, (10 - len(data_subset), 1))
            data_subset = np.vstack([data_subset, padding])
        else:
            data_subset = filtered_data[numeric_columns].iloc[:10].values

    # Debug: Print the shape of data being used for prediction
    print(f"Input data shape after padding or selection for location {location}: {data_subset.shape}")

    # Reshape the data to match the input shape expected by the model (batch_size, 10, 22)
    input_data = data_subset.reshape(1, 10, 22)  # Add batch dimension (1)
    return input_data

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    req_data = request.get_json(force=True)
    
    # Extract locations and timestamp from the request
    locations = req_data.get('locations')  # Should be a list of 5 locations
    timestamp = req_data.get('timestamp')  # Should be a single timestamp

    # Validate the input
    if len(locations) != 5 or not timestamp:
        return jsonify({'error': 'Please provide exactly 5 locations and a timestamp.'}), 400
    
    prediction_results = []
    
    # Loop over each location to get data and make predictions
    for location in locations:
        # Get the input data for the current location
        input_data = get_data_for_location(location, timestamp)
        
        if input_data is None:
            predicted_jam_factor = 0  # Assign zero if no data is found, though this should not happen with padding
        else:
            # Make prediction for the current location
            try:
                prediction = model.predict(input_data)  # Ensure input_data has the right shape
                
                # Convert prediction to a numerical value
                if isinstance(prediction, (list, np.ndarray)):
                    predicted_jam_factor = float(prediction[0])
                else:
                    predicted_jam_factor = float(prediction)

            except Exception as e:
                return jsonify({'error': f'Model prediction failed for location {location}: {str(e)}'}), 500

        # Append prediction for the current location
        prediction_results.append({'location': location, 'predicted_jamFactor': max(0, predicted_jam_factor)})  # Clip negative values to zero
    
    # Sort the predictions by the lowest jam factor
    sorted_predictions = sorted(prediction_results, key=lambda x: x['predicted_jamFactor'])
    
    # Send the sorted prediction as a JSON response
    return jsonify({'sorted_predictions': sorted_predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Start the Flask server
