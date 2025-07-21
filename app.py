from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import sklearn
import serial
import time

from joblib import load

import numpy as np
import pickle



from sklearn.preprocessing import MinMaxScaler
import threading
import time
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
# Define a dictionary for mapping attack categories to more descriptive labels
label_mapping = {
    'Generic': 'a',
    'worms': 'b',
    'Analysis': 'c',
    'Fuzzer': 'd',
    'Exploits': 'e',
    'Normal': 'f',
    'Backdoor': 'g',
    'Reconnaissance': 'h',
    'Dos': 'i'
}
# Load the Random Forest classifier model
with open('model_.pkl', 'rb') as file:
    model = pickle.load(file)

import numpy as np

# Load the existing attack categories from the npy file
attack_categories = np.load("classes.npy", allow_pickle=True)

# Display the current categories
print("Current categories:", attack_categories)


# Global variables to control the prediction process and store results
stop_thread = False
input_data = []
predicted_data = []
file_uploaded = False  # Flag to track file upload status


# Function to preprocess input data
def preprocess_input(X_test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_test_scaled = scaler.fit_transform(X_test)
    return X_test_scaled


# Background prediction process
def predict_data(X_test):
    global stop_thread, input_data, predicted_data, file_uploaded
    input_data = []
    predicted_data = []
    file_uploaded = True  # Set file upload flag to True

    # Preprocess the input data
    X_test_processed = preprocess_input(X_test)

    # Predict all data
    y_pred = model.predict(X_test_processed)

    # Convert numeric predictions to attack types
    attack_types = [attack_categories[int(pred)] for pred in y_pred]

    # Collect input and predictions
    for i in range(len(attack_types)):
        if stop_thread:
            break
        input_data.append(list(X_test.iloc[i].values))
        predicted_data.append(attack_types[i])
        label = label_mapping.get(attack_types[i], 'unknown')
        print(f" {label}")
       # ser = serial.Serial('COM10', 9600)
        time.sleep(0.2)
       # label=label+'\n'
       # ser.write(label.encode())
        #time.sleep(3)
        #data = ser.readline().decode().strip()
        print(data)
      #  ser.close()
        #time.sleep(2)  # Small delay to simulate data flow


@app.route('/')
def index():
    global file_uploaded
    return render_template('index.html', file_uploaded=file_uploaded)


@app.route('/upload', methods=['POST'])
def upload_file():
    global stop_thread, file_uploaded
    stop_thread = False
    file_uploaded = False  # Reset file upload flag

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400  # Return error response with status code 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400  # Return error response with status code 400

    df = pd.read_csv(file)

    # Extract relevant columns for prediction
    X_test = df[['dttl', 'swin', 'dwin', 'tcprtt', 'synack', 'ackdat',
                 'proto_tcp', 'proto_udp', 'service_dns', 'state_CON', 'state_FIN',
                 'ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports']]

    # Start the prediction in a background thread
    threading.Thread(target=predict_data, args=(X_test,)).start()

    return render_template('index.html', file_uploaded=True)


@app.route('/data')
def data():
    global input_data, predicted_data
    return jsonify({'input_data': input_data, 'predicted_data': predicted_data})


@app.route('/stop', methods=['POST'])
def stop():
    global stop_thread
    stop_thread = True
    return '', 204


@app.route('/bargraph')
def bargraph():
    global predicted_data

    attack_counts = pd.Series(predicted_data).value_counts()

    # Plot the bar graph
    plt.figure(figsize=(10, 6))
    attack_counts.plot(kind='bar')
    plt.xlabel('Attack Type')
    plt.ylabel('Count')
    plt.title('Attack Type Distribution')
    plt.xticks(rotation=45)

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True, port=9000)