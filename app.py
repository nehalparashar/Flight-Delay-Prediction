from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model, scaler, encoder, and columns
model = joblib.load('flight_delay_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
columns = joblib.load('columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Input
        departure_delay = float(request.form['departure_delay'])
        airline = request.form['airline']
        origin = request.form['origin']
        destination = request.form['destination']
        scheduled_departure = int(request.form['scheduled_departure'])
        taxi_out = float(request.form['taxi_out'])

        # Input DF
        input_df = pd.DataFrame([{
            'AIRLINE': airline,
            'ORIGIN_AIRPORT': origin,
            'DESTINATION_AIRPORT': destination,
            'DEPARTURE_DELAY': departure_delay,
            'SCHEDULED_DEPARTURE': scheduled_departure,
            'TAXI_OUT': taxi_out
        }])

        # Encode
        encoded = encoder.transform(input_df[['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

        # Combine
        numeric_df = input_df[['DEPARTURE_DELAY', 'SCHEDULED_DEPARTURE', 'TAXI_OUT']]
        final_input = pd.concat([numeric_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # Align with training column order
        final_input = final_input.reindex(columns=columns, fill_value=0)

        # Scale
        scaled_input = scaler.transform(final_input)

        # Predict
        prediction = model.predict(scaled_input)[0]
        return render_template('result.html', prediction=round(prediction, 2))

    except Exception as e:
        return f"<h3>Error: {str(e)}</h3>"

if __name__ == '__main__':
    app.run(debug=True) 

if __name__ == '__main__':
    print("Launching Flask app...")

    print("Loading model...")
    model = joblib.load('flight_delay_model.pkl')
    print("Model loaded.")

    print("Loading scaler...")
    scaler = joblib.load('scaler.pkl')
    print("Scaler loaded.")

    print("Loading encoder...")
    encoder = joblib.load('encoder.pkl')
    print("Encoder loaded.")

    print("Loading columns...")
    columns = joblib.load('columns.pkl')
    print("All components loaded!")

    app.run(debug=True)

