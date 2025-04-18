TOPIC: U.S. DOMESTIC FLIGHT DELAY PREDICTION USING MACHINE LEARNING
This project predicts flight arrival delays based on various features like airline, departure time, origin/destination airports, and more. It uses machine learning and is deployed as a web app using Flask.

PROJECT STRUCTURE

flight_delay_prediction
│
├── flight_delay_project
│   ├── training_data.ipynb          # Feature selection & final training
│   ├── app.py                       # Flask web app backend
│   ├── flight_delay_model.pkl       # Trained ML model
│   ├── scaler.pkl                   # StandardScaler for preprocessing
│   ├── templates
│   │   └── index.html               # Web UI for user input
│   ├── static
│   │   └── style.css                # CSS file for styling the web app
│   └── flights.csv                  # Sample data used in app



FEATURES

- Clean data preprocessing pipeline
- Exploratory data analysis (EDA)
- Trained with `RandomForestRegressor`
- One-hot encoding + scaling
- Interactive web app with dropdown inputs for airports & airlines
- Predicts estimated arrival delay in minutes


TOOLS & LIBRARIES

- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- Flask (for web app)
- Jupyter Notebook (for development)

HOW TO RUN THE APP LOCALLY

1. Clone the repo:

   ```bash
   git clone https://github.com/nehalparashar/Flight-Delay-Prediction.git
   cd flight-delay-prediction/flight_delay_project

INSTALL THE DEPENDENCIES
pip install -r requirements.txt

RUN THE APP
python app.py

Open your browser and go to http://127.0.0.1:5000/ to use the app.

SAMPLE PREDICTION

Enter values like:
Airline: Delta Air Lines Inc.
Origin Airport: JFK
Destination Airport: LAX
Departure Time: 10:00 AM
Taxi Out Time: 15 minutes
Departure Delay: 5 minutes

Click Predict to get the estimated arrival delay.

NOTES-

Model trained on 1000-record sample for faster processing.
You can easily extend the dataset and retrain the model for higher accuracy.
Feature encoding and scaling are preserved using .pkl files.

LIVE DEPLOYMENT COMING SOON
Planning to deploy this on Render / Railway / Heroku — stay tuned!


