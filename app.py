from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import scipy.stats as stats

app = Flask(__name__)

# Loading my ensemble model
model = joblib.load('ensemble_model.pkl')

# Loading the scaler that was used during training
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Getting user input from the HTML form
        feature1 = float(request.form['movement_reactions'])
        feature2 = float(request.form['mentality_composure'])
        feature3 = float(request.form['passing'])
        feature4 = float(request.form['potential'])
        feature5 = float(request.form['release_clause_eur'])
        feature6 = float(request.form['dribbling'])
        feature7 = float(request.form['wage_eur'])
        feature8 = float(request.form['power_shot_power'])
        feature9 = float(request.form['value_eur'])
        feature10 = float(request.form['lcm'])
        feature11 = float(request.form['cm'])
        feature12 = float(request.form['rcm'])

        # Creating a user input list
        user_inputs = [[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12]]

        # Applying the same scaling as used during training
        user_inputs_scaled = scaler.transform(user_inputs)

        # Making predictions using the ensemble model
        prediction = model.predict(user_inputs_scaled)[0]

        # Calculating the confidence interval
        n = len(model.estimators_)  # Number of estimators in the ensemble
        std_error = prediction.std()
        margin_of_error = std_error * stats.t.ppf((1 + 0.95) / 2, n - 1)  # 95% confidence interval

        # Defining the confidence level (95% in this case)
        confidence_level = 95

        # Displaying the prediction, confidence interval, and confidence level on the web page
        return render_template('index.html', prediction=round(prediction,2), confidence_interval=(round(prediction - margin_of_error,2), round(prediction + margin_of_error,2)), confidence_level=confidence_level)

if __name__ == '__main__':
    app.run(debug=True)
