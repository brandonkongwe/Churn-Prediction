from flask import Flask, request, render_template
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import joblib


app = Flask(__name__)

# Loading the pre-trained deep neural network
model = load_model('dnn_model.keras')

# Loading the scaler to scale the data
min_scaler = joblib.load('min_scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = request.form.to_dict()
        features = pd.DataFrame([input_features])
        print(features)

        scaled_features = min_scaler.transform(features)
        print(scaled_features)

        prediction = model.predict(scaled_features)
        print(prediction)

        output = round(prediction[0, 0] * 100, 1)
        print(output)

        # output = 'Yes' if prediction[0, 0] > 0.5 else 'No'
        # print(output)

        return render_template('index.html', result='Churn Probability: {}%'.format(output))
    except Exception as e:
        return render_template('index.html', result='Error: {}'.format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)
