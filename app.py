from flask import Flask, request, render_template
from keras.models import load_model
import pandas as pd
import joblib


# Initializing Flask application
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
        # Getting input features from user input
        input_features = request.form.to_dict()

        # Placing the input features into a Pandas dataframe
        features = pd.DataFrame([input_features])
        print(features)

        # Using the loaded data scaler to scale the input features
        scaled_features = min_scaler.transform(features)
        print(scaled_features)

        # Performing prediction using the loaded DNN model
        prediction = model.predict(scaled_features)
        print(prediction)

        # Getting prediction output
        output = round(prediction[0, 0] * 100, 1)
        print(output)

        return render_template('index.html', result='This customer\'s churn probability is {}%'.format(output))
    except Exception as e:
        return render_template('index.html', result='Error: {}'.format(str(e)))


if __name__ == "__main__":
    app.run(debug=True)
