# app.py

from flask import Flask, request, render_template
import pickle
import numpy as np

# Load trained model
model_path = 'rock_mine_model.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Create flask app
app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    # Get input values from form
    input_features = [float(x) for x in request.form.values()]

    # Convert into numpy array
    final_features = np.array(input_features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(final_features)

    # Output
    output = 'Rock' if prediction[0] == 'R' else 'Mine'

    return render_template(
        'index.html',
        prediction_text='Prediction: {}'.format(output)
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True)