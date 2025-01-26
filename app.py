from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open('study_grade_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get study hours from the form
    study_hours = float(request.form['study_hours'])
    
    # Predict the grade
    prediction = model.predict(np.array([[study_hours]]))[0]
    
    # Return the result to the webpage
    return render_template('index.html', prediction_text=f'Predicted Grade: {round(prediction, 2)}')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
