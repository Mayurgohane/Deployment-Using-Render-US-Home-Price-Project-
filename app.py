from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained models
with open('ridge_model.pkl', 'rb') as file:
    ridge_model = pickle.load(file)

with open('lasso_model.pkl', 'rb') as file:
    lasso_model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    
    # Standardize the features
    final_features_scaled = scaler.transform(final_features)
    
    # Make predictions
    ridge_prediction = ridge_model.predict(final_features_scaled)
    lasso_prediction = lasso_model.predict(final_features_scaled)
    
    ridge_output = round(ridge_prediction[0], 2)
    lasso_output = round(lasso_prediction[0], 2)
    
    return render_template('index.html', 
                           ridge_prediction_text=f'Ridge Prediction: {ridge_output}',
                           lasso_prediction_text=f'Lasso Prediction: {lasso_output}')

if __name__ == "__main__":
    app.run(debug=True)
