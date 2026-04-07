from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib 
import numpy as np


app = Flask(__name__)
CORS(app)


# Load the trained model
model = joblib.load('model/diabetes_model.pkl')
scaler = joblib.load('model/scaler.pkl')


#home route
@app.route('/')
def home():
    return jsonify({"status": "API is running ✅"})

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Validate input data
        required = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        # Check for missing fields
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        features = [data[f] for f in required]
        input_array = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        probability = round(float(model.predict_proba(input_scaled)[0][1]) * 100, 1)

        return jsonify({"prediction": int(prediction), "probability": probability,"result": "Diabetic" if prediction == 1 else "Non-Diabetic"})
           
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)