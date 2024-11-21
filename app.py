from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

# Enable CORS for all routes
app = Flask(__name__)
CORS(app)

# Load the saved model and scaler (if you have one)
with open('best_model__.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Flask app is running!"})

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict_thyroid_disease():
    try:
        # Get data from request
        data = request.get_json()

        # Define the required fields based on the selected features
        required_fields = [
            'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick', 
            'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
            'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured', 'T3_measured',
            'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 
            'age', 'sex', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG'
        ]

        # Ensure that all 28 required fields are present in the data
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing fields'}), 400

        # Extract the relevant features from the input data
        features = np.array([[ 
            data['on_thyroxine'], 
            data['query_on_thyroxine'], 
            data['on_antithyroid_medication'], 
            data['sick'], 
            data['pregnant'], 
            data['thyroid_surgery'], 
            data['I131_treatment'], 
            data['query_hypothyroid'], 
            data['query_hyperthyroid'], 
            data['lithium'], 
            data['goitre'], 
            data['tumor'], 
            data['hypopituitary'], 
            data['psych'], 
            data['TSH_measured'], 
            data['T3_measured'], 
            data['TT4_measured'], 
            data['T4U_measured'], 
            data['FTI_measured'], 
            data['TBG_measured'],
            data['age'],        # Add the additional features here
            data['sex'], 
            data['TSH'], 
            data['T3'], 
            data['TT4'], 
            data['T4U'], 
            data['FTI'], 
            data['TBG']
        ]])

        # Proceed with prediction as usual
        prediction = model.predict(features)[0]

        # Return the result based on prediction
        if prediction == 1:  # Assuming model returns 1 for disease and 0 for no disease
            return jsonify({'prediction': 'Thyroid disease detected'}), 200
        else:
            return jsonify({'prediction': 'No thyroid disease'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
