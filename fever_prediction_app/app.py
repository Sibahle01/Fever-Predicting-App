from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved components
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders_dict = joblib.load('label_encoder.pkl')

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        
        user_data = {
            'Temperature': [float(request.form['temperature'])],
            'Age': [int(request.form['age'])],
            'Gender': [request.form['gender']],
            'BMI': [float(request.form['bmi'])],
            'Headache': [request.form['headache']],
            'Body_Ache': [request.form['body_ache']],
            'Fatigue': [request.form['fatigue']],
            'Chronic_Conditions': [request.form['chronic_conditions']],
            'Allergies': [request.form['allergies']],
            'Smoking_History': [request.form['smoking_history']],
            'Alcohol_Consumption': [request.form['alcohol_consumption']],
            'Humidity': [float(request.form['humidity'])],
            'AQI': [int(request.form['aqi'])],
            'Physical_Activity': [request.form['physical_activity']],
            'Diet_Type': [request.form['diet_type']],
            'Heart_Rate': [int(request.form['heart_rate'])],
            'Blood_Pressure': [request.form['blood_pressure']],
            'Previous_Medication': [request.form['previous_medication']]
        }
        
        
        input_df = pd.DataFrame(user_data)
        
        
        for column, encoder in encoders_dict.items():
            if column in input_df.columns:
                try:
                    input_df[column] = encoder.transform(input_df[column])
                except ValueError as e:
                    print(f"Error transforming {column}: {str(e)}")
                    # Handle unseen labels by using a default value
                    input_df[column] = encoder.transform([encoder.classes_[0]])
        
        
        numerical_columns = input_df.select_dtypes(include=['float64', 'int64']).columns
        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
        
        
        prediction = model.predict(input_df)[0]
        
        return render_template('result.html', prediction=prediction)
    
    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)