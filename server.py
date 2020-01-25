from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

soc_code_encoder = joblib.load('Joblib Files/soc_code_encoder.joblib')
work_state_encoder = joblib.load('Joblib Files/work_state_encoder.joblib')
employment_days_scaler = joblib.load('Joblib Files/employment_days_scaler.joblib')

classifier = joblib.load('Joblib Files/random_forest.joblib')

dataset = pd.read_excel('soc_name_list.xlsx')
dataset = dataset.filter(['2018 SOC Code', '2018 SOC Title'])
dataset = dataset.drop_duplicates()
dataset = dataset[dataset['2018 SOC Code'].isin(soc_code_encoder.classes_)]

@app.route('/', methods=['POST'])
def get_prediction():
  try:
    content = request.get_json()
    
    content['occupation_code'] = soc_code_encoder.transform([content['occupation_code'].lower()])[0]
    content['worksite_state'] = work_state_encoder.transform([content['worksite_state'].lower()])[0]
    content['employment_duration_days'] = employment_days_scaler.transform([[content['employment_duration_days']]])[0][0]
    
    print(content)
    
    data = [[
      content['occupation_code'],
      content['full_time_position'],
      content['worksite_state'],
      content['employment_duration_days']
    ]]

    prediction = classifier.predict(data)
    probability = np.max(classifier.predict_proba(data))

    label = {True: 'Certified', False: 'Denied'}

    response = jsonify({
      "status": "Prediction made",
      "result": "Prediction: " + label[prediction[0]] + " (" + str(np.round(probability*100, 1)) + "%)"
    })

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 200

  except Exception as error:
    return jsonify({
      "status": "Could not make prediction",
      "error": str(error)
    }), 500

def capitalize(string):
  return ' '.join([s.capitalize() for s in string.split()])

@app.route('/states')
def get_states():
  return jsonify(
    list(
      map(lambda x: {"name": capitalize(x)}, work_state_encoder.classes_)
    )
  )

@app.route('/occupations')
def get_occupation_codes():
  return jsonify(
    list(
      map(lambda x: {
        "name": x[1],
        "code": x[0]
      }, dataset.values)
    )
  )

if __name__ == '__main__':
  app.run(threaded=True, port=5000)