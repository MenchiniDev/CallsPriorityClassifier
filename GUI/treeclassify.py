import joblib
import pandas as pd
import json
from datetime import datetime

dataFrame = pd.read_csv('../datasets/balancedDataset.csv')

with open('../TrainingTest/utils/label_mappings.json', 'r') as f:
    label_mappings = json.load(f)

def get_mode_value(topic):
    return dataFrame[topic].mode()[0]

def convert_to_encoded(value, topic):
    reverse_mappings = {v: k for k, v in label_mappings[topic].items()}
    return reverse_mappings.get(value, get_mode_value(topic))

def tree_classify(date, location, city, description, clf):
    data = pd.DataFrame({
        'date': [date], 
        'incidentLocation': [location], 
        'district': [city], 
        'description': [description]
    })

    Neighborhood = get_mode_value('Neighborhood')
    PoliceDistrict = get_mode_value('PoliceDistrict')
    Census_Tracts = get_mode_value('Census_Tracts')
    hour = get_mode_value('hour')
    minute = get_mode_value('minute')

    data['year'] = pd.to_datetime(data['date']).dt.year
    data['month'] = pd.to_datetime(data['date']).dt.month
    data['day'] = pd.to_datetime(data['date']).dt.day
    data['dayofweek'] = pd.to_datetime(data['date']).dt.dayofweek
    data['hour'] = hour
    data['minute'] = minute
    data['Neighborhood'] = Neighborhood
    data['PoliceDistrict'] = PoliceDistrict
    data['Census_Tracts'] = Census_Tracts

    data['description'] = convert_to_encoded(data['description'].values[0], 'description')
    data['incidentLocation'] = convert_to_encoded(data['incidentLocation'].values[0], 'incidentLocation')
    data['district'] = convert_to_encoded(data['district'].values[0], 'district')

    dfpred = data[['district', 'description', 'incidentLocation', 'Neighborhood', 'PoliceDistrict', 'Census_Tracts', 
                      'year', 'month', 'day', 'hour', 'minute', 'dayofweek']]

    print("DataFrame for prediction:")
    print(dfpred)

    # Caricamento del modello
    if clf == 'Adaboost':
        classifier = joblib.load('../models/adaBoost_model.pkl')
    elif clf == 'DecisionTree':
        classifier = joblib.load('../models/random_forest_classifier.pkl')
    else:
        raise ValueError("Classifier not recognized. Please use 'Adaboost' or 'DecisionTree'.")

    prediction = classifier.predict_proba(dfpred)
    print("Prediction:", prediction)
    return prediction

# try:
#     prediction = tree_classify("2021-12-12", "Via Diotisalvi", "pisa", "HIT AND RUN", "Adaboost")
#     print("Prediction:", prediction)
# except Exception as e:
#     print("Error:", e)
