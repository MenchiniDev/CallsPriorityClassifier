import pickle
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier

features = ['district', 'description', 'incidentLocation', 'Neighborhood', 'PoliceDistrict', 'Census_Tracts', 'year', 'month', 'day', 'hour', 'minute', 'dayofweek']


with open('../models/random_forest_classifier.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)

random_forest_rules = []
for estimator in random_forest_model.estimators_:
    rules = export_text(estimator, feature_names=features)
    random_forest_rules.append(rules)

with open('../models/adaboost_classifier.pkl', 'rb') as f:
    adaboost_model = pickle.load(f)

adaboost_rules = []
for estimator in adaboost_model.estimators_:
    rules = export_text(estimator, feature_names=features)
    adaboost_rules.append(rules)

print("Regole decisionali del modello Random Forest:")
for rules in random_forest_rules:
    print(rules)

print("Regole decisionali del modello Adaboost:")
for rules in adaboost_rules:
    print(rules)