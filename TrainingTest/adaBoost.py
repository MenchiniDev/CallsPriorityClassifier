import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from sklearn.pipeline import Pipeline
#from sklearn.tree import tree
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from utils.save import saveResults

def buildAdaBoost():
    
    df = pd.read_csv('../datasets/balancedDataset.csv', sep=',')
    k = 0
    # Define the features and target variable
    X = df.drop('priority', axis=1)
    y = df['priority']
    class_names=["no Emergency", "low", "medium", "high", "emergency"]

    kf = KFold(n_splits=5,
            shuffle=True,
            random_state=123
            )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    selector = SelectKBest(k=12)
    X_train_selected = selector.fit_transform(X_train, y_train)

    best_model = None

    if not os.path.exists('../models/parameters/abParams.json') or not os.path.exists('../models/adaBoost_model.pkl'):
        print("adaBoost model is being trained...")
        params_directory = "../models/parameters"
        classifier = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), algorithm='SAMME')

        pipe = Pipeline(steps=[('selector', selector),
                               ('classifier', classifier)
                                ])

        param_grid = {
            'classifier__n_estimators': [150, 300],
            'classifier__learning_rate': [2.0, 3.0]
        }

        grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_selected, y_train)

        print("Best Hyperparameters: ", grid_search.best_params_)
        print("Best Accuracy Score: ", grid_search.best_score_)

        best_model = grid_search.best_estimator_
        best_model.fit(X_train_selected, y_train)
        joblib.dump(best_model, '../models/parameters/adaBoost_model.pkl')
        params_path = os.path.join(params_directory, 'abParams.json')
        with open(params_path, 'w') as params_file:
            json.dump(grid_search.best_params_, params_file)

    else:
        print("finded adaBoost model and hyperparameters")
        best_model = joblib.load('../models/adaBoost_model.pkl')
        pipe = Pipeline(steps=[('selector', selector)])
        with open('../models/parameters/abParams.json', 'r') as file:
            hyperparameters = json.load(file)
        best_model.set_params(**hyperparameters)

    # Cross-validation to evaluate the model and save best results    
    best_fscore = 0.0
    for train, val in (kf.split(X, y)):
        print(f'FOLD {k}')
        X_tr = X.to_numpy()[train]
        y_tr = y.to_numpy()[train]
        X_val = X.to_numpy()[val]
        y_val = y.to_numpy()[val]
        best_model.fit(X_tr, y_tr)
        y_pred = best_model.predict(X_val)
        cr = classification_report(y_val, y_pred, output_dict=True)
        if float(cr["macro avg"]['f1-score']) > best_fscore:
            best_fscore = cr["macro avg"]['f1-score']
            best_val_cr = cr
            best_conf_matrix = confusion_matrix(y_val, y_pred)
        k += 1
        # Save the results
        result_path = "../models/results/bestADAmatrix.png"
        plt.figure(figsize=(8, 6))
        sns.heatmap(best_conf_matrix, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.savefig(f'../models/results/adaConfusionMatrix{k}.png')
        plt.show()

    
    plt.figure(figsize=(15, 15))
    plt.savefig("../models/results/adaBoost_treeplot.png", format='png', bbox_inches="tight", dpi=100)

# vuidl = buildAdaBoost()