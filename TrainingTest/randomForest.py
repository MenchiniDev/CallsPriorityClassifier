import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import pipeline, tree
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from sklearn.pipeline import Pipeline
import seaborn as sns

from utils.save import saveResults


def buildRandomForest():
    df = pd.read_csv('../datasets/balancedDataset.csv', sep=',')
    # Define the features and target variable
    X = df.drop('priority', axis=1)
    y = df['priority']
    class_names=["no Emergency", "low", "medium", "high", "emergency"]
    k=0;

    kf = KFold(n_splits=5,
            shuffle=True,
            random_state=123
            )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    selector = SelectKBest(k=8)
    X_train_selected = selector.fit_transform(X_train, y_train)

    best_model = None

    if not os.path.exists('../models/parameters/rfParams.json') or not os.path.exists('../models/random_forest_classifier.pkl'):
        classifier = RandomForestClassifier(random_state=42, n_jobs=-1, verbose=2)

        pipe = Pipeline(steps=[
            ('selector', selector),
            ('classifier', classifier),
            ])

        param_grid = {
        'classifier__n_estimators': [200, 300],
        'classifier__max_features': [None, 'sqrt'],
        'classifier__max_depth': [ 30, 50],
        'classifier__min_samples_split': [10, 20],
        }

        grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=2, error_score='raise')
        grid_search.fit(X_train_selected, y_train)

        best_params = grid_search.best_params_
        print("Best parameters found for Random Forest: ", best_params)
        best_model = grid_search.best_estimator_
        joblib.dump(best_model, '../models/random_forest_classifier.pkl')

        params_path = '../models/parameters/best_params.json'
        with open(params_path, 'w') as f:
            json.dump(best_params, f)
        print("Best parameters saved successfully!")
    else:
        print("finded RandomForest model and hyperparameters")
        best_model = joblib.load('../models/random_forest_classifier.pkl')
        pipe = Pipeline(steps=[('selector', selector)])
        with open('../models/parameters/rfParams.json', 'r') as file:
            hyperparameters = json.load(file)
            best_model.set_params(**hyperparameters)

        # Cross-validation to evaluate the model and save best results
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
        saveResults("adaBoost", best_conf_matrix, best_val_cr, "validation")

    
    plt.figure(figsize=(15, 15))
    tree.plot_tree(classifier, feature_names=X, class_names=class_names, filled=True, fontsize=6)
    plt.savefig("../models/results/adaBoost_treeplot.png", format='png', bbox_inches="tight", dpi=100)


    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.savefig('../models/results/rfConfusionMatrix.png')
    plt.show()

# cdew = buildRandomForest()