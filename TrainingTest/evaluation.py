import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict

df = pd.read_csv('../datasets/balancedDataset.csv')

X = df.drop('priority', axis=1)
y = df['priority']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

selector = SelectKBest(f_classif, k=12)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

rf_model = joblib.load('../models/random_forest_classifier.pkl')
ab_model = joblib.load('../models/adaBoost_model.pkl')

def evaluate_models(models, X_test, y_test, class_labels):
    scores = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
        scores[model_name] = report
    return scores

models = {'Random Forest': rf_model,'AdaBoost': ab_model}
class_labels = [0,1,2]

def plot_roc_curves(models, X_test, y_test, class_labels):
    for model_name, model in models.items():
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(class_labels):
            y_test_bin = (y_test == label).astype(int)
            y_test_prob = model.predict_proba(X_test)[:, i]
            
            fpr, tpr, _ = roc_curve(y_test_bin, y_test_prob)
            auc = roc_auc_score(y_test_bin, y_test_prob)
            
            plt.plot(fpr, tpr, label=f'Class {label} (AUC = {auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc='best')
        plt.show()

class_labels = np.unique(y)

models = {
    'Random Forest': rf_model,
    'AdaBoost': ab_model
}

# Plot ROC
# plot_roc_curves(models, X_test_selected, y_test, class_labels)

rf_probs_train = rf_model.predict_proba(X_train_selected)
rf_probs_test = rf_model.predict_proba(X_test_selected)

for i, label in enumerate(class_labels):
    y_train_bin = (y_train == label).astype(int)
    y_test_bin = (y_test == label).astype(int)
    
    rf_train_auc = roc_auc_score(y_train_bin, rf_probs_train[:, i])
    rf_test_auc = roc_auc_score(y_test_bin, rf_probs_test[:, i])
    
    print(f"Random Forest Model - Class {label}:")
    print("Train AUC:", rf_train_auc)
    print("Test AUC:", rf_test_auc)

ab_probs_train = ab_model.predict_proba(X_train_selected)
ab_probs_test = ab_model.predict_proba(X_test_selected)

for i, label in enumerate(class_labels):
    y_train_bin = (y_train == label).astype(int)
    y_test_bin = (y_test == label).astype(int)
    
    ab_train_auc = roc_auc_score(y_train_bin, ab_probs_train[:, i])
    ab_test_auc = roc_auc_score(y_test_bin, ab_probs_test[:, i])
    
    print(f"AdaBoost Model - Class {label}:")
    print("Train AUC:", ab_train_auc)
    print("Test AUC:", ab_test_auc)

    def plot_confusion_matrix(model, X, y, class_labels):
        y_pred = cross_val_predict(model, X, y, cv=5)
        cm = confusion_matrix(y, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels)
        plt.yticks(tick_marks, class_labels)
        
        thresh = cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

# plot_confusion_matrix(rf_model, X_test_selected, y_test, class_labels)