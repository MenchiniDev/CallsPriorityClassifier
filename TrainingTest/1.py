import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split 

def chi_square_test():
    df = pd.read_csv('../datasets/911Calls.csv', sep='\t')

    df.drop(['Unnamed: 19'], axis=1, inplace=True)

    df.drop_duplicates(inplace=True)

    group_mapping = {
        'Non-Emergency': 0,
        'Low': 0,
        'Out of Service': 0,
        'Medium': 1,
        'High': 2,
        'Emergency': 2,
    }
    df['priority'] = df['priority'].map(group_mapping)
    df.dropna(inplace=True)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].astype('category').apply(lambda x: x.cat.codes)

    y = df['priority']
    X = df.drop(columns='priority')
    
    print(X.describe())
    chi2_statistics, p_values = chi2(X, y)
    
    for i in range(len(chi2_statistics)):
        print(f'chi2-stat Att.{i}: {chi2_statistics[i]:.4f}  -  p-value Att.{i}: {p_values[i]:.4e}')
    
    X_indices = np.arange(X.shape[1])
    
    plt.figure(figsize=(10, 6))
    plt.bar(X_indices, p_values, width=0.5, color='blue', edgecolor='black')
    plt.title("Feature univariate score (chi-square test)")
    plt.xlabel("Feature number")
    plt.ylabel(r"Univariate score ($p_{value}$)")
    plt.xticks(X_indices)
    plt.axhline(y=0.05, color='red', linestyle='--', label='Significance level (0.05)')
    plt.legend()
    plt.show()


chi_square_test()


