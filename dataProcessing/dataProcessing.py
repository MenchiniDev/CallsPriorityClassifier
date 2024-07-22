import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

def undersampling():
    df = pd.read_csv('../datasets/911Calls.csv', sep='\t')

    df.drop(['Unnamed: 19', 'recordId', 'callKey', 'location', 'ESRI_OID', 'callNumber', 'NeedsSync', 'ZIPCode', 'SheriffDistricts', 'Community_Statistical_Areas'], axis=1, inplace=True)
    df.drop(['CouncilDistrict', 'PolicePost'], axis=1, inplace=True)

    df.drop_duplicates(inplace=True)
    df.to_csv('../datasets/reductedDataset.csv', index=False)

    #grouping labels into priority 
    group_mapping = {
        'Non-Emergency': 0,
        'Low': 0,
        'Out of Service': 0,
        'Medium': 1,
        'High': 2,
        'Emergency': 2,
    }
    df['priority'] = df['priority'].map(group_mapping)
    
    df['callDateTime'] = pd.to_datetime(df['callDateTime'], utc=True)
    df['year'] = df['callDateTime'].dt.year
    df['month'] = df['callDateTime'].dt.month
    df['day'] = df['callDateTime'].dt.day
    df['hour'] = df['callDateTime'].dt.hour
    df['minute'] = df['callDateTime'].dt.minute
    df['dayofweek'] = df['callDateTime'].dt.dayofweek
    df.drop(columns='callDateTime', inplace=True)
    df.dropna(inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].astype('category').apply(lambda x: x.cat.codes)
    df.to_csv("../datasets/encodedDataset.csv", index=False)


    y = df['priority']
    X = df.drop(columns='priority')

    ros = RandomUnderSampler(random_state=4)
    X_balanced, y_balanced = ros.fit_resample(X, y)

    df_balanced = pd.DataFrame(X_balanced, columns=X.columns)
    df_balanced['priority'] = y_balanced

    df_balanced.to_csv('../datasets/balancedDataset.csv', index=False)
    return df_balanced

# df = undersampling()