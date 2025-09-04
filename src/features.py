import pandas as pd, numpy as np

def engineer_features(df):
    df = df.copy()
    df['log_amount'] = np.log1p(df['amount'])
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    return df[['log_amount','hour']], df['fraud']
