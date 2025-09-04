import pandas as pd
from src.features import engineer_features

def test_engineer_features():
    df = pd.DataFrame({'amount':[10,20],'timestamp':pd.date_range('2024-01-01',periods=2),'fraud':[0,1]})
    X,y = engineer_features(df)
    assert 'log_amount' in X.columns
    assert len(X)==2
