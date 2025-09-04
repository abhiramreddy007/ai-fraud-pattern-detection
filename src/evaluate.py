import argparse, pandas as pd, joblib
from sklearn.metrics import classification_report
from src.features import engineer_features

def main(data, artifacts):
    df = pd.read_csv(data)
    X,y = engineer_features(df)
    iso = joblib.load(artifacts+'/iso.pkl')
    sup = joblib.load(artifacts+'/sup.pkl')
    iso_scores = iso.predict(X)
    preds = (sup.predict(iso_scores.reshape(-1,1))>0.5).astype(int)
    print(classification_report(y,preds))

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/sample.csv')
    ap.add_argument('--artifacts_dir', default='artifacts')
    a=ap.parse_args(); main(a.data,a.artifacts_dir)
