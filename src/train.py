import argparse, pandas as pd, numpy as np, joblib, os
from src.features import engineer_features
from src.models.isoforest import IsoModel
from src.models.supervisor import Supervisor

def main(data, artifacts):
    df = pd.read_csv(data)
    X,y = engineer_features(df)
    iso = IsoModel(); iso.fit(X)
    iso_scores = iso.predict(X)
    sup = Supervisor(); sup.fit(iso_scores.reshape(-1,1), y)
    os.makedirs(artifacts,exist_ok=True)
    joblib.dump(iso, artifacts+'/iso.pkl'); joblib.dump(sup, artifacts+'/sup.pkl')
    print('Models saved.')

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/sample.csv')
    ap.add_argument('--artifacts_dir', default='artifacts')
    a=ap.parse_args(); main(a.data,a.artifacts_dir)
