import streamlit as st, pandas as pd, joblib
from src.features import engineer_features

st.title('Fraud Detection Dashboard')
try:
    df = pd.read_csv('data/sample.csv').head(200)
    X,y = engineer_features(df)
    iso = joblib.load('artifacts/iso.pkl')
    sup = joblib.load('artifacts/sup.pkl')
    iso_scores = iso.predict(X)
    preds = (sup.predict(iso_scores.reshape(-1,1))>0.5).astype(int)
    df['pred'] = preds
    st.write(df[['user','amount','fraud','pred']])
except Exception as e:
    st.error(str(e))
