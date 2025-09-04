import pandas as pd, time, json

def stream(path='data/sample.csv'):
    for row in pd.read_csv(path).to_dict(orient='records'):
        print(json.dumps(row))
        time.sleep(0.1)

if __name__=='__main__': stream()
