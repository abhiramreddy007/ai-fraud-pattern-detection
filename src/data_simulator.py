import pandas as pd, numpy as np, argparse

def generate(n=10000, seed=42):
    np.random.seed(seed)
    users = np.random.randint(1, 500, n)
    amounts = np.random.exponential(100, n)
    times = pd.date_range('2024-01-01', periods=n, freq='min')
    fraud_flags = np.random.choice([0,1], size=n, p=[0.98,0.02])
    return pd.DataFrame(dict(user=users, amount=amounts, timestamp=times, fraud=fraud_flags))

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/sample.csv')
    ap.add_argument('--rows', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    df = generate(a.rows,a.seed)
    df.to_csv(a.out,index=False)
    print(f'Saved {a.rows} rows to {a.out}')
