from sklearn.linear_model import LogisticRegression
class Supervisor:
    def __init__(self): self.clf = LogisticRegression()
    def fit(self,X,y): self.clf.fit(X,y)
    def predict(self,X): return self.clf.predict_proba(X)[:,1]
