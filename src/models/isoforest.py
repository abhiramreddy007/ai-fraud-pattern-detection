from sklearn.ensemble import IsolationForest
class IsoModel:
    def __init__(self): self.model = IsolationForest(n_estimators=100, contamination=0.02)
    def fit(self,X): self.model.fit(X)
    def predict(self,X): return -self.model.decision_function(X)
