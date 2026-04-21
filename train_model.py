import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("WineQT.csv")

X = df.drop("quality", axis=1)  
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "C:\\Users\\shrey\\OneDrive\\Desktop\\temporary\\Classification\\WineQT.pkl")