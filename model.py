import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load Dataset
df = pd.read_csv("data/diabetes.csv")

# 2. Features & Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")

# 6. Save Model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)
