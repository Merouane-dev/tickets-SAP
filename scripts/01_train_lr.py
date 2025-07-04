# scripts/01_train_lr.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

# --- 1. Lecture
df = pd.read_csv("data/tickets_anonymises.csv")

# --- 2. Conversion des dates
df["HORO_DATE_OUVERTURE"] = pd.to_datetime(df["HORO_DATE_OUVERTURE"], errors="coerce", utc=True)
df["RESOLVED_AT"] = pd.to_datetime(df["DUREE_RESO_H"], errors="coerce", utc=True)
df["DUREE_RESO_H"] = (df["RESOLVED_AT"] - df["HORO_DATE_OUVERTURE"]).dt.total_seconds() / 3600
df["JOUR_OUVERTURE"] = df["HORO_DATE_OUVERTURE"].dt.dayofweek
df["HEURE_OUVERTURE"] = df["HORO_DATE_OUVERTURE"].dt.hour

# --- 3. Nettoyage
df = df.dropna(subset=["DUREE_RESO_H", "TYPE", "PRIORITE"])
df = df[df["DUREE_RESO_H"] > 0]

# --- 4. Encodage
features = ["TYPE", "PRIORITE", "JOUR_OUVERTURE", "HEURE_OUVERTURE"]
encoder = OneHotEncoder(drop="first", sparse_output=False)
X_encoded = encoder.fit_transform(df[features])
X = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(features))
y = df["DUREE_RESO_H"]

# --- 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 6. Entraînement
model = LinearRegression().fit(X_train, y_train)

# --- 7. Évaluation
r2  = r2_score(y_test, model.predict(X_test))
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"✅  R² = {r2:.3f} — MAE = {mae:.2f} h")

# --- 8. Sauvegarde
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/linear_regression.joblib")
joblib.dump(encoder, "models/encoder.joblib")

# --- 9. Figures
Path("figures").mkdir(exist_ok=True)
plt.figure()
plt.scatter(y_test, model.predict(X_test), alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "--")
plt.xlabel("Vraie DUREE_RESO_H")
plt.ylabel("Prédiction")
plt.title("Fig. N-5-1 : y_test vs y_pred")
plt.savefig("figures/Fig_N-5-1.png", dpi=300)

pd.Series(model.coef_, index=X.columns).sort_values().to_csv("figures/Tab_N-5-A.csv")
