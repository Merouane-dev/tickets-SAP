# scripts/02_cross_validation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# --- 1. Chargement et traitement dates --------------------------
df = pd.read_csv("data/tickets_anonymises.csv")
df["HORO_DATE_OUVERTURE"] = pd.to_datetime(df["HORO_DATE_OUVERTURE"], errors="coerce", utc=True)
df["RESOLVED_AT"] = pd.to_datetime(df["DUREE_RESO_H"], errors="coerce", utc=True)

# --- 2. Calcul de durée réelle en heures -------------------------
df["DUREE_RESO_H"] = (df["RESOLVED_AT"] - df["HORO_DATE_OUVERTURE"]).dt.total_seconds() / 3600

# --- 3. Nettoyage minimal ----------------------------------------
cat_cols = ["TYPE", "PRIORITE"]
df = df.dropna(subset=cat_cols + ["DUREE_RESO_H"])
df = df[df["DUREE_RESO_H"] > 0]

# --- 4. Features & target ----------------------------------------
X = df[cat_cols]
y = df["DUREE_RESO_H"]

# --- 5. Encodage + pipeline --------------------------------------
encoder = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
])
model = make_pipeline(encoder, LinearRegression())

# --- 6. Validation croisée ---------------------------------------
n_samples = len(y)
if n_samples < 2:
    raise ValueError(f"Pas assez de données : {n_samples} ligne(s) seulement.")

cv = min(5, n_samples)
scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")

print("✅ Validation croisée réussie")
print("MAE (par fold) :", scores.round(2))
print("Variance      :", np.round(scores.var(), 2))
