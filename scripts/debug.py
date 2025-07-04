import pandas as pd
from pathlib import Path

# Charger les données en parsant les dates
fpath = Path("data/tickets_anonymises.csv")
df = pd.read_csv(fpath, parse_dates=["HORO_DATE_OUVERTURE"])

# Corriger le nom de colonne : DUREE_RESO_H est en fait RESOLVED_AT
df = df.rename(columns={"DUREE_RESO_H": "RESOLVED_AT"})
df["RESOLVED_AT"] = pd.to_datetime(df["RESOLVED_AT"], errors="coerce")

# Calculer la vraie durée en heures
df["DUREE_RESO_H"] = (df["RESOLVED_AT"] - df["HORO_DATE_OUVERTURE"]).dt.total_seconds() / 3600

# Supprimer les durées nulles ou négatives
df = df[df["DUREE_RESO_H"] > 0]

# Sauvegarder le fichier corrigé
out_path = Path("data/tickets_anonymises_fixed.csv")
df.to_csv(out_path, index=False)

print("✅ Durées corrigées :", df.shape, "lignes conservées.")