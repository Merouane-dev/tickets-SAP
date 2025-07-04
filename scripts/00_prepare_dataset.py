import pandas as pd
from pathlib import Path

RAW = Path("data/customer_support_tickets.csv")
OUT = Path("data/tickets_anonymises.csv")

cols_map = {           # mapping Kaggle → SAP
    "Ticket ID": "ID_TICKET",
    "Ticket Type": "TYPE",
    "Ticket Priority": "PRIORITE",
    "Date of Purchase": "HORO_DATE_OUVERTURE",
    "Time to Resolution": "RESOLVED_AT",
    "Ticket Status": "STATUT_FERMETURE",
}

df = pd.read_csv(RAW).rename(columns=cols_map)

# 1️ Durée réelle (heures)
df["DUREE_RESO_H"] = (
    pd.to_datetime(df["RESOLVED_AT"])
    - pd.to_datetime(df["HORO_DATE_OUVERTURE"])
).dt.total_seconds() / 3600

# 2️ Nettoyage minimal
df = df.dropna(subset=["DUREE_RESO_H", "TYPE", "PRIORITE"])
df = df[df["DUREE_RESO_H"] > 0]

# 3️ k-anonymity ≥ 15
mask = (df.groupby(["TYPE", "PRIORITE"]).transform("size") < 15)
df.loc[mask, ["TYPE", "PRIORITE"]] = "OTHER"

df.to_csv(OUT, index=False)
print(f"✅  Jeu final : {len(df):,} tickets anonymisés")
