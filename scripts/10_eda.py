import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -- Lecture
df = pd.read_csv("data/tickets_anonymises.csv")

# -- Convertir les dates
df["HORO_DATE_OUVERTURE"] = pd.to_datetime(df["HORO_DATE_OUVERTURE"], errors="coerce", utc=True)
df["RESOLVED_AT"] = pd.to_datetime(df["DUREE_RESO_H"], errors="coerce", utc=True)

# -- Durée en heures
df["DUREE_RESO_H"] = (df["RESOLVED_AT"] - df["HORO_DATE_OUVERTURE"]).dt.total_seconds() / 3600

# -- Nettoyage
df = df.dropna(subset=["DUREE_RESO_H"])
df = df[df["DUREE_RESO_H"] > 0]

# -- Feature artificielle : longueur du libellé TYPE
df["TYPE_LEN"] = df["TYPE"].astype(str).apply(len)

# -- Créer dossier de sortie
Path("figures").mkdir(exist_ok=True)

# -- Histogramme
plt.figure()
plt.hist(df["DUREE_RESO_H"], bins=50)
plt.xlabel("Durée de résolution (h)")
plt.ylabel("Fréquence")
plt.title("Histogramme DUREE_RESO_H")
plt.xlim(df["DUREE_RESO_H"].min(), df["DUREE_RESO_H"].max())  # adapte dynamiquement
plt.savefig("figures/Fig_N-4-1.png", dpi=300)

# -- Heatmap des corrélations
numeric_df = df[["DUREE_RESO_H", "TYPE_LEN"]]
plt.figure()
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Corrélations — features numériques")
plt.savefig("figures/Fig_N-4-2.png", dpi=300)

# -- Stats descriptives
df[["DUREE_RESO_H"]].describe().T.to_csv("figures/Tab_N-4-A.csv")
print("✅ Figures générées dans /figures/")
