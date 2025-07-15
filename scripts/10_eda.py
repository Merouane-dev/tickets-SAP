import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -- Lecture -----------------------------------------------------------------
df = pd.read_csv("data/tickets_anonymises.csv")

# -- Conversion des dates ----------------------------------------------------
# 1) Date d’ouverture
df["HORO_DATE_OUVERTURE"] = pd.to_datetime(
    df["HORO_DATE_OUVERTURE"], errors="coerce", utc=True
)

# 2) Date de résolution (RESOLVED_AT existe dans le CSV, donc on la lit telle quelle)
df["RESOLVED_AT"] = pd.to_datetime(
    df["RESOLVED_AT"], errors="coerce", utc=True
)

# -- Durée en heures ---------------------------------------------------------
# ⬅️  Calculer la durée à partir des deux dates, pas l’inverse
df["DUREE_RESO_H"] = (
    df["RESOLVED_AT"] - df["HORO_DATE_OUVERTURE"]
).dt.total_seconds() / 3600

# -- Nettoyage ---------------------------------------------------------------
df = df.dropna(subset=["DUREE_RESO_H", "TYPE", "PRIORITE"])
df = df[df["DUREE_RESO_H"] > 0]

# -- Feature artificielle ----------------------------------------------------
df["TYPE_LEN"] = df["TYPE"].astype(str).str.len()

# -- Dossier de sortie -------------------------------------------------------
Path("figures").mkdir(exist_ok=True)

# -- Histogramme -------------------------------------------------------------
plt.figure()
plt.hist(df["DUREE_RESO_H"], bins=40)        # harmonisé : 40 classes
plt.xlabel("Durée de résolution (h)")
plt.ylabel("Fréquence")
plt.title("Histogramme DUREE_RESO_H")
plt.xlim(df["DUREE_RESO_H"].min(), df["DUREE_RESO_H"].max())
plt.savefig("figures/Fig_N-4-1.png", dpi=300)

# -- Heat-map des corrélations ----------------------------------------------
numeric_df = df[["DUREE_RESO_H", "TYPE_LEN"]]
plt.figure()
sns.heatmap(numeric_df.corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Corrélations — features numériques (Spearman)")
plt.savefig("figures/Fig_N-4-2.png", dpi=300)

# -- Statistiques descriptives ----------------------------------------------
df[["DUREE_RESO_H"]].describe().T.to_csv("figures/Tab_N-4-A.csv")

print("✅ Figures et tableau générés dans le dossier /figures/")
