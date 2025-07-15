# SAP Ticket Resolution Time – Proof-of-Concept 📊

Ce dépôt contient le code, les figures et les modèles développés dans le cadre de mon mémoire de fin d’études.  
Objectif : démontrer la faisabilité d’une **estimation automatique de la durée de résolution** d’un ticket d’incident SAP à l’aide de techniques de data-science.

## 💾 Contenu du dépôt

.
├── data/ # Jeu de données anonymisé (CSV)
├── scripts/
│ ├── 00_prepare_dataset.py # Nettoyage & mapping Kaggle → SAP
│ ├── 10_eda.py # Analyse exploratoire + figures
│ └── 01_train_lr.py # Entraînement du modèle linéaire
├── api/
│ └── app.py # API Flask (prototype widget Fiori)
├── models/ # Modèle et encoder (générés)
├── figures/ # Figures & tableaux (générés)
├── requirements.txt # Dépendances Python
└── README.md # Ce fichier

bash
Copier
Modifier

## 🚀 Installation rapide

 
# 1. Cloner le dépôt
git clone https://github.com/Merouane-dev/tickets-SAP.git
cd tickets-SAP

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate   # sous Windows : .venv\\Scripts\\activate

# 3. Installer les dépendances
pip install -r requirements.txt
🏃 Exécution des scripts
Préparer le dataset

bash
Copier
Modifier
python scripts/00_prepare_dataset.py
Exploration & figures

bash
Copier
Modifier
python scripts/10_eda.py
Entraîner le modèle linéaire

bash
Copier
Modifier
python scripts/01_train_lr.py
Les artefacts sont enregistrés dans models/ et figures/.

🌐 Prototype d’API Flask
bash
Copier
Modifier
cd api
python app.py
Le service expose un endpoint /predict qui renvoie la durée estimée à partir des champs du ticket passés en JSON.

📊 Résultats de base
Metric	Valeur
R²	≈ –0,02
MAE	≈ 4 400 h (~6 mois)

Ces résultats mettent en évidence la nécessité d’enrichir les variables et d’utiliser des modèles non linéaires pour atteindre une précision exploitable.


🙏 Remerciements
Jeu de données original : Customer Support Ticket Dataset (Kaggle).
Ce travail s’inscrit dans le cadre du mémoire de fin d’études –  « Data-science appliquée aux SI ».
