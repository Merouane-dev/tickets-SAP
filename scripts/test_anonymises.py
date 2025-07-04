import pandas as pd
df = pd.read_csv("data/tickets_anonymises.csv")
print(df[['ID_TICKET','TYPE','PRIORITE','DUREE_RESO_H']].head())
print(df.describe())