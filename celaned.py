import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Charger les données
df = pd.read_csv("immoweb-dataset.csv")

# 2. Supprimer la colonne inutile
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# 3. Supprimer les doublons
df.drop_duplicates(inplace=True)

# 4. Nettoyer les espaces dans les colonnes de texte
str_cols = df.select_dtypes(include='object').columns
for col in str_cols:
    df[col] = df[col].astype(str).str.strip()

# 5. Supprimer les colonnes vides
df.dropna(axis=1, how='all', inplace=True)

# 6. Afficher les colonnes avec des valeurs manquantes
missing_values = df.isnull().sum()
print("Colonnes avec des valeurs manquantes :")
print(missing_values[missing_values > 0].sort_values(ascending=False))

# 7. Visualisation des valeurs manquantes
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Visualisation des valeurs manquantes", fontsize=16)
plt.show()

# 8. Imputation automatique simple :
# Remplir les colonnes numériques manquantes avec la médiane
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Remplir les colonnes catégorielles manquantes avec "Unknown"
cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].fillna("Unknown")

# 9. Vérification finale des valeurs manquantes
missing_values_final = df.isnull().sum().sum()
print(f"\nNombre total de valeurs manquantes après imputation : {missing_values_final}")

# 10. Sauvegarder le dataset nettoyé
df.to_csv("immoweb-dataset-clean.csv", index=False)
print("\n✅ Dataset nettoyé, imputé et sauvegardé sous 'immoweb-dataset-clean.csv'")
