import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Charger les données nettoyées
df = pd.read_csv("immoweb-dataset-clean.csv")

# 2. Nombre de lignes et de colonnes
n_rows, n_cols = df.shape
print(f"Nombre de lignes: {n_rows}")
print(f"Nombre de colonnes: {n_cols}")

# 3. Pourcentage de valeurs manquantes par colonne
missing_pct = df.isnull().mean() * 100
print("\nPourcentage de valeurs manquantes par colonne:")
print(missing_pct.sort_values(ascending=False))

# Visualisation des % de valeurs manquantes
plt.figure(figsize=(10,6))
missing_pct[missing_pct > 0].sort_values(ascending=False).plot(kind='bar', color='salmon')
plt.title("% de valeurs manquantes par colonne")
plt.ylabel("%")
plt.show()

# 4. Variables quantitatives et qualitatives
num_vars = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_vars = df.select_dtypes(include='object').columns.tolist()

print(f"\nVariables quantitatives : {num_vars}")
print(f"\nVariables qualitatives : {cat_vars}")

# Idée : transformer les qualitatives en numériques (ex: one-hot encoding ou label encoding)
# Exemple simple : df_encoded = pd.get_dummies(df, columns=cat_vars)

# 5. Corrélation entre les variables et le prix
# On suppose qu'il y a une colonne 'price' dans le dataset :
# Sélection des colonnes numériques uniquement
df_num = df.select_dtypes(include=['float64', 'int64'])

# Calcul de la corrélation entre les variables numériques et le prix
if 'price' in df_num.columns:
    corr_price = df_num.corr()['price'].sort_values(ascending=False)

    # Visualisation
    plt.figure(figsize=(12,8))
    sns.barplot(x=corr_price.values, y=corr_price.index, palette='viridis')
    plt.title("Corrélation des variables numériques avec le prix")
    plt.show()

    # Affichage
    print("\nCorrélation des variables numériques avec le prix :")
    print(corr_price)
else:
    print("⚠️ La colonne 'price' n'est pas de type numérique ou n'existe pas.")

# 6. Corrélation entre variables (heatmap)
plt.figure(figsize=(14,12))
sns.heatmap(df_num.corr(), cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
plt.title("Heatmap des corrélations entre variables numériques")
plt.show()

# 7. Variables les plus influentes sur le prix
if 'price' in df.columns:
    top_influential = corr_price.drop('price').head(5)
    least_influential = corr_price.drop('price').tail(5)

    print("\nVariables les plus influentes sur le prix :")
    print(top_influential)

    print("\nVariables les moins influentes sur le prix :")
    print(least_influential)

