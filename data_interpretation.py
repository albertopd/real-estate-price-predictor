import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Paramètres style global ===
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})

# === 1. Charger les données ===
df = pd.read_csv("immoweb-dataset-clean.csv")

# === 2. Supprimer colonnes inutiles ===
cols_to_drop = ['url', 'id']
df.drop(columns=cols_to_drop, inplace=True)

# Supprimer colonnes avec >80% de NaN
high_na_cols = df.columns[df.isnull().mean() > 0.80].tolist()
df.drop(columns=high_na_cols, inplace=True)

# === 3. Créer colonne prix / m2 ===
df['price_per_m2'] = df['price'] / df['habitableSurface']

# === 4. Outliers ===
plt.figure(figsize=(14,8))
sns.boxplot(data=df[['price', 'habitableSurface', 'price_per_m2']])
plt.title("Outliers sur prix, surface habitable et prix/m²")
plt.ylabel("Valeur (€ ou m²)")
plt.xticks([0, 1, 2], ['Prix (€)', 'Surface habitable (m²)', 'Prix par m² (€)'])
plt.tight_layout()
plt.show()

# === 5. Colonnes restantes ===
print("\nColonnes restantes pour l'analyse :")
print(df.columns.tolist())

# === 6. Histogramme des surfaces ===
plt.figure(figsize=(10,6))
sns.histplot(df['habitableSurface'], bins=50, kde=True, color='skyblue')
plt.title("Distribution des surfaces habitables")
plt.xlabel("Surface habitable (m²)")
plt.ylabel("Nombre de propriétés")
plt.tight_layout()
plt.show()

# === 7. Corrélation des variables numériques ===
num_cols = df.select_dtypes(include=['float64', 'int64'])
corr_price = num_cols.corr()['price'].sort_values(ascending=False)

print("\nCorrélation avec le prix :")
print(corr_price)

# Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(num_cols.corr(), cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title("Heatmap des corrélations")
plt.tight_layout()
plt.show()

# === 8. Variables importantes ===
important_vars = ['habitableSurface', 'bedroomCount', 'bathroomCount', 'subtype', 'province']
print("\nVariables importantes pour le prix :")
for var in important_vars:
    if var in df.columns:
        print(f"- {var}")

# === 9. Prix par municipalité ===
municipality_price = df.groupby('locality').agg(
    avg_price=('price', 'mean'),
    median_price=('price', 'median'),
    avg_price_per_m2=('price_per_m2', 'mean'),
    count=('price', 'count')
).sort_values(by='avg_price', ascending=False)

# Top 10
top10_muni = municipality_price.head(10).reset_index()

plt.figure(figsize=(12,6))
sns.barplot(data=top10_muni, x='locality', y='avg_price', palette='rocket')
plt.title("Top 10 municipalités les + chères (prix moyen)")
plt.xlabel("Municipalité")
plt.ylabel("Prix moyen (€)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bottom 10
bottom10_muni = municipality_price.tail(10).reset_index()

plt.figure(figsize=(12,6))
sns.barplot(data=bottom10_muni, x='locality', y='avg_price', palette='crest')
plt.title("Top 10 municipalités les - chères (prix moyen)")
plt.xlabel("Municipalité")
plt.ylabel("Prix moyen (€)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === 10. Prix par province ===
province_price = df.groupby('province').agg(
    avg_price=('price', 'mean'),
    median_price=('price', 'median'),
    avg_price_per_m2=('price_per_m2', 'mean'),
    count=('price', 'count')
).sort_values(by='avg_price', ascending=False)

province_price_reset = province_price.reset_index()

plt.figure(figsize=(12,6))
sns.barplot(data=province_price_reset, x='province', y='avg_price', palette='mako')
plt.title("Prix moyen par province")
plt.xlabel("Province")
plt.ylabel("Prix moyen (€)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === 11. Prix par type de bien ===
type_price = df.groupby('subtype').agg(
    avg_price=('price', 'mean'),
    median_price=('price', 'median'),
    avg_price_per_m2=('price_per_m2', 'mean'),
    count=('price', 'count')
).sort_values(by='avg_price', ascending=False)

type_price_reset = type_price.reset_index()

plt.figure(figsize=(14,6))
sns.barplot(data=type_price_reset, x='subtype', y='avg_price', palette='viridis')
plt.title("Prix moyen par type de bien")
plt.xlabel("Type de bien")
plt.ylabel("Prix moyen (€)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
