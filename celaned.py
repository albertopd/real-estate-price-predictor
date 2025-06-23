import pandas as pd

# 1️⃣ Charger le fichier
df = pd.read_csv('zimmo_real_estate_jgchoti.csv')

# 2️⃣ Supprimer les doublons
df = df.drop_duplicates()

# 3️⃣ Nettoyer les espaces dans les colonnes texte (object)
df_obj = df.select_dtypes(include=['object'])
df[df_obj.columns] = df_obj.apply(lambda x: x.astype(str).str.strip())

# 4️⃣ Supprimer les lignes où il manque des valeurs essentielles :
# Par exemple : code, type, price, postcode, city
df = df.dropna(subset=['code', 'type', 'price', 'postcode', 'city'])

# 5️⃣ Convertir les colonnes numériques qui doivent être numériques (juste au cas où)
colonnes_numeriques = [
    'price',
    'living area(m²)',
    'ground area(m²)',
    'bedroom',
    'bathroom',
    'garage',
    'EPC(kWh/m²)',
    'year built',
    'mobiscore'
]
# Forcer la conversion (et mettre NaN en cas de problème)
for col in colonnes_numeriques:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 6️⃣ Vérifier qu'il n'y a plus de valeurs manquantes critiques
print(df.isnull().sum())

# 7️⃣ Sauvegarder le dataset nettoyé
df.to_csv('zimmo_real_estate_clean.csv', index=False)

print("✅ Nettoyage terminé ! Le fichier nettoyé est 'zimmo_real_estate_clean.csv'")
