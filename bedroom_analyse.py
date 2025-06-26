import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("immoweb-dataset-clean.csv")

plt.figure(figsize=(10,6))
sns.barplot(
    data=df, 
    x='bedroomCount', 
    y='price', 
    estimator='mean', 
    ci='sd', 
    palette='Blues'
)
plt.title("Prix moyen en fonction du nombre de chambres")
plt.xlabel("Nombre de chambres")
plt.ylabel("Prix moyen (€)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(
    data=df, 
    x='bedroomCount', 
    y='price', 
    palette='coolwarm'
)
plt.title("Distribution du prix selon le nombre de chambres")
plt.xlabel("Nombre de chambres")
plt.ylabel("Prix (€)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
scatter = sns.scatterplot(
    data=df, 
    x='bedroomCount', 
    y='habitableSurface', 
    size='price', 
    hue='price', 
    palette='viridis', 
    sizes=(20, 400), 
    legend='full'
)

plt.title("Relation entre nombre de chambres, surface habitable et prix")
plt.xlabel("Nombre de chambres")
plt.ylabel("Surface habitable (m²)")
plt.legend(title='Prix (€)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


