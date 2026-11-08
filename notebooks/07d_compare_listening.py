import pandas as pd
from sklearn.metrics import adjusted_rand_score

out = "../outputs/listening/"
key = pd.read_csv(out + "key.csv")
fiche = pd.read_csv(out + "fiche_ecoute_propre.csv")

# La colonne "groupe" peut avoir un nom légèrement différent selon Excel
col = [c for c in fiche.columns if c.startswith("groupe")][0]
fiche = fiche.rename(columns={col: "groupe"})

df = key.merge(fiche[["code", "groupe"]], on="code")
df["groupe"] = df["groupe"].astype(str).str.strip().str.upper()
print(df[["code", "type", "cluster", "groupe"]].to_string(index=False))

# --- Normaux reconnus ---
nor = df[df["type"] == "normal"]
ok = (nor["groupe"] == "N").sum()
print(f"\nNormaux reconnus : {ok}/{len(nor)}")
fausses = df[(df["type"] != "normal") & (df["groupe"] == "N")]
print(f"Anomalies prises pour des normaux : {len(fausses)}/{(df['type'] != 'normal').sum()}")

# --- Tableau croisé groupes perçus x clusters (anomalies seulement) ---
an = df[df["type"] != "normal"]
print("\nTableau croisé (lignes = cluster réel, colonnes = groupe proposé) :")
print(pd.crosstab(an["cluster"], an["groupe"]))

# --- ARI (sans les anomalies classées N) ---
an2 = an[an["groupe"].isin(["A", "B", "C"])]
if an2["groupe"].nunique() > 1:
    ari = adjusted_rand_score(an2["cluster"], an2["groupe"])
    print(f"\nARI (groupes perçus vs clusters), n={len(an2)} : {ari:.3f}")
else:
    print("\nARI non calculable : un seul groupe utilisé.")