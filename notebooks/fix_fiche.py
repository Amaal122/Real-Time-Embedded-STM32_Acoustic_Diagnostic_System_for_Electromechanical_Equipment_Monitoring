import pandas as pd

path = "../outputs/listening/fiche_ecoute.csv"
rows = []
with open(path, encoding="utf-8-sig") as f:
    next(f)                                   # on saute l'en-tête abîmé
    for line in f:
        line = line.strip()
        if not line:
            continue
        code_part, vals = line.split(";", 1)  # avant / après le premier ;
        code = code_part.split(",")[0]
        v = vals.split(";")
        rows.append([code] + v[:5] + [v[5] if len(v) > 5 else ""] +
                    [v[6] if len(v) > 6 else ""])

cols = ["code", "regulier", "claquements", "sifflement_aigu",
        "grave_dominant", "bruit_large", "groupe", "remarque"]
df = pd.DataFrame(rows, columns=cols)
df["groupe"] = df["groupe"].str.strip().str.upper()
print(df.to_string(index=False))
df.to_csv("../outputs/listening/fiche_ecoute_propre.csv", index=False)