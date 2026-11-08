import numpy as np

# AUC et pAUC par graine : [graine 0, graine 1, graine 2]
res = {
    "A": {"AUC": [1.0000, 0.9999, 0.9999], "pAUC": [1.0000, 0.9997, 0.9997]},
    "B": {"AUC": [0.9999, 0.9996,0.9999], "pAUC": [0.9993, 0.9980,0.9996]},
}

for v, m in res.items():
    for name, vals in m.items():
        a = np.array(vals)
        print(f"Variante {v} | {name:4s} : {a.mean():.4f} ± {a.std(ddof=1):.4f}   {vals}")