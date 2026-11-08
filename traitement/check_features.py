import numpy as np, pandas as pd
d = r"C:\Users\bougu\Real-Time-Embedded-STM32_Acoustic_Diagnostic_System_for_Electromechanical_Equipment_Monitoring\windows\slider"
names = open(d + r"\features_names.txt").read().split("\n")
F = np.fromfile(d + r"\features.f32", dtype=np.float32).reshape(-1, len(names))
meta = pd.read_csv(d + r"\meta.csv")
assert len(meta) == len(F), (len(meta), len(F))
print("lignes :", len(F), "| classes :", meta.label.value_counts().to_dict())
print("ids :", meta.id.value_counts().to_dict())

df = pd.DataFrame(F, columns=names)
print("\nEnergie relative moyenne par bande (0 = 0-500 Hz ... 15 = 7500-8000 Hz) :")
print(df[[f"w{b:02d}_rms_rel" for b in range(16)]].mean().round(4).to_string())

y = (meta.label == "abnormal").values
from scipy.stats import f_oneway
res = [(n, f_oneway(df[n][~y], df[n][y]).statistic) for n in names]
print("\nTop 10 ANOVA brut (indicatif, pas une sélection) :")
for n, s in sorted(res, key=lambda r: -r[1])[:10]: print(f"{n:16s} {s:10.1f}")