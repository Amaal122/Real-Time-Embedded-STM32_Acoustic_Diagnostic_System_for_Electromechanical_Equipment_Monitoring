import numpy as np, pandas as pd
from scipy.stats import f_oneway

d = r"C:\Users\bougu\Real-Time-Embedded-STM32_Acoustic_Diagnostic_System_for_Electromechanical_Equipment_Monitoring\windows\slider"
names = open(d + r"\features_names.txt").read().split("\n")
F = np.fromfile(d + r"\features.f32", dtype=np.float32).reshape(-1, len(names))
meta = pd.read_csv(d + r"\meta.csv")
df = pd.DataFrame(F, columns=names)
y = (meta.label == "abnormal").values

print("Part de chaque classe par machine :")
print(pd.crosstab(meta.id, meta.label, normalize="index").round(3))

print("\nTop 6 ANOVA par machine :")
for i in sorted(meta.id.unique()):
    m = (meta.id == i).values
    s = [(n, f_oneway(df[n][m & ~y], df[n][m & y]).statistic) for n in names]
    print(i, [n for n, _ in sorted(s, key=lambda r: -r[1])[:6]])