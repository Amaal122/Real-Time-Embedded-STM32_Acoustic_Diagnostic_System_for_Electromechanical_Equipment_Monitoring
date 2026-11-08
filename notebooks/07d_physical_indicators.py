import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, kruskal
from scipy.signal import butter, sosfiltfilt, find_peaks

d = "../outputs/features/"
base = "../data/slider/id_00"
SR = 16000
WIN, HOP = 800, 400                 # 50 ms / 25 ms -> enveloppe à 40 Hz
FS_ENV = SR / HOP
LAG_MIN, LAG_MAX = int(0.3 * FS_ENV), int(4.0 * FS_ENV)

names = np.load(d + "names.npy")
abn = np.load(d + "abn_idx.npy")
ref = np.load(d + "labels_A_residu_k3.npy")
tr = np.load(d + "split.npz")["train"]

sos_hp = butter(4, 4000, btype="highpass", fs=SR, output="sos")

def acf(x):
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:n]
    return a / (a[0] + 1e-12)

def indicators(path):
    x = sf.read(path)[0][:, 0].astype(np.float64)
    env = 20 * np.log10(librosa.feature.rms(y=x, frame_length=WIN, hop_length=HOP)[0] + 1e-10)
    a = acf(env)
    seg = a[LAG_MIN:LAG_MAX]
    pk, props = find_peaks(seg, prominence=0.05)
    if len(pk):
        j = pk[np.argmax(props["prominences"])]
        period, height = (LAG_MIN + j) / FS_ENV, seg[j]
    else:
        period, height = np.nan, 0.0
    xh = sosfiltfilt(sos_hp, x)
    hf = 10 * np.log10((xh ** 2).sum() / ((x ** 2).sum() + 1e-12) + 1e-12)
    return dict(periode_s=period, pic_acf=height, ecart_env_dB=env.std(),
                kurtosis=kurtosis(x), crete=np.abs(x).max() / (np.sqrt((x ** 2).mean()) + 1e-12),
                hf_dB=hf)

rows = []
for i, gi in enumerate(abn):
    r = indicators(os.path.join(base, names[gi])); r.update(groupe=f"cluster {ref[i]}", cl=int(ref[i]))
    rows.append(r)
rng = np.random.default_rng(0)
for gi in rng.choice(tr, 100, replace=False):
    r = indicators(os.path.join(base, names[gi])); r.update(groupe="normal", cl=-1)
    rows.append(r)

df = pd.DataFrame(rows)
df.to_csv("../outputs/indicateurs_physiques.csv", index=False)

cols = ["periode_s", "pic_acf", "ecart_env_dB", "kurtosis", "crete", "hf_dB"]
print("=== Médiane [Q1 ; Q3] par groupe ===")
for c in cols:
    print(f"\n{c}")
    for g in ["normal", "cluster 0", "cluster 1", "cluster 2"]:
        v = df.loc[df.groupe == g, c].dropna()
        print(f"  {g:10s} n={len(v):3d} | {v.median():8.2f} [{v.quantile(.25):7.2f} ; {v.quantile(.75):7.2f}]")

print("\n=== Kruskal-Wallis entre les 3 clusters d'anomalies ===")
an = df[df.cl >= 0]
print(f"{'indicateur':14s} {'H':>7s} {'p':>10s} {'epsilon2':>9s}")
for c in cols:
    groups = [an.loc[an.cl == k, c].dropna() for k in range(3)]
    H, p = kruskal(*groups)
    print(f"{c:14s} {H:7.1f} {p:10.2e} {H / (len(an.dropna(subset=[c])) - 1):9.3f}")
print(f"\nFichiers sans pic de périodicité détecté : "
      f"{an.periode_s.isna().sum()}/{len(an)}")

fig, ax = plt.subplots(2, 3, figsize=(13, 7))
order = ["normal", "cluster 0", "cluster 1", "cluster 2"]
for a, c in zip(ax.ravel(), cols):
    a.boxplot([df.loc[df.groupe == g, c].dropna() for g in order], tick_labels=order)
    a.set_title(c); a.tick_params(axis="x", labelsize=8); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("../outputs/indicateurs_physiques.png", dpi=150); plt.show()