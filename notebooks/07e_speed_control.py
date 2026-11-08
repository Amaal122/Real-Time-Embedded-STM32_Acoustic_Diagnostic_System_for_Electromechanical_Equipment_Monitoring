import os
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import find_peaks

d = "../outputs/features/"
base = "../data/slider/id_00"
SR, WIN, HOP = 16000, 800, 400
FS = SR / HOP
LMIN, LMAX = int(0.3 * FS), int(4.0 * FS)

names = np.load(d + "names.npy")
tr = np.load(d + "split.npz")["train"]
abn = np.load(d + "abn_idx.npy")
ref = np.load(d + "labels_A_residu_k3.npy")
spec = np.load(d + "X_A_logmel.npy").mean(axis=2)      # (N, 64)

def period(path):
    x = sf.read(path)[0][:, 0]
    env = 20 * np.log10(librosa.feature.rms(y=x, frame_length=WIN, hop_length=HOP)[0] + 1e-10)
    e = env - env.mean(); n = len(e)
    f = np.fft.rfft(e, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:n]; a /= a[0] + 1e-12
    seg = a[LMIN:LMAX]
    pk, pr = find_peaks(seg, prominence=0.05)
    return (LMIN + pk[np.argmax(pr["prominences"])]) / FS if len(pk) else np.nan

per_n = np.array([period(os.path.join(base, names[i])) for i in tr])
per_a = np.array([period(os.path.join(base, names[i])) for i in abn])

fast_n = per_n < 3.0
print(f"Normaux (train) : rapides {fast_n.sum()} | lents {(~fast_n).sum()} | sans pic {np.isnan(per_n).sum()}")
print("Anomalies, période courte par cluster :",
      [int((per_a[ref == c] < 3.0).sum()) for c in range(3)], "sur", np.bincount(ref))

# Effet de la vitesse seule sur le spectre (normaux)
v_speed = spec[tr][fast_n].mean(axis=0) - spec[tr][~fast_n].mean(axis=0)
# Différences entre clusters (anomalies)
sa = spec[abn]
v_c2 = sa[ref == 2].mean(axis=0) - sa[ref != 2].mean(axis=0)
v_01 = sa[ref == 0].mean(axis=0) - sa[ref == 1].mean(axis=0)

def r(a, b): return np.corrcoef(a, b)[0, 1]
print(f"\nEffet moyen de la vitesse (normaux)     : {np.abs(v_speed).mean():.2f} dB")
print(f"Différence cluster 2 vs {{0,1}}            : {np.abs(v_c2).mean():.2f} dB")
print(f"Différence cluster 0 vs 1                : {np.abs(v_01).mean():.2f} dB")
print(f"\nCorrélation de forme spectrale :")
print(f"  vitesse (normaux) vs cluster 2 vs {{0,1}} : r = {r(v_speed, v_c2):+.2f}")
print(f"  vitesse (normaux) vs cluster 0 vs 1      : r = {r(v_speed, v_01):+.2f}")