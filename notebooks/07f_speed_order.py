import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import find_peaks

d = "../outputs/features/"
base = "../data/slider/id_00"
SR, WIN, HOP = 16000, 800, 400
FS = SR / HOP
LMIN, LMAX = int(0.3 * FS), int(4.0 * FS)

y = np.load(d + "y.npy")
names = np.load(d + "names.npy")

def period(path):
    x = sf.read(path)[0][:, 0]
    env = 20 * np.log10(librosa.feature.rms(y=x, frame_length=WIN, hop_length=HOP)[0] + 1e-10)
    e = env - env.mean(); n = len(e)
    f = np.fft.rfft(e, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:n]; a /= a[0] + 1e-12
    seg = a[LMIN:LMAX]
    pk, pr = find_peaks(seg, prominence=0.05)
    return (LMIN + pk[np.argmax(pr["prominences"])]) / FS if len(pk) else np.nan

def voisins(fast):
    p = fast.mean()
    return (fast[1:] == fast[:-1]).mean(), p**2 + (1 - p)**2

# Normaux : TOUS, dans l'ordre des numéros de fichier
nor = np.where(y == 0)[0]
nid = np.array([int(os.path.basename(n).split(".")[0]) for n in names[nor]])
nor = nor[np.argsort(nid)]
fast_n = np.array([period(os.path.join(base, names[i])) < 3.0 for i in nor])
s, e = voisins(fast_n)
print(f"Normaux   : rapides {fast_n.mean():.2f} | voisins de même vitesse {s:.3f} (attendu {e:.3f})")

# Anomalies : reprises du CSV de 07d (déjà dans l'ordre des fichiers)
df = pd.read_csv("../outputs/indicateurs_physiques.csv")
an = df[df.cl >= 0].reset_index(drop=True)
fast_a = (an.periode_s < 3.0).to_numpy()
s, e = voisins(fast_a)
print(f"Anomalies : rapides {fast_a.mean():.2f} | voisins de même vitesse {s:.3f} (attendu {e:.3f})")