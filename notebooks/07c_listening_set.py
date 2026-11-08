import os, csv
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

d = "../outputs/features/"
base = "../data/slider/id_00"
out = "../outputs/listening"
os.makedirs(out, exist_ok=True)
K = 3

y = np.load(d + "y.npy")
names = np.load(d + "names.npy")
tr = np.load(d + "split.npz")["train"]
abn = np.load(d + "abn_idx.npy")
ref = np.load(d + "labels_A_residu_k3.npy")

# --- Reconstruire exactement la représentation de 7a ---
XA = np.load(d + "X_A_logmel.npy")
mu_norm = XA[tr].mean(axis=(0, 2))
F = XA[abn].mean(axis=2) - mu_norm
P = PCA(n_components=0.95, random_state=0).fit_transform(StandardScaler().fit_transform(F))
km = KMeans(K, n_init=10, random_state=0).fit(P)
print("Labels identiques à 7a :", bool(np.array_equal(km.labels_, ref)))
assert np.array_equal(km.labels_, ref), "Les labels diffèrent de 7a : arrêtez-vous ici."

# --- Sélection automatique ---
rng = np.random.default_rng(0)
sel = []   # (type, cluster, indice global)
for c in range(K):
    members = np.where(ref == c)[0]
    dist = np.linalg.norm(P[members] - km.cluster_centers_[c], axis=1)
    central = members[np.argsort(dist)[:3]]
    others = np.setdiff1d(members, central)
    rand = rng.choice(others, 2, replace=False)
    for m in central: sel.append(("central", c, abn[m]))
    for m in rand:    sel.append(("aleatoire", c, abn[m]))
for i in rng.choice(tr, 5, replace=False):
    sel.append(("normal", -1, i))

# --- Lecture canal 0, gain commun ---
audio = []
for _, _, gi in sel:
    x, sr = sf.read(os.path.join(base, names[gi]))
    audio.append(x[:, 0].astype(np.float32))
gain = 0.9 / max(np.abs(a).max() for a in audio)
print(f"Gain commun : {20*np.log10(gain):+.1f} dB (même valeur pour tous les fichiers)")

# --- Ordre aléatoire et codes aveugles ---
order = rng.permutation(len(sel))
rows = []
for code, j in enumerate(order, start=1):
    typ, c, gi = sel[j]
    fname = f"L{code:02d}.wav"
    sf.write(os.path.join(out, fname), audio[j] * gain, sr, subtype="PCM_16")
    rows.append([fname, typ, c, names[gi]])

with open(os.path.join(out, "key.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["code", "type", "cluster", "fichier_origine"])
    w.writerows(rows)

# --- Feuille d'écoute vide à remplir ---
with open(os.path.join(out, "fiche_ecoute.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["code", "regulier(1-5)", "claquements(0/1)", "sifflement_aigu(0/1)",
                "grave_dominant(0/1)", "bruit_large(0/1)", "groupe_propose(A/B/C/N)", "remarque"])
    for r in rows: w.writerow([r[0]] + [""] * 7)

# --- Spectrogrammes de contrôle (non aveugle) ---
fig, axes = plt.subplots(4, 5, figsize=(16, 9))
for ax, j in zip(axes.ravel(), range(len(sel))):
    typ, c, gi = sel[j]
    S = librosa.power_to_db(librosa.feature.melspectrogram(
        y=audio[j], sr=16000, n_fft=1024, hop_length=512, n_mels=64), ref=1.0)
    ax.imshow(S, origin="lower", aspect="auto", vmin=-70, vmax=-10)
    ax.set_title(("normal" if c < 0 else f"cluster {c}") + f" ({typ[:4]})", fontsize=8)
    ax.axis("off")
plt.tight_layout(); plt.savefig("../outputs/listening_spectrograms.png", dpi=130)
print("\nFichiers dans", out, ": L01.wav ... L20.wav, key.csv, fiche_ecoute.csv")
print("NE PAS OUVRIR key.csv avant d'avoir rempli la fiche.")