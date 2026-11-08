import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.stats import kruskal, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

d = "../outputs/features/"
y = np.load(d + "y.npy")
names = np.load(d + "names.npy")
tr = np.load(d + "split.npz")["train"]
abn = np.load(d + "abn_idx.npy")
ref = np.load(d + "labels_A_residu_k3.npy")
K = 3

XA = np.load(d + "X_A_logmel.npy")
spec = XA.mean(axis=2)                        # (N, 64) spectre moyen par fichier
mu_norm = spec[tr].mean(axis=0)               # profil du normal (train)
freqs = librosa.mel_frequencies(n_mels=66, fmin=0, fmax=8000)[1:-1]   # centres en Hz

# ===== PARTIE 1 : signature de chaque cluster =====
print("=== 1. Écart au normal par cluster (dB) ===")
diff = {}
for c in range(K):
    diff[c] = spec[abn][ref == c].mean(axis=0) - mu_norm
    top = np.argsort(diff[c])[::-1][:5]
    print(f"\nCluster {c} (n={np.sum(ref == c)}) | écart moyen {diff[c].mean():.1f} dB | "
          f"min {diff[c].min():.1f} | max {diff[c].max():.1f}")
    print("  5 bandes les plus élevées :",
          ", ".join(f"b{b} ({freqs[b]:.0f} Hz, {diff[c][b]:+.1f} dB)" for b in top))

print("\n=== Écarts entre clusters (dB) ===")
for a, b in [(0, 1), (0, 2), (1, 2)]:
    dd = diff[a] - diff[b]
    print(f"Cluster {a} - {b} : moyenne {dd.mean():+.1f} dB | écart-type entre bandes {dd.std():.1f} dB "
          f"| min {dd.min():+.1f} | max {dd.max():+.1f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for c in range(K):
    ax[0].plot(freqs, diff[c], label=f"cluster {c}")
ax[0].set_xscale("symlog", linthresh=500); ax[0].set_xlabel("Fréquence (Hz)")
ax[0].set_ylabel("Écart au normal (dB)"); ax[0].set_title("Cluster - normal")
ax[0].legend(); ax[0].grid(alpha=.3)
for c in range(1, K):
    ax[1].plot(freqs, diff[c] - diff[0], label=f"cluster {c} - cluster 0")
ax[1].set_xscale("symlog", linthresh=500); ax[1].set_xlabel("Fréquence (Hz)")
ax[1].set_ylabel("dB"); ax[1].set_title("Différences entre clusters")
ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("../outputs/cluster_signatures.png", dpi=150); plt.show()

# ===== PARTIE 2 : dérive des fichiers NORMAUX avec l'ordre =====
print("\n=== 2. Dérive des fichiers normaux selon l'ordre d'enregistrement ===")
nor = np.where(y == 0)[0]
nid = np.array([int(os.path.basename(n).split(".")[0]) for n in names[nor]])
order = np.argsort(nid)

lvl = XA[nor].mean(axis=(1, 2))
rho, p = spearmanr(nid, lvl)
print(f"Niveau global vs numéro : Spearman rho={rho:+.3f}, p={p:.2e}")

R = spec[nor] - mu_norm
Z = StandardScaler().fit_transform(R)
P = PCA(n_components=0.95, random_state=0).fit_transform(Z)
rho2, p2 = spearmanr(nid, P[:, 0])
print(f"PC1 du résidu vs numéro : Spearman rho={rho2:+.3f}, p={p2:.2e}")

lab = KMeans(K, n_init=10, random_state=0).fit_predict(P)
ls = lab[order]
same = (ls[1:] == ls[:-1]).mean()
prop = np.bincount(lab) / len(lab)
print(f"Clustering des normaux (k={K}) : voisins consécutifs de même cluster {same:.3f} "
      f"(attendu si ordre indépendant : {np.sum(prop**2):.3f})")
H, pk = kruskal(*[nid[lab == c] for c in range(K)])
print(f"Kruskal-Wallis : H={H:.1f}, p={pk:.2e}")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].scatter(nid, lvl, s=5); ax[0].set_xlabel("Numéro (normal)")
ax[0].set_ylabel("Niveau global (dB)"); ax[0].set_title("Normaux : niveau vs ordre")
ax[1].scatter(nid, P[:, 0], s=5, c=lab, cmap="tab10"); ax[1].set_xlabel("Numéro (normal)")
ax[1].set_ylabel("PC1 du résidu"); ax[1].set_title("Normaux : PC1 vs ordre")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("../outputs/normal_drift.png", dpi=150); plt.show()