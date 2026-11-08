import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

d = "../outputs/features/"
K = 3

y = np.load(d + "y.npy")
names = np.load(d + "names.npy")
tr = np.load(d + "split.npz")["train"]
abn = np.where(y == 1)[0]
file_id = np.array([int(os.path.basename(n).split(".")[0]) for n in names[abn]])

def prep(F):
    Z = StandardScaler().fit_transform(F)
    return PCA(n_components=0.95, random_state=0).fit_transform(Z)

# --- Référence : résidu spectral, variante A ---
XA = np.load(d + "X_A_logmel.npy")
mu_norm = XA[tr].mean(axis=(0, 2))
P = prep(XA[abn].mean(axis=2) - mu_norm)

ref = KMeans(K, n_init=10, random_state=0).fit_predict(P)
np.save(d + "labels_A_residu_k3.npy", ref)
np.save(d + "abn_idx.npy", abn)
print("Taille des clusters :", np.bincount(ref))

# ===== 1. STABILITÉ =====
print("\n=== 1. Stabilité ===")
aris = [adjusted_rand_score(ref, KMeans(K, n_init=10, random_state=s).fit_predict(P))
        for s in range(1, 21)]
print(f"Graines K-means (20) : ARI moyen {np.mean(aris):.3f} (min {np.min(aris):.3f})")

rng = np.random.default_rng(0)
aris_sub = []
for _ in range(50):
    sub = rng.choice(len(P), int(0.8 * len(P)), replace=False)
    km = KMeans(K, n_init=10, random_state=0).fit(P[sub])
    aris_sub.append(adjusted_rand_score(ref, km.predict(P)))
print(f"Sous-échantillons 80 % (50) : ARI moyen {np.mean(aris_sub):.3f} "
      f"(min {np.min(aris_sub):.3f})")

# Accord avec les autres représentations (variante A)
for name in ["emb", "bandErr"]:
    Pn = prep(np.load(d + f"{name}_A.npy")[abn])
    ln = KMeans(K, n_init=10, random_state=0).fit_predict(Pn)
    print(f"ARI (résidu vs {name:7s}) : {adjusted_rand_score(ref, ln):.3f}")

# ===== 2. ARTEFACTS : ordre des fichiers =====
print("\n=== 2. Numéro de fichier par cluster ===")
for c in range(K):
    ids = file_id[ref == c]
    print(f"Cluster {c} : n={len(ids):3d} | min {ids.min():4d} | "
          f"médiane {int(np.median(ids)):4d} | max {ids.max():4d}")

H, p = kruskal(*[file_id[ref == c] for c in range(K)])
print(f"Kruskal-Wallis (numéro vs cluster) : H={H:.1f}, p={p:.2e}")

order = np.argsort(file_id)
ls = ref[order]
same = (ls[1:] == ls[:-1]).mean()
prop = np.bincount(ref) / len(ref)
print(f"Voisins consécutifs de même cluster : {same:.3f} "
      f"(attendu si ordre indépendant : {np.sum(prop**2):.3f})")

# ===== 3. Niveau global par cluster =====
print("\n=== 3. Niveau global (dB) par cluster ===")
lvl = XA[abn].mean(axis=(1, 2))
for c in range(K):
    print(f"Cluster {c} : {lvl[ref == c].mean():.1f} dB (+/- {lvl[ref == c].std():.1f})")

# --- Figure : cluster en fonction du numéro de fichier ---
plt.figure(figsize=(9, 3.5))
plt.scatter(file_id, ref, c=ref, cmap="tab10", s=14)
plt.yticks(range(K)); plt.xlabel("Numéro de fichier (anormal)")
plt.ylabel("Cluster"); plt.title("Cluster vs ordre d'enregistrement")
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("../outputs/cluster_vs_fileid.png", dpi=150)
plt.show()