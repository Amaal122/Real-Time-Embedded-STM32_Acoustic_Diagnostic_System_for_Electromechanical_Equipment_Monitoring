import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

d = "../outputs/features/"
y = np.load(d + "y.npy")
sp = np.load(d + "split.npz")
tr = sp["train"]
abn = np.where(y == 1)[0]          # les 356 anomalies
print("Anomalies :", len(abn))

def prep(F):
    Z = StandardScaler().fit_transform(F)
    return PCA(n_components=0.95, random_state=0).fit_transform(Z)

# --- Représentations par variante ---
def reps(V):
    X = np.load(d + {"A": "X_A_logmel.npy", "B": "X_B_logmel_nolevel.npy"}[V])
    mu_norm = X[tr].mean(axis=(0, 2))                  # profil moyen du normal (train)
    resid = X[abn].mean(axis=2) - mu_norm              # résidu spectral (64)
    band = np.load(d + f"bandErr_{V}.npy")[abn]        # erreur par bande (64)
    emb = np.load(d + f"emb_{V}.npy")[abn]             # embedding latent (8)
    return {"residu": resid, "bandErr": band, "emb": emb}

results = {}
for V in ["A", "B"]:
    for name, F in reps(V).items():
        P = prep(F)
        print(f"\n=== {V} | {name} | PCA -> {P.shape[1]} composantes ===")
        best_k, best_s = None, -1
        for k in range(2, 9):
            lab = KMeans(k, n_init=10, random_state=0).fit_predict(P)
            s = silhouette_score(P, lab); db = davies_bouldin_score(P, lab)
            print(f"k={k} | silhouette {s:.3f} | Davies-Bouldin {db:.3f}")
            if s > best_s: best_k, best_s = k, s
        results[(V, name)] = (P, best_k)
        print("k optimal (silhouette) :", best_k)

# --- Comparaison A vs B à k=3 (repère théorique) pour chaque représentation ---
print("\n=== ARI entre A et B (k=3) ===")
for name in ["residu", "bandErr", "emb"]:
    la = KMeans(3, n_init=10, random_state=0).fit_predict(results[("A", name)][0])
    lb = KMeans(3, n_init=10, random_state=0).fit_predict(results[("B", name)][0])
    print(f"{name:8s} : ARI = {adjusted_rand_score(la, lb):.3f}")

# --- Visualisation 2D (bandErr, variante A) ---
import umap
P, k = results[("A", "bandErr")]
lab = KMeans(k, n_init=10, random_state=0).fit_predict(P)
E = umap.UMAP(random_state=0).fit_transform(P)
plt.scatter(E[:, 0], E[:, 1], c=lab, cmap="tab10", s=12)
plt.title(f"Anomalies : bandErr (A), k={k}")
plt.savefig("../outputs/clusters_A_bandErr.png", dpi=150)
plt.show()