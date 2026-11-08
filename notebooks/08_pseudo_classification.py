import os
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

d = "../outputs/features/"
y = np.load(d + "y.npy")
names = np.load(d + "names.npy")
abn = np.load(d + "abn_idx.npy")
lab = np.load(d + "labels_A_residu_k3.npy")
sp = np.load(d + "split.npz")
XA = np.load(d + "X_A_logmel.npy")[abn]
XB = np.load(d + "X_B_logmel_nolevel.npy")[abn]
fid = np.array([int(os.path.basename(n).split(".")[0]) for n in names[abn]])

# Groupes de fichiers contigus (~18 fichiers) pour la validation par blocs
groups = fid // 18
print("Anomalies :", len(lab), "| classes :", np.bincount(lab),
      "| classe majoritaire :", round(np.bincount(lab).max() / len(lab), 3))

feats = {
    "moyenne (circulaire)": XA.mean(axis=2),
    "ecart-type temporel":  XA.std(axis=2),
    "moyenne + ecart-type": np.hstack([XA.mean(axis=2), XA.std(axis=2)]),
    "moyenne sans niveau":  XB.mean(axis=2),
    "numero de fichier":    fid.reshape(-1, 1).astype(float),
}
models = {
    "SVM": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
    "RF":  lambda: RandomForestClassifier(n_estimators=300, random_state=0),
}
cv_rand = StratifiedKFold(5, shuffle=True, random_state=0)
cv_block = GroupKFold(5)

def run(F, mk, cv, grp=None):
    p = cross_val_predict(mk(), F, lab, cv=cv, groups=grp)
    return p, accuracy_score(lab, p), balanced_accuracy_score(lab, p)

print("\n=== Exactitude (équilibrée) en validation croisée ===")
print(f"{'features':24s} {'modèle':4s} | {'aléatoire':>16s} | {'par blocs':>16s}")
for fn, F in feats.items():
    for mn, mk in models.items():
        _, a1, b1 = run(F, mk, cv_rand)
        _, a2, b2 = run(F, mk, cv_block, groups)
        print(f"{fn:24s} {mn:4s} | {a1:.3f} ({b1:.3f}) | {a2:.3f} ({b2:.3f})")

# --- Matrice de confusion : moyenne + écart-type, SVM, par blocs ---
F = feats["moyenne + ecart-type"]
pred, a, b = run(F, models["SVM"], cv_block, groups)
print("\nMatrice de confusion (lignes = cluster réel, colonnes = prédit) :")
print(confusion_matrix(lab, pred))

# ===== PIPELINE COMPLET : détection puis attribution =====
print("\n=== Pipeline complet (autoencodeur A, graine 0) ===")
scores = np.load(d + "scores_A.npy")
thr = np.percentile(scores[sp["val"]], 95)
tn = sp["test_norm"]
fpr = (scores[tn] > thr).mean()
det = scores[abn] > thr
print(f"Seuil (95e percentile val) : {thr:.4f}")
print(f"Faux positifs (test normal)  : {fpr:.3f} ({int((scores[tn] > thr).sum())}/{len(tn)})")
print(f"Anomalies détectées          : {det.mean():.3f} ({int(det.sum())}/{len(det)})")
prec = det.sum() / (det.sum() + (scores[tn] > thr).sum())
print(f"Précision de l'alarme        : {prec:.3f}")
print(f"Exactitude d'attribution sur les anomalies détectées : "
      f"{accuracy_score(lab[det], pred[det]):.3f}")
for c in range(3):
    m = det & (lab == c)
    print(f"  cluster {c} : détectées {m.sum()}/{(lab == c).sum()}")

np.save(d + "pred_cluster_oof.npy", pred)