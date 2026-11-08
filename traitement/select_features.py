#!/usr/bin/env python3
"""
select_features.py : compare ANOVA et JSD pour choisir les features, en leave-one-id-out.

Protocole (pour chaque machine id laissee de cote, tour par tour) :
  1. train = les autres ids ; test = l'id laisse de cote (jamais vu pour la selection ni l'entrainement)
  2. classement des features SUR LE TRAIN SEULEMENT, par : anova, jsd, importance modele, aleatoire
  3. filtre de redondance (|corr| > seuil -> on garde la mieux classee)
  4. pour chaque k : entrainement sur les k meilleures, F1 macro sur l'id de test
Deux variantes :  all  = les 65 features ;  nolevel = sans les features dependantes du niveau (gain du micro)

Sorties (dans --dir) : select_results.csv, select_curves.png, select_top.txt

Exemple : python select_features.py --dir "C:\\...\\windows\\slider"
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

# features qui dependent du niveau absolu du signal (donc du gain du micro de la carte)
LEVEL = ["t_std", "t_max", "t_min", "t_rms", "t_energy", "t_mean_abs", "t_peak2peak"]


# ---------------------------------------------------------------- scores
def anova_scores(X, y):
    f, _ = f_classif(X, y)
    return np.nan_to_num(f)          # on garde F comme classement, pas la p-value (fenetres correlees)


def jsd_scores(X, y, n_bins=30):
    """Divergence de Jensen-Shannon (base 2, dans [0,1]) entre les histogrammes normal / anormal.
    Binning commun : quantiles des donnees d'entrainement poolees ; lissage de Laplace."""
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        edges = np.unique(np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            continue
        hp, _ = np.histogram(X[y == 0, j], bins=edges)
        hq, _ = np.histogram(X[y == 1, j], bins=edges)
        nb = len(hp)
        p = (hp + 0.5) / (hp.sum() + 0.5 * nb)
        q = (hq + 0.5) / (hq.sum() + 0.5 * nb)
        out[j] = jensenshannon(p, q, base=2) ** 2    # jensenshannon renvoie la racine : on la met au carre
    return out


# ---------------------------------------------------------------- modele
def make_model():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, verbosity=0), "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.1), "sklearn-histgb"


def balanced_weights(y):
    n = len(y)
    w = np.where(y == 1, n / (2 * (y == 1).sum()), n / (2 * (y == 0).sum()))
    return w


def fit_predict(Xtr, ytr, Xte):
    m, _ = make_model()
    m.fit(Xtr, ytr, sample_weight=balanced_weights(ytr))
    return m.predict(Xte)


def importance_scores(Xtr, ytr):
    m, _ = make_model()
    m.fit(Xtr, ytr, sample_weight=balanced_weights(ytr))
    if hasattr(m, "feature_importances_"):
        return m.feature_importances_
    from sklearn.ensemble import RandomForestClassifier   # repli si le modele n'expose pas d'importance
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced", random_state=0)
    rf.fit(Xtr, ytr)
    return rf.feature_importances_


# ---------------------------------------------------------------- selection
def rank_with_redundancy(scores, Xtr, thr):
    """Indices tries du meilleur au pire ; une feature trop correlee (|r|>thr) a une mieux classee est ecartee."""
    order = np.argsort(-scores)
    C = np.abs(np.nan_to_num(np.corrcoef(Xtr, rowvar=False)))
    kept = []
    for j in order:
        if all(C[j, i] <= thr for i in kept):
            kept.append(j)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ks", default="5,10,15,20,30,45")
    ap.add_argument("--corr", type=float, default=0.95, help="seuil de redondance")
    ap.add_argument("--random-draws", type=int, default=3)
    ap.add_argument("--features", default="features", help="prefixe des fichiers de features")
    a = ap.parse_args()

    names = open(os.path.join(a.dir, a.features + "_names.txt")).read().split("\n")
    F = np.fromfile(os.path.join(a.dir, a.features + ".f32"), dtype=np.float32).reshape(-1, len(names))
    meta = pd.read_csv(os.path.join(a.dir, "meta.csv"))
    assert len(meta) == len(F), "meta.csv et features ne sont pas alignes"
    y = (meta.label == "abnormal").values.astype(int)
    ids = meta.id.values
    ks = [int(k) for k in a.ks.split(",")]
    _, model_name = make_model()
    print("modele :", model_name)

    rows, top_counts = [], {}
    for variant in ("all", "nolevel"):
        cols = [i for i, n in enumerate(names) if variant == "all" or n not in LEVEL]
        Xv = F[:, cols]
        nv = [names[i] for i in cols]
        for test_id in sorted(set(ids)):
            tr, te = ids != test_id, ids == test_id
            Xtr, ytr, Xte, yte = Xv[tr], y[tr], Xv[te], y[te]
            rankings = {
                "anova": rank_with_redundancy(anova_scores(Xtr, ytr), Xtr, a.corr),
                "jsd": rank_with_redundancy(jsd_scores(Xtr, ytr), Xtr, a.corr),
                "importance": rank_with_redundancy(importance_scores(Xtr, ytr), Xtr, a.corr),
            }
            for meth, r in rankings.items():            # stabilite : top-20 de chaque tour
                for j in r[:20]:
                    top_counts.setdefault((variant, meth, nv[j]), 0)
                    top_counts[(variant, meth, nv[j])] += 1
            rng = np.random.default_rng(0)
            for k in ks + ["all"]:
                kk = len(nv) if k == "all" else k
                for meth, r in rankings.items():
                    sel = r[:kk]
                    f1 = f1_score(yte, fit_predict(Xtr[:, sel], ytr, Xte[:, sel]), average="macro")
                    rows.append(dict(variant=variant, test_id=test_id, method=meth, k=kk, n_used=len(sel), f1=f1))
                f1s = []
                for _ in range(a.random_draws):
                    sel = rng.choice(len(nv), size=min(kk, len(nv)), replace=False)
                    f1s.append(f1_score(yte, fit_predict(Xtr[:, sel], ytr, Xte[:, sel]), average="macro"))
                rows.append(dict(variant=variant, test_id=test_id, method="random", k=kk, n_used=kk, f1=np.mean(f1s)))
            print(f"[{variant}] id de test {test_id} termine")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.dir, "select_results.csv"), index=False)

    summ = res.groupby(["variant", "method", "k"]).f1.agg(["mean", "std"]).round(3)
    print("\nF1 macro moyen (± ecart entre machines), leave-one-id-out :")
    print(summ.to_string())

    with open(os.path.join(a.dir, "select_top.txt"), "w") as fh:
        for (variant, meth), g in pd.DataFrame(
                [(v, m, n, c) for (v, m, n), c in top_counts.items()],
                columns=["variant", "method", "feature", "count"]).groupby(["variant", "method"]):
            line = ", ".join(f"{r.feature}({r.count}/4)" for r in g.sort_values("count", ascending=False).head(15).itertuples())
            fh.write(f"[{variant}] {meth}: {line}\n")
    print("\nFeatures presentes dans le top-20 (nombre de tours sur 4) -> select_top.txt")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, variant in zip(axes, ("all", "nolevel")):
        for meth in ("anova", "jsd", "importance", "random"):
            g = res[(res.variant == variant) & (res.method == meth)].groupby("k").f1.agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=2, label=meth)
        ax.set_title("65 features" if variant == "all" else "sans features de niveau")
        ax.set_xlabel("k (nombre de features)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("F1 macro (machine jamais vue)")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(a.dir, "select_curves.png"), dpi=130)
    print("Courbes -> select_curves.png")


if __name__ == "__main__":
    main()