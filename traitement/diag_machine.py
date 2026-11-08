#!/usr/bin/env python3
"""
diag_machine.py : pourquoi une machine (ex. id_06) a un mauvais F1 ? seuil ou vraie absence de separation ?

Pour chaque machine, trois configurations avec la variante de features choisie :
  cross_raw  : entraine sur les 3 autres machines, teste sur celle-ci (le cas deja mesure)
  cross_norm : pareil, mais les features de CHAQUE machine sont centrees-reduites avec les
               fenetres NORMALES de cette machine seulement (= une calibration sur place)
  within_raw : entraine et teste sur la MEME machine, decoupage par clip (GroupKFold, 5 plis)
               -> modele par machine ; les 10 fenetres d'un clip restent du meme cote

Metriques par machine :
  auc      : separation normal/anormal, independante du seuil (0,5 = hasard, 1 = parfait)
  f1       : F1 macro au seuil 0,5 (la mesure deja utilisee)
  pred_abn : part de fenetres predites anormales ; a comparer a true_abn (part reelle)
  baseline : F1 macro d'un modele qui repond toujours "normal"

Exemple : python diag_machine.py --dir "C:\\...\\windows\\slider" --variant nolevel
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from config import feature_columns
from select_features import balanced_weights, make_model

warnings.filterwarnings("ignore")


def fit_proba(Xtr, ytr, Xte):
    m, _ = make_model()
    m.fit(Xtr, ytr, sample_weight=balanced_weights(ytr))
    return m.predict_proba(Xte)[:, 1]


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    pa = y.mean()
    return dict(auc=roc_auc_score(y, p), f1=f1_score(y, pred, average="macro"),
                pred_abn=pred.mean(), true_abn=pa,
                baseline=(1 - pa) / (2 - pa))   # F1 macro de "toujours normal" : (F1 normal + 0) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--features", default="features")
    ap.add_argument("--variant", choices=("all", "nolevel"), default="nolevel")
    a = ap.parse_args()

    names = open(os.path.join(a.dir, a.features + "_names.txt")).read().split("\n")
    cols = feature_columns(names, a.variant)
    X = np.fromfile(os.path.join(a.dir, a.features + ".f32"), dtype=np.float32).reshape(-1, len(names))[:, cols].astype(np.float64)
    meta = pd.read_csv(os.path.join(a.dir, "meta.csv"))
    assert len(meta) == len(X)
    y = (meta.label == "abnormal").values.astype(int)
    ids = meta.id.values

    # normalisation par machine, calculee sur les fenetres normales de la machine
    Xn = X.copy()
    for i in np.unique(ids):
        m = ids == i
        ref = X[m & (y == 0)]
        Xn[m] = (X[m] - ref.mean(0)) / (ref.std(0) + 1e-9)

    rows = []
    for test_id in sorted(set(ids)):
        te, tr = ids == test_id, ids != test_id
        rows.append(dict(test_id=test_id, config="cross_raw", **metrics(y[te], fit_proba(X[tr], y[tr], X[te]))))
        rows.append(dict(test_id=test_id, config="cross_norm", **metrics(y[te], fit_proba(Xn[tr], y[tr], Xn[te]))))
        Xi, yi, gi = X[te], y[te], meta["clip"].values[te]
        p = np.zeros(len(yi))
        for a_idx, b_idx in GroupKFold(n_splits=5).split(Xi, yi, gi):
            p[b_idx] = fit_proba(Xi[a_idx], yi[a_idx], Xi[b_idx])
        rows.append(dict(test_id=test_id, config="within_raw", **metrics(yi, p)))
        print(f"{test_id} termine")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.dir, f"diag_machine_{a.variant}.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nAUC (0,5 = hasard) :")
    print(res.pivot(index="config", columns="test_id", values="auc").round(2).to_string())
    print("\nF1 macro au seuil 0,5 :")
    print(res.pivot(index="config", columns="test_id", values="f1").round(2).to_string())
    print("\nPart predite anormale (a comparer a la part reelle) :")
    print(res.pivot(index="config", columns="test_id", values="pred_abn").round(2).to_string())
    print("part reelle d'anormaux :", res[res.config == "cross_raw"].set_index("test_id").true_abn.round(2).to_dict())
    print("F1 d'un modele qui repond toujours normal :", res[res.config == "cross_raw"].set_index("test_id").baseline.round(2).to_dict())


if __name__ == "__main__":
    main()