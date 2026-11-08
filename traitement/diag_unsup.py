#!/usr/bin/env python3
"""
diag_unsup.py : detection NON supervisee (on n'apprend que le son NORMAL), deux scenarios.

  within_unsup : sur UNE machine, apprendre le normal sur une partie des clips, puis detecter sur des
                 clips jamais vus de la meme machine (= "calibration sur site" : on enregistre le son normal
                 de la machine installee, puis on detecte l'ecart). Decoupage par clip (GroupKFold, 5 plis).
  cross_unsup  : apprendre le normal sur les 3 AUTRES machines, detecter sur la 4e (= un seul modele pour toutes
                 les machines, sans rien apprendre de la nouvelle).

Detecteurs (sur features standardisees avec les stats du normal d'entrainement) :
  zmean   : moyenne des |z| sur les features (le plus simple possible, repere)
  maha    : distance de Mahalanobis au normal (covariance Ledoit-Wolf)
  iforest : Isolation Forest

Metriques : AUC (independante du seuil) et F1 macro avec un seuil fixe = percentile 95 des scores du normal
d'entrainement (seuil choisi sans voir les anomalies).

Exemple : python diag_unsup.py --dir "C:\\...\\windows\\slider"              (sans features de niveau)
          python diag_unsup.py --dir "C:\\...\\windows\\slider" --keep-level (toutes les features)
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from config import feature_columns

warnings.filterwarnings("ignore")

DETECTORS = ["zmean", "maha", "iforest"]


def standardize(Xtr_normal, *Xs):
    mu, sd = Xtr_normal.mean(0), Xtr_normal.std(0) + 1e-9
    return [np.clip((X - mu) / sd, -10, 10) for X in (Xtr_normal,) + Xs]


def fit_score(name, Ztr, Zte):
    """Ztr : fenetres normales d'entrainement (standardisees). Renvoie (scores train, scores test) ; haut = anormal."""
    if name == "zmean":
        f = lambda Z: np.mean(np.abs(Z), axis=1)
        return f(Ztr), f(Zte)
    if name == "maha":
        lw = LedoitWolf().fit(Ztr)
        return lw.mahalanobis(Ztr), lw.mahalanobis(Zte)
    m = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1).fit(Ztr)
    return -m.score_samples(Ztr), -m.score_samples(Zte)


def run(Xtr_normal, Xte, name):
    Ztr, Zte = standardize(Xtr_normal, Xte)
    s_tr, s_te = fit_score(name, Ztr, Zte)
    thr = np.percentile(s_tr, 95)
    return s_te, s_te > thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--features", default="features")
    ap.add_argument("--keep-level", action="store_true", help="conserve les 7 features dependantes du niveau")
    a = ap.parse_args()

    names = open(os.path.join(a.dir, a.features + "_names.txt")).read().split("\n")
    variant = "all" if a.keep_level else "nolevel"
    cols = feature_columns(names, variant)
    X = np.fromfile(os.path.join(a.dir, a.features + ".f32"), dtype=np.float32).reshape(-1, len(names))[:, cols].astype(np.float64)
    meta = pd.read_csv(os.path.join(a.dir, "meta.csv"))
    assert len(meta) == len(X)
    y = (meta.label == "abnormal").values.astype(int)
    ids, clips = meta.id.values, meta["clip"].values
    print(f"{X.shape[1]} features ({variant})")

    rows = []
    for test_id in sorted(set(ids)):
        te = ids == test_id
        Xm, ym, gm = X[te], y[te], clips[te]
        # --- within : normal appris sur une partie des clips de la machine
        for det in DETECTORS:
            s = np.zeros(len(ym))
            pred = np.zeros(len(ym), dtype=bool)
            for tr_i, te_i in GroupKFold(n_splits=5).split(Xm, ym, gm):
                normal_tr = Xm[tr_i][ym[tr_i] == 0]
                s[te_i], pred[te_i] = run(normal_tr, Xm[te_i], det)
            rows.append(dict(protocol="within_unsup", detector=det, test_id=test_id, auc=roc_auc_score(ym, s),
                             f1=f1_score(ym, pred, average="macro"), pred_abn=pred.mean()))
        # --- cross : normal appris sur les autres machines
        normal_other = X[(~te) & (y == 0)]
        for det in DETECTORS:
            s, pred = run(normal_other, Xm, det)
            rows.append(dict(protocol="cross_unsup", detector=det, test_id=test_id, auc=roc_auc_score(ym, s),
                             f1=f1_score(ym, pred, average="macro"), pred_abn=pred.mean()))
        print(f"{test_id} termine")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.dir, f"diag_unsup_{a.features}_{variant}.csv"), index=False)
    for col, title in (("auc", "AUC (0,5 = hasard)"), ("f1", "F1 macro (seuil = percentile 95 du normal d'entrainement)"),
                       ("pred_abn", "Part predite anormale")):
        print(f"\n{title} :")
        print(res.pivot_table(index=["protocol", "detector"], columns="test_id", values=col).round(2).to_string())
    print("\nPart reelle d'anormaux :", pd.Series(y).groupby(ids).mean().round(2).to_dict())
    print("F1 d'un modele qui repond toujours normal :",
          {i: round((1 - p) / (2 - p), 2) for i, p in pd.Series(y).groupby(ids).mean().items()})


if __name__ == "__main__":
    main()