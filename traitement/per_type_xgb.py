"""
Modèle XGBoost par type de machine, avec découpage par clip (aucune fuite entre fenêtres d'un même clip).

Protocole (proche du rapport de référence, mais évalué sur le test naturel) :
  - un modèle par type de machine, tous les ids (00, 02, 04, 06) mélangés ;
  - découpage 70 / 15 / 15 stratifié, au niveau du clip (--split clip) ;
  - sous-échantillonnage du normal sur le train seulement (--balance train) ;
  - seuil choisi sur la validation, jamais sur le test ;
  - répété sur plusieurs graines.

Utilisation (depuis la racine du projet) :
  .\\venv\\Scripts\\python.exe .\\traitement\\per_type_xgb.py --root windows --machines slider pump fan valve --variant all
  .\\venv\\Scripts\\python.exe .\\traitement\\per_type_xgb.py --root windows --machines slider --variant nolevel --split window   # contrôle : découpage par fenêtre (avec fuite)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except ImportError:
    sys.exit("xgboost n'est pas installé dans cet environnement : pip install xgboost")

LEVEL = ["t_std", "t_max", "t_min", "t_rms", "t_energy", "t_mean_abs", "t_peak2peak"]
try:
    from config import LEVEL  # noqa: F811  (source unique si config.py est disponible)
except Exception:
    pass

CLIP_CANDIDATES = ["clip", "clip_id", "file", "filename", "path", "wav"]
THRESHOLDS = np.linspace(0.05, 0.95, 19)


def load_machine(root, machine, label_col, id_col, clip_col, wpc):
    d = Path(root) / machine
    names = [l.strip() for l in open(d / "features_names.txt", encoding="utf-8") if l.strip()]
    X = np.fromfile(d / "features.f32", dtype=np.float32).reshape(-1, len(names))
    meta = pd.read_csv(d / "meta.csv")
    if len(meta) != len(X):
        sys.exit(f"{machine}: {len(meta)} lignes dans meta.csv mais {len(X)} dans features.f32")
    for col in (label_col, id_col):
        if col not in meta.columns:
            sys.exit(f"{machine}: colonne '{col}' absente de meta.csv. Colonnes disponibles : {list(meta.columns)}")

    y = meta[label_col]
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).str.lower().str.startswith(("ab", "an")).astype(int)
    y = y.astype(int).values

    lower = {c.lower(): c for c in meta.columns}
    clip = None
    if clip_col:
        clip = meta[clip_col].astype(str).values
    else:
        for cand in CLIP_CANDIDATES:
            if cand in lower:
                clip = meta[lower[cand]].astype(str).values
                break
    if clip is None:
        print(f"[attention] {machine}: pas de colonne clip dans meta.csv ({list(meta.columns)}). "
              f"Je suppose {wpc} fenêtres consécutives par clip.")
        clip = (np.arange(len(meta)) // wpc).astype(str)
    ids = meta[id_col].astype(str).values
    return names, X, y, ids, clip


def select_columns(names, variant):
    return [i for i, n in enumerate(names) if variant == "all" or n not in LEVEL]


def split_masks(y, ids, clip, seed, mode):
    n = len(y)
    if mode == "window":
        idx = np.arange(n)
        tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
        va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
    else:
        clips = pd.DataFrame({"clip": clip, "y": y}).drop_duplicates("clip")
        c_tr, c_tmp = train_test_split(clips, test_size=0.30, stratify=clips["y"], random_state=seed)
        c_va, c_te = train_test_split(c_tmp, test_size=0.50, stratify=c_tmp["y"], random_state=seed)
        tr = np.where(np.isin(clip, c_tr["clip"]))[0]
        va = np.where(np.isin(clip, c_va["clip"]))[0]
        te = np.where(np.isin(clip, c_te["clip"]))[0]
    return tr, va, te


def undersample_normal(idx, y, rng):
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    if len(neg) > len(pos) > 0:
        neg = rng.choice(neg, size=len(pos), replace=False)
    out = np.concatenate([pos, neg])
    rng.shuffle(out)
    return out


def best_threshold(y_true, p):
    scores = [f1_score(y_true, (p >= t).astype(int), average="macro") for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])


def run_one(X, y, ids, clip, seed, mode, balance):
    rng = np.random.default_rng(seed)
    tr, va, te = split_masks(y, ids, clip, seed, mode)
    tr_fit = undersample_normal(tr, y, rng) if balance == "train" else tr

    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=seed, n_jobs=-1,
    )
    model.fit(X[tr_fit], y[tr_fit])

    p_va = model.predict_proba(X[va])[:, 1]
    p_te = model.predict_proba(X[te])[:, 1]
    thr = best_threshold(y[va], p_va)

    te_bal = undersample_normal(te, y, rng)
    p_bal = model.predict_proba(X[te_bal])[:, 1]

    aucs = []
    for i in np.unique(ids[te]):
        m = ids[te] == i
        if len(np.unique(y[te][m])) == 2:
            aucs.append(roc_auc_score(y[te][m], p_te[m]))

    return {
        "auc": roc_auc_score(y[te], p_te),
        "auc_min_id": float(np.min(aucs)) if aucs else np.nan,
        "f1_macro_thr_val": f1_score(y[te], (p_te >= thr).astype(int), average="macro"),
        "f1_macro_0.5": f1_score(y[te], (p_te >= 0.5).astype(int), average="macro"),
        "f1_anom_0.2_balanced": f1_score(y[te_bal], (p_bal >= 0.2).astype(int)),
        "baseline_always_normal": f1_score(y[te], np.zeros(len(te), dtype=int), average="macro"),
        "thr_val": thr,
        "n_test": len(te),
        "anom_ratio_test": float(y[te].mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="windows")
    ap.add_argument("--machines", nargs="+", default=["slider", "pump", "fan", "valve"])
    ap.add_argument("--variant", choices=["all", "nolevel"], default="nolevel")
    ap.add_argument("--split", choices=["clip", "window"], default="clip")
    ap.add_argument("--balance", choices=["train", "none"], default="train")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--clip-col", default=None)
    ap.add_argument("--windows-per-clip", type=int, default=10)
    ap.add_argument("--out", default="results")
    ap.add_argument("--save-model-dir", default=None,
                    help="dossier où sauvegarder un modèle final par machine")
    args = ap.parse_args()

    rows = []
    for machine in args.machines:
        names, X, y, ids, clip = load_machine(
            args.root, machine, args.label_col, args.id_col, args.clip_col, args.windows_per_clip)
        cols = select_columns(names, args.variant)
        Xs = X[:, cols]
        print(f"\n{machine}: {len(y)} fenêtres, {len(np.unique(clip))} clips, "
              f"{len(cols)} features ({args.variant}), split={args.split}")
        for seed in range(args.seeds):
            r = run_one(Xs, y, ids, clip, seed, args.split, args.balance)
            r.update({"machine": machine, "seed": seed})
            rows.append(r)

    df = pd.DataFrame(rows)
    metrics = ["auc", "auc_min_id", "f1_macro_thr_val", "f1_macro_0.5",
               "f1_anom_0.2_balanced", "baseline_always_normal"]
    summary = df.groupby("machine")[metrics].agg(["mean", "std"]).round(3)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"\n=== {args.variant} | split={args.split} | balance={args.balance} | {args.seeds} graines ===")
    print(summary.to_string())

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stem = f"per_type_xgb_{args.variant}_{args.split}"
    df.to_csv(out / f"{stem}.csv", index=False)
    summary.to_csv(out / f"{stem}_summary.csv")
    print(f"\nRésultats écrits dans {out / (stem + '.csv')} et {out / (stem + '_summary.csv')}")

    if args.save_model_dir:
        model_dir = Path(args.save_model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        for machine in args.machines:
            names, X, y, ids, clip = load_machine(
                args.root, machine, args.label_col, args.id_col, args.clip_col, args.windows_per_clip)
            cols = select_columns(names, args.variant)
            Xs = X[:, cols]
            rng = np.random.default_rng(0)
            fit_idx = undersample_normal(np.arange(len(y)), y, rng) if args.balance == "train" else np.arange(len(y))
            model = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=0, n_jobs=-1,
            )
            model.fit(Xs[fit_idx], y[fit_idx])
            machine_rows = df[df.machine == machine]
            threshold = float(machine_rows["thr_val"].mean())
            model_path = model_dir / f"{machine}_xgb_{args.variant}.json"
            metadata_path = model_dir / f"{machine}_xgb_{args.variant}.json.meta"
            model.save_model(model_path)
            metadata_path.write_text(json.dumps({
                "machine": machine,
                "variant": args.variant,
                "feature_count": len(cols),
                "feature_names": [names[i] for i in cols],
                "threshold": threshold,
                "balance": args.balance,
                "training_rows": int(len(fit_idx)),
            }, indent=2), encoding="utf-8")
            print(f"Modèle sauvegardé : {model_path}")
            print(f"Métadonnées sauvegardées : {metadata_path}")


if __name__ == "__main__":
    main()