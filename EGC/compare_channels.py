"""
compare_channels.py
Compare des configurations de canaux MIMII SANS entraîner de modèle, directement sur les
fichiers bruts (8 canaux), avant EGC et avant tout filtrage.

Deux mesures :
  1) Matrice de corrélation 8x8 entre canaux : vérifie que les micros voisins sont bien les
     plus corrélés (donc que l'ordre des canaux suit le cercle) et mesure la redondance.
  2) Séparabilité normal / anormal avec un détecteur volontairement très simple :
     - on calcule le spectre (log, 64 bandes entre 300 et 7000 Hz) de chaque fichier
     - référence = moyenne et écart-type par bande sur la moitié des fichiers NORMAUX
     - score d'un fichier = distance moyenne (en écarts-types) à ce spectre de référence
     - AUC entre les fichiers normaux restants et les fichiers anormaux
     (0,5 = aucune séparation, 1,0 = séparation parfaite)
  Cette mesure est un indicateur RELATIF entre configurations, pas la performance du modèle final.

Arborescence attendue : <input>/<machine>/id_XX/normal|abnormal/*.wav
  python compare_channels.py --input "C:/.../dataset mimii/-6_dB_pump"
  python compare_channels.py --input ... --configs 0 0,1 2,3 0,4 0,1,2,3,4,5,6,7 --max-files 100

Dépendances : pip install numpy scipy soundfile
"""

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal, stats

FS = 16000


def band_logspec(x, fmin=300, fmax=7000, nb=64):
    """Spectre moyenné en nb bandes, en échelle log."""
    f, p = signal.welch(x, FS, nperseg=1024)
    m = (f >= fmin) & (f <= fmax)
    p = p[m]
    bands = np.array([c.mean() for c in np.array_split(p, nb)])
    return np.log10(bands + 1e-12)


def auc(neg, pos):
    """AUC (Mann-Whitney) : probabilité qu'un fichier anormal ait un score plus élevé."""
    s = np.concatenate([neg, pos])
    r = stats.rankdata(s)
    n0, n1 = len(neg), len(pos)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def load8(path):
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != FS or a.shape[1] != 8:
        raise ValueError(f"{path.name}: sr={sr}, canaux={a.shape[1]} (16000 Hz et 8 canaux attendus)")
    return a


def main():
    p = argparse.ArgumentParser(description="Comparaison de canaux MIMII sans modèle")
    p.add_argument("--input", required=True, help="Dossier ex: .../-6_dB_pump (contient pump/id_XX/...)")
    p.add_argument("--configs", nargs="+", default=["0", "0,1", "2,3", "4,5", "0,4", "0,1,2,3,4,5,6,7"],
                   help="Configurations : listes de canaux séparés par des virgules (moyenne = EGC)")
    p.add_argument("--max-files", type=int, default=100, help="Fichiers max par classe et par id (défaut 100)")
    p.add_argument("--seed", type=int, default=0, help="Graine du tirage des fichiers (change-la pour tester la stabilité)")
    args = p.parse_args()

    configs = [[int(c) for c in s.split(",")] for s in args.configs]
    names = [",".join(map(str, c)) for c in configs]
    if any(c < 0 or c > 7 for cfg in configs for c in cfg):
        raise SystemExit("Les canaux doivent être entre 0 et 7.")

    root = Path(args.input)
    ids = sorted(d for d in root.rglob("id_*") if (d / "normal").is_dir() and (d / "abnormal").is_dir())
    if not ids:
        raise SystemExit(f"Aucun dossier id_XX avec normal/ et abnormal/ trouvé dans {root}")

    rng = random.Random(args.seed)
    corr_sum, corr_n = np.zeros((8, 8)), 0
    auc_table = {n: [] for n in names}
    id_names = []

    for d in ids:
        normal = sorted((d / "normal").glob("*.wav"))
        abnormal = sorted((d / "abnormal").glob("*.wav"))
        rng.shuffle(normal)
        rng.shuffle(abnormal)
        normal, abnormal = normal[:args.max_files], abnormal[:args.max_files]
        if len(normal) < 10 or len(abnormal) < 5:
            print(f"{d.name}: pas assez de fichiers (normal={len(normal)}, abnormal={len(abnormal)}), ignoré")
            continue

        def spectra(files):
            out = {n: [] for n in names}
            for f in files:
                a = load8(f)
                nonlocal_corr(a)
                for n, cfg in zip(names, configs):
                    out[n].append(band_logspec(a[:, cfg].mean(axis=1)))
            return {n: np.array(v) for n, v in out.items()}

        def nonlocal_corr(a):
            nonlocal corr_sum, corr_n
            corr_sum += np.corrcoef(a.T)
            corr_n += 1

        half = len(normal) // 2
        ref, test_n, test_a = spectra(normal[:half]), spectra(normal[half:]), spectra(abnormal)
        id_names.append(d.name)
        for n in names:
            mu, sd = ref[n].mean(axis=0), ref[n].std(axis=0) + 1e-6
            score = lambda S: np.mean(((S - mu) / sd) ** 2, axis=1)
            auc_table[n].append(auc(score(test_n[n]), score(test_a[n])))
        print(f"{d.name} traité : {len(normal)} normaux, {len(abnormal)} anormaux")

    if not id_names:
        raise SystemExit("Aucun id exploitable.")

    # --- 1) corrélation entre canaux ---
    C = corr_sum / corr_n
    print("\n=== Corrélation moyenne entre canaux (fichiers bruts, 8 canaux) ===")
    print("     " + " ".join(f"  c{j}" for j in range(8)))
    for i in range(8):
        print(f"  c{i} " + " ".join(f"{C[i, j]:5.2f}" for j in range(8)))
    nb = np.mean([C[i, (i + 1) % 8] for i in range(8)])
    op = np.mean([C[i, (i + 4) % 8] for i in range(8)])
    if nb - op > 0.02:
        verdict = "ordre circulaire cohérent (voisins plus corrélés)"
    elif op - nb > 0.02:
        verdict = "ATTENTION : ordre des canaux inattendu (opposés plus corrélés)"
    else:
        verdict = "pas de différence nette entre voisins et opposés : on ne peut pas conclure"
    print(f"Voisins (45°) : {nb:.3f} | opposés (180°) : {op:.3f}  -> {verdict}")

    # --- 2) séparabilité ---
    print("\n=== AUC normal/anormal (détecteur spectral simple, plus haut = mieux) ===")
    print("config".ljust(18) + "".join(i.rjust(9) for i in id_names) + "   moyenne")
    for n in names:
        print(n.ljust(18) + "".join(f"{v:9.3f}" for v in auc_table[n]) + f"   {np.mean(auc_table[n]):.3f}")
    print("\nLecture : avec quelques dizaines de fichiers, l'AUC fluctue d'environ 0,05 à 0,10 d'un id à "
          "l'autre (même pour des paires de géométrie identique). Ne conclus que sur un écart net ET "
          "cohérent sur tous les ids. Augmente --max-files pour plus de fiabilité. "
          "Ce test ne remplace pas la comparaison avec le modèle final.")


if __name__ == "__main__":
    main()