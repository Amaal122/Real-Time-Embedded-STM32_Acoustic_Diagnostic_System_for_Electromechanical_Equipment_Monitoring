"""
prepare_windows.py
Prépare les données d'entraînement directement à partir des clips MIMII bruts (8 canaux) :

  pour chaque clip :
    1. tirage au hasard de K paires de micros VOISINS (noté dans les métadonnées)
    2. EGC : moyenne des deux canaux de la paire
    3. filtre Butterworth 300-7000 Hz (ordre 4 = 2 biquads, causal)
    4. découpe en fenêtres de 1 s (10 par clip de 10 s)

Sorties (dans --output) :
  windows.f32   fenêtres, float32, à relire avec np.fromfile(...).reshape(-1, 16000)
  meta.csv      une ligne par fenêtre : row, machine, id, label, clip, pair_a, pair_b, win

Règles à respecter ensuite : découper train/validation/test PAR MACHINE (colonne id),
jamais par fenêtre ni par clip. Toutes les fenêtres et variantes d'un même clip restent ensemble.

Exemples :
  python prepare_windows.py --input "C:/.../dataset mimii/-6_dB_pump" --output "C:/.../windows/pump"
  python prepare_windows.py --input ... --output ... --variants 2          # 2 paires différentes par clip
  python prepare_windows.py --input ... --output ... --pairs 1,2           # paire fixe (1,2)
  python prepare_windows.py --input ... --output ... --pairs 0,1 1,2 2,3   # tirage dans une liste

Dépendances : pip install numpy scipy soundfile
"""

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

FS = 16000
WIN = FS            # fenêtre de 1 s
SOS = signal.butter(2, [300.0, 7000.0], btype="bandpass", fs=FS, output="sos")  # ordre 4 = 2 biquads


def filt(x):
    """Filtrage causal ; l'état initial est réglé sur le 1er échantillon (réduit le transitoire)."""
    zi = signal.sosfilt_zi(SOS) * x[0]
    return signal.sosfilt(SOS, x, zi=zi)[0]


def main():
    p = argparse.ArgumentParser(description="Clips MIMII 8 canaux -> fenêtres 1 s (paire aléatoire)")
    p.add_argument("--input", required=True, help="Dossier ex: .../-6_dB_pump (contient pump/id_XX/...)")
    p.add_argument("--output", required=True)
    p.add_argument("--variants", type=int, default=1, help="Paires différentes tirées par clip (défaut 1)")
    p.add_argument("--pairs", nargs="+", default=None,
                   help="Paires autorisées, ex: 0,1 1,2 (défaut : les 8 paires de voisins)")
    p.add_argument("--seed", type=int, default=0, help="Graine du tirage (résultats reproductibles)")
    args = p.parse_args()

    pairs = [tuple(int(v) for v in s.split(",")) for s in args.pairs] if args.pairs \
        else [(i, (i + 1) % 8) for i in range(8)]
    if any(len(pr) != 2 or min(pr) < 0 or max(pr) > 7 or pr[0] == pr[1] for pr in pairs):
        raise SystemExit("Chaque paire doit être de la forme a,b avec a et b différents, entre 0 et 7.")
    if args.variants > len(pairs):
        raise SystemExit(f"--variants ({args.variants}) dépasse le nombre de paires disponibles ({len(pairs)}).")

    in_root, out_root = Path(args.input), Path(args.output)
    files = sorted(in_root.rglob("*.wav"))
    if not files:
        raise SystemExit(f"Aucun fichier .wav trouvé dans {in_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    needed = len(files) * args.variants * 10 * WIN * 4
    free = shutil.disk_usage(out_root).free
    print(f"{len(files)} clips | {args.variants} variante(s) | {len(pairs)} paire(s) possibles")
    print(f"Espace disque libre : {free / 1e9:.1f} Go | nécessaire (estimé) : {needed / 1e9:.1f} Go")
    if free < needed * 1.05:
        raise SystemExit("Espace disque insuffisant : libère de la place avant de relancer.")

    rng = np.random.default_rng(args.seed)
    n_rows, errors, consecutive = 0, [], 0
    pair_count, label_count = Counter(), Counter()

    with open(out_root / "windows.f32", "wb") as fw, \
         open(out_root / "meta.csv", "w", newline="", encoding="utf-8") as fm:
        meta = csv.writer(fm)
        meta.writerow(["row", "machine", "id", "label", "clip", "pair_a", "pair_b", "win"])

        for i, f in enumerate(files, 1):
            try:
                audio, sr = sf.read(f, dtype="float32", always_2d=True)
                if sr != FS or audio.shape[1] != 8:
                    raise ValueError(f"sr={sr}, canaux={audio.shape[1]} (16000 Hz et 8 canaux attendus)")
                n_win = audio.shape[0] // WIN
                if n_win != 10:
                    raise ValueError(f"{n_win} fenêtres au lieu de 10")

                rel = f.relative_to(in_root).parts          # machine / id_XX / normal|abnormal / fichier
                machine, id_, label, name = rel[0], rel[1], rel[2], rel[3]
                chosen = rng.choice(len(pairs), size=args.variants, replace=False)

                for ci in chosen:
                    a, b = pairs[ci]
                    y = filt(audio[:, [a, b]].astype(np.float64).mean(axis=1))
                    w = y[:n_win * WIN].reshape(n_win, WIN).astype(np.float32)
                    fw.write(w.tobytes())
                    for k in range(n_win):
                        meta.writerow([n_rows + k, machine, id_, label, f"{id_}/{label}/{name}", a, b, k])
                    n_rows += n_win
                    pair_count[(a, b)] += 1
                    label_count[label] += n_win
                consecutive = 0
            except Exception as e:
                errors.append((str(f), f"{type(e).__name__}: {e}"))
                consecutive += 1
                if consecutive >= 20:
                    print("\nARRÊT : 20 erreurs consécutives (disque plein ou fichiers illisibles ?).")
                    break
            if i % 200 == 0 or i == len(files):
                print(f"  {i}/{len(files)} clips traités")

    print(f"\nTerminé : {n_rows} fenêtres écrites, {len(errors)} erreur(s).")
    print("Fenêtres par étiquette :", dict(label_count))
    print("Paires tirées (nombre de clips) :", {f"{a},{b}": c for (a, b), c in sorted(pair_count.items())})
    for path, err in errors[:10]:
        print(f"  ERREUR {path} -> {err}")


if __name__ == "__main__":
    main()
