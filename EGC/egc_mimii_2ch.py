"""
egc_mimii_2ch.py
Réduit les fichiers MIMII (8 canaux, 16 kHz) à 2 canaux, pour simuler la paire
stéréo U2/U6 de la STM32F469I-DISCO (micros à ~25 mm l'un de l'autre).

Chaque canal de sortie est un "micro virtuel" = moyenne (EGC) d'un groupe de canaux MIMII.
  - groupe d'un seul canal  -> simple sélection (fidèle à la carte : 1 micro = 1 canal)
  - groupe de plusieurs canaux -> EGC (moyenne), attention : améliore le SNR de ce micro virtuel

Exemples :
  # 1) Stéréo fidèle à la carte : micros MIMII #1 et #2 (canaux 0 et 1, 26 mm)
  python egc_mimii_2ch.py --input "D:/MIMII/-6_dB_pump" --output "D:/MIMII_stereo/-6_dB_pump"

  # 2) Idem, puis EGC des deux canaux -> mono (équivalent de (gauche+droite)/2 sur la carte)
  python egc_mimii_2ch.py --input ... --output ... --mono

  # 3) EGC 8 -> 2 : 4 canaux par micro virtuel
  python egc_mimii_2ch.py --input ... --output ... --left 0 1 2 3 --right 4 5 6 7

Dépendances : pip install numpy soundfile
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def virtual_mic(audio: np.ndarray, group: list[int]) -> np.ndarray:
    """EGC d'un groupe de canaux : moyenne simple. audio : (n_samples, n_channels)."""
    return np.mean(audio[:, group], axis=1)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def main():
    p = argparse.ArgumentParser(description="MIMII 8 canaux -> 2 canaux (EGC par groupe)")
    p.add_argument("--input", required=True, help="Dossier racine MIMII (ex: .../-6_dB_pump)")
    p.add_argument("--output", required=True, help="Dossier de sortie (arborescence reproduite)")
    p.add_argument("--left", type=int, nargs="+", default=[0],
                   help="Canaux MIMII formant le canal gauche (défaut : 0)")
    p.add_argument("--right", type=int, nargs="+", default=[1],
                   help="Canaux MIMII formant le canal droit (défaut : 1)")
    p.add_argument("--mono", action="store_true",
                   help="Moyenne les 2 canaux de sortie -> fichier mono (EGC à 2 canaux)")
    args = p.parse_args()

    # --- Vérifications des arguments -------------------------------------------------
    everything = args.left + args.right
    if any(c < 0 or c > 7 for c in everything):
        raise SystemExit("Les indices de canaux doivent être compris entre 0 et 7.")
    if set(args.left) & set(args.right):
        raise SystemExit("Les groupes gauche et droit ne doivent pas partager de canal.")
    if len(args.left) > 1 or len(args.right) > 1:
        print("ATTENTION : un groupe de plusieurs canaux améliore le SNR de ce micro virtuel "
              "(~10*log10(N) dB au mieux). Les micros réels de la carte n'ont pas ce gain : "
              "le modèle verrait des données plus propres à l'entraînement qu'au déploiement.")

    in_root, out_root = Path(args.input), Path(args.output)
    files = sorted(in_root.rglob("*.wav"))
    if not files:
        raise SystemExit(f"Aucun fichier .wav trouvé dans {in_root}")

    mode = "mono (EGC des 2 canaux)" if args.mono else "stéréo (2 canaux)"
    print(f"{len(files)} fichiers | gauche={args.left} droite={args.right} | sortie : {mode}")

    ok, errors = 0, []
    for i, f in enumerate(files, 1):
        try:
            audio, sr = sf.read(f, dtype="float32", always_2d=True)  # (n_samples, 8)

            if audio.shape[1] <= max(everything):
                raise ValueError(f"{audio.shape[1]} canaux seulement dans le fichier")

            left = virtual_mic(audio, args.left)
            right = virtual_mic(audio, args.right)

            if args.mono:
                out = np.mean(np.stack([left, right], axis=1), axis=1)   # (n_samples,)
            else:
                out = np.stack([left, right], axis=1)                    # (n_samples, 2)

            out_path = out_root / f.relative_to(in_root)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, out, sr, subtype="FLOAT")
            ok += 1

            # Contrôle sur le tout premier fichier
            if i == 1:
                corr = float(np.corrcoef(left, right)[0, 1])
                print(f"[contrôle] {f.name}: sr={sr} Hz, entrée={audio.shape}, sortie={out.shape}, "
                      f"durée={len(left)/sr:.1f} s")
                print(f"[contrôle] RMS gauche={rms(left):.5f} | RMS droite={rms(right):.5f} "
                      f"| corrélation G/D={corr:.3f}")
                if sr != 16000:
                    print(f"[contrôle] ATTENTION : fréquence {sr} Hz (16000 attendu pour MIMII)")

        except Exception as e:
            errors.append((str(f), str(e)))

        if i % 200 == 0 or i == len(files):
            print(f"  {i}/{len(files)} traités")

    print(f"\nTerminé : {ok} fichiers convertis, {len(errors)} erreurs.")
    for path, err in errors[:10]:
        print(f"  ERREUR {path} -> {err}")


if __name__ == "__main__":
    main()
