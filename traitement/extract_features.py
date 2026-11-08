#!/usr/bin/env python3
"""
extract_features.py : calcule les features de chaque fenetre de 1 s.

Entrees  : windows.f32 (float32 brut, N x 16000) + meta.csv (une ligne par fenetre)
Sorties  : features.f32  (float32, N x F, meme ordre de lignes que windows.f32)
           feature_names.txt (un nom par colonne, dans l'ordre)

Trois groupes, activables separement (pour l'ablation) :
  temporal : 12 features sur la forme d'onde
  spectral :  5 features sur le spectre de puissance (FFT)
  wpt      : 48 features = 16 sous-bandes x 3 stats (wavelet packet coif5, niveau 4)

Exemples :
  python extract_features.py --dir slider --groups temporal,spectral,wpt
  python extract_features.py --dir slider --groups temporal,spectral --out feats_ts
  python extract_features.py --dir slider --limit 200        # test rapide + chronometre
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy import stats

FS = 16000          # frequence d'echantillonnage (Hz)
WIN = 16000         # echantillons par fenetre
EPS = 1e-12         # protection contre les divisions par 0 (fenetres quasi silencieuses)
SILENCE_RMS = 1e-7  # en dessous : fenetre consideree silencieuse (voir _is_silent)

WAVELET = "coif5"
LEVEL = 4
WPT_MODE = "symmetric"   # traitement des bords : a reproduire a l'identique en C
N_BANDS = 2 ** LEVEL     # 16 sous-bandes de 500 Hz


# --------------------------------------------------------------------------
# Groupe 1 : 12 features temporelles
# --------------------------------------------------------------------------
TEMPORAL_NAMES = [
    "t_mean", "t_std", "t_max", "t_min", "t_rms", "t_kurtosis", "t_skewness",
    "t_crest", "t_energy", "t_zcr", "t_mean_abs", "t_peak2peak",
]


def temporal_features(x):
    mean = x.mean()
    std = x.std()
    rms = np.sqrt(np.mean(x ** 2))
    mean_abs = np.mean(np.abs(x))
    peak = np.max(np.abs(x))
    # kurtosis (Fisher, 0 pour une gaussienne) et skewness : 0 si le signal est constant
    if std < EPS:
        kurt, skew = 0.0, 0.0
    else:
        kurt = stats.kurtosis(x, fisher=True, bias=True)
        skew = stats.skew(x, bias=True)
    crest = peak / (rms + EPS)
    energy = np.sum(x ** 2)
    zcr = np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))   # taux de passages par zero
    return np.array([mean, std, x.max(), x.min(), rms, kurt, skew,
                     crest, energy, zcr, mean_abs, x.max() - x.min()])


# --------------------------------------------------------------------------
# Groupe 2 : 5 features spectrales
# --------------------------------------------------------------------------
SPECTRAL_NAMES = ["s_centroid", "s_bandwidth", "s_rolloff85", "s_flatness", "s_peak_ratio"]
_FREQS = np.fft.rfftfreq(WIN, 1.0 / FS)
_HANN = np.hanning(WIN)


def spectral_features(x):
    p = np.abs(np.fft.rfft(x * _HANN)) ** 2          # spectre de puissance (fenetre de Hann)
    tot = p.sum() + EPS
    pn = p / tot                                      # spectre normalise (somme = 1)
    centroid = np.sum(_FREQS * pn)
    bandwidth = np.sqrt(np.sum(((_FREQS - centroid) ** 2) * pn))
    rolloff = _FREQS[np.searchsorted(np.cumsum(pn), 0.85)]     # frequence sous laquelle 85 % de l'energie
    flatness = np.exp(np.mean(np.log(p + EPS))) / (np.mean(p) + EPS)   # 1 = bruit blanc, 0 = tonal
    peak_ratio = pn.max()                              # part de l'energie dans la raie dominante
    return np.array([centroid, bandwidth, rolloff, flatness, peak_ratio])


# --------------------------------------------------------------------------
# Groupe 3 : 48 features wavelet packet (16 sous-bandes x 3 stats)
# --------------------------------------------------------------------------
WPT_NAMES = [f"w{b:02d}_{s}" for b in range(N_BANDS) for s in ("rms_rel", "kurt", "entropy")]


def band_stats(coeffs):
    """coeffs : liste de 16 tableaux (un par sous-bande, tries par frequence croissante).
    Retourne 48 valeurs : par bande (energie relative, kurtosis, entropie de Shannon)."""
    energies = np.array([np.sum(c ** 2) for c in coeffs])
    total = energies.sum() + EPS
    out = []
    for c, e in zip(coeffs, energies):
        rms_rel = e / total                              # part de l'energie totale (sans unite)
        if np.std(c) < EPS:
            kurt = 0.0
        else:
            kurt = stats.kurtosis(c, fisher=True, bias=True)
        # entropie de Shannon de la distribution d'energie des coefficients, normalisee dans [0,1]
        q = c ** 2 / (e + EPS)
        ent = -np.sum(q * np.log2(q + EPS)) / np.log2(len(c))
        out += [rms_rel, kurt, ent]
    return np.array(out)


def wpt_features(x):
    import pywt   # import local : evite de le requerir si le groupe n'est pas demande
    wp = pywt.WaveletPacket(data=x, wavelet=WAVELET, mode=WPT_MODE, maxlevel=LEVEL)
    nodes = wp.get_level(LEVEL, order="freq")            # tries par frequence, pas par ordre naturel
    return band_stats([n.data for n in nodes])


# --------------------------------------------------------------------------
GROUPS = {
    "temporal": (TEMPORAL_NAMES, temporal_features),
    "spectral": (SPECTRAL_NAMES, spectral_features),
    "wpt": (WPT_NAMES, wpt_features),
}


def _is_silent(x):
    return np.sqrt(np.mean(x ** 2)) < SILENCE_RMS


def window_features(x, groups):
    """Toutes les features d'une fenetre, dans l'ordre des groupes demandes."""
    x = np.asarray(x, dtype=np.float64)
    n = sum(len(GROUPS[g][0]) for g in groups)
    if _is_silent(x):                                     # fenetre vide : on evite des valeurs aberrantes
        return np.zeros(n)
    f = np.concatenate([GROUPS[g][1](x) for g in groups])
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


# --- parallelisation : chaque processus ouvre le fichier en memmap -----------
_W = None


def _init(path, groups):
    global _W, _G
    _W = np.memmap(path, dtype=np.float32, mode="r").reshape(-1, WIN)
    _G = groups


def _work(i):
    return window_features(_W[i], _G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dossier contenant windows.f32 et meta.csv")
    ap.add_argument("--groups", default="temporal,spectral,wpt")
    ap.add_argument("--out", default="features", help="prefixe des fichiers de sortie")
    ap.add_argument("--limit", type=int, default=0, help="ne traiter que les N premieres fenetres")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"groupe inconnu : {g} (choix : {list(GROUPS)})")
    path = os.path.join(a.dir, "windows.f32")
    n_total = os.path.getsize(path) // (WIN * 4)
    n = min(n_total, a.limit) if a.limit else n_total
    names = [nm for g in groups for nm in GROUPS[g][0]]

    t0 = time.time()
    with Pool(a.jobs, initializer=_init, initargs=(path, groups)) as pool:
        rows = pool.map(_work, range(n), chunksize=64)
    feats = np.array(rows, dtype=np.float32)
    dt = time.time() - t0

    out_f = os.path.join(a.dir, a.out + ".f32")
    feats.tofile(out_f)
    with open(os.path.join(a.dir, a.out + "_names.txt"), "w") as fh:
        fh.write("\n".join(names))
    print(f"{n} fenetres x {len(names)} features -> {out_f}")
    print(f"temps : {dt:.1f} s ({1000 * dt * a.jobs / n:.2f} ms/fenetre/coeur)")
    print(f"NaN : {int(np.isnan(feats).sum())}  ;  fenetres silencieuses : "
          f"{int((np.abs(feats).sum(1) == 0).sum())}")


if __name__ == "__main__":
    main()