#!/usr/bin/env python3
"""
extract_logmel.py : represente chaque fenetre de 1 s par un log-mel (64 bandes de 300 a 7000 Hz).

Entrees : windows.f32 (float32 brut, N x 16000)
Sorties : logmel.f32        N x 128 : moyenne et ecart-type de chaque bande sur le temps
          logmel_names.txt  noms des 128 colonnes (lm_mean_00..63, lm_std_00..63)
          logmel_full.npy   (option --full) N x 30 x 64 : le log-mel complet, pour un auto-encodeur

Calcul, pour chaque fenetre :
  1. 30 trames de 1024 echantillons (64 ms), pas de 512 (32 ms), fenetre de Hann, sans bourrage
  2. spectre de puissance (FFT), puis 64 filtres triangulaires espaces en echelle mel entre 300 et 7000 Hz
  3. log en dB : 10 * log10(energie + 1e-10)
  4. par bande : moyenne et ecart-type sur les 30 trames
  5. independance du gain (par defaut) : on retire de toutes les moyennes la moyenne des 64 bandes de la fenetre.
     Un gain multiplie le signal et ajoute le meme decalage en dB a toutes les bandes ; ce decalage disparait.
     --keep-level garde le niveau absolu.
Les fenetres quasi silencieuses (RMS < 1e-7) donnent des zeros.

Exemple : python extract_logmel.py --dir "C:\\...\\windows\\slider"
Ensuite : python diag_unsup.py --dir "C:\\...\\windows\\slider" --features logmel
"""
import argparse
import os
import time

import numpy as np

SR = 16000
WIN = 16000
NFFT = 1024
HOP = 512
N_MELS = 64
FMIN, FMAX = 300.0, 7000.0
SILENCE_RMS = 1e-7
N_FRAMES = 1 + (WIN - NFFT) // HOP        # 30


def mel_filterbank():
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(FMIN), hz2mel(FMAX), N_MELS + 2))
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SR)
    fb = np.zeros((N_MELS, len(freqs)))
    for i in range(N_MELS):
        lo, c, hi = pts[i:i + 3]
        fb[i] = np.maximum(0.0, np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)))
    return fb


FB = mel_filterbank()
HANN = np.hanning(NFFT)


def logmel_batch(w, keep_level=False):
    """w : (B, 16000) -> log-mel (B, 30, 64) en dB, silence = zeros."""
    w = np.asarray(w, dtype=np.float64)
    frames = np.lib.stride_tricks.sliding_window_view(w, NFFT, axis=1)[:, ::HOP]          # (B, 30, 1024)
    power = np.abs(np.fft.rfft(frames * HANN, axis=2)) ** 2                                # (B, 30, 513)
    lm = 10.0 * np.log10(np.einsum("btf,mf->btm", power, FB) + 1e-10)                      # (B, 30, 64)
    if not keep_level:
        lm = lm - lm.mean(axis=(1, 2), keepdims=True)
    silent = np.sqrt(np.mean(w ** 2, axis=1)) < SILENCE_RMS
    lm[silent] = 0.0
    return lm


def summary(lm):
    return np.concatenate([lm.mean(axis=1), lm.std(axis=1)], axis=1)                       # (B, 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="logmel")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="enregistre aussi le log-mel complet (N x 30 x 64)")
    ap.add_argument("--keep-level", action="store_true", help="garde le niveau absolu (par defaut : independant du gain)")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    path = os.path.join(a.dir, "windows.f32")
    n_total = os.path.getsize(path) // (WIN * 4)
    n = min(n_total, a.limit) if a.limit else n_total
    W = np.memmap(path, dtype=np.float32, mode="r").reshape(-1, WIN)

    names = [f"lm_mean_{b:02d}" for b in range(N_MELS)] + [f"lm_std_{b:02d}" for b in range(N_MELS)]
    out = np.zeros((n, 2 * N_MELS), dtype=np.float32)
    full = np.lib.format.open_memmap(os.path.join(a.dir, a.out + "_full.npy"), mode="w+", dtype=np.float32,
                                     shape=(n, N_FRAMES, N_MELS)) if a.full else None
    t0 = time.time()
    for i in range(0, n, a.batch):
        lm = logmel_batch(W[i:i + a.batch], a.keep_level)
        out[i:i + len(lm)] = summary(lm)
        if full is not None:
            full[i:i + len(lm)] = lm
    dt = time.time() - t0
    out.tofile(os.path.join(a.dir, a.out + ".f32"))
    with open(os.path.join(a.dir, a.out + "_names.txt"), "w") as fh:
        fh.write("\n".join(names))
    if full is not None:
        full.flush()
    print(f"{n} fenetres x {out.shape[1]} features -> {os.path.join(a.dir, a.out + '.f32')}")
    print(f"temps : {dt:.1f} s ; NaN : {int(np.isnan(out).sum())} ; fenetres a zero : {int((np.abs(out).sum(1) == 0).sum())}")
    if full is not None:
        print(f"log-mel complet -> {a.out}_full.npy ({n} x {N_FRAMES} x {N_MELS}, a charger avec np.load)")


if __name__ == "__main__":
    main()