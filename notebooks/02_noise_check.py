import glob, os
import numpy as np
import soundfile as sf

base = "../data/slider/id_00"
normal = sorted(glob.glob(os.path.join(base, "normal", "*.wav")))
abnormal = sorted(glob.glob(os.path.join(base, "abnormal", "*.wav")))

def rms_db(path, channel=0):
    y, sr = sf.read(path)            # y: (n_samples, 8)
    x = y[:, channel]
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)

rng = np.random.default_rng(0)       # graine fixe
sn = rng.choice(normal, 50, replace=False)
sa = rng.choice(abnormal, 50, replace=False)

rn = np.array([rms_db(f) for f in sn])
ra = np.array([rms_db(f) for f in sa])

print(f"Niveau RMS normal   : {rn.mean():.1f} dB (+/- {rn.std():.1f})")
print(f"Niveau RMS anormal  : {ra.mean():.1f} dB (+/- {ra.std():.1f})")