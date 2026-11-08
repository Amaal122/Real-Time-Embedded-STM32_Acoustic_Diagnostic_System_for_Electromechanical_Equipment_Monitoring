import glob, os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# --- Paramètres (à noter dans config.md) ---
SR = 16000
N_FFT = 1024
HOP = 512
N_MELS = 64
CHANNEL = 0

base = "../data/slider/id_00"
out_dir = "../outputs/features"
os.makedirs(out_dir, exist_ok=True)

def logmel(path):
    y, sr = sf.read(path)                 # (n_samples, 8)
    assert sr == SR
    x = y[:, CHANNEL].astype(np.float32)
    S = librosa.feature.melspectrogram(
        y=x, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    return librosa.power_to_db(S, ref=1.0, amin=1e-10)   # (64, ~313)

classes = {"normal": 0, "abnormal": 1}
X_list, y_list, names = [], [], []

for cls, label in classes.items():
    files = sorted(glob.glob(os.path.join(base, cls, "*.wav")))
    for f in tqdm(files, desc=cls):
        X_list.append(logmel(f))
        y_list.append(label)
        names.append(f"{cls}/{os.path.basename(f)}")

X_A = np.stack(X_list).astype(np.float32)     # (N, 64, T)  niveau conservé
y = np.array(y_list)
names = np.array(names)

# Variante B : on retire la moyenne globale de chaque fichier
X_B = X_A - X_A.mean(axis=(1, 2), keepdims=True)

np.save(os.path.join(out_dir, "X_A_logmel.npy"), X_A)
np.save(os.path.join(out_dir, "X_B_logmel_nolevel.npy"), X_B)
np.save(os.path.join(out_dir, "y.npy"), y)
np.save(os.path.join(out_dir, "names.npy"), names)

print("\nX_A :", X_A.shape, X_A.dtype)
print("X_B :", X_B.shape)
print("y   :", y.shape, "| normal =", (y == 0).sum(), "| abnormal =", (y == 1).sum())
print("Moyenne globale A (normal)  :", X_A[y == 0].mean().round(1), "dB")
print("Moyenne globale A (anormal) :", X_A[y == 1].mean().round(1), "dB")
print("Moyenne globale B (contrôle):", X_B.mean().round(4), "dB (doit être ~0)")