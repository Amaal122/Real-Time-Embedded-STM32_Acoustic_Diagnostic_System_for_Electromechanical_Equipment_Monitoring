import numpy as np
import matplotlib.pyplot as plt

d = "../outputs/features/"
X = np.load(d + "X_A_logmel.npy"); y = np.load(d + "y.npy")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, lab, title in zip(ax, [0, 1], ["Normal (moyenne)", "Anormal (moyenne)"]):
    im = a.imshow(X[y == lab].mean(axis=0), origin="lower", aspect="auto")
    a.set_title(title); a.set_xlabel("Trame"); a.set_ylabel("Bande mel")
    plt.colorbar(im, ax=a, label="dB")
plt.tight_layout()
plt.savefig("../outputs/mean_logmel_A.png", dpi=150)

# profil spectral moyen
plt.figure(figsize=(7, 4))
plt.plot(X[y == 0].mean(axis=(0, 2)), label="normal")
plt.plot(X[y == 1].mean(axis=(0, 2)), label="anormal")
plt.xlabel("Bande mel"); plt.ylabel("dB"); plt.legend(); plt.grid(alpha=.3)
plt.savefig("../outputs/mean_spectrum_A.png", dpi=150)
plt.show()