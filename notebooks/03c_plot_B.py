import numpy as np
import matplotlib.pyplot as plt

d = "../outputs/features/"
XB = np.load(d + "X_B_logmel_nolevel.npy"); y = np.load(d + "y.npy")

plt.figure(figsize=(7, 4))
plt.plot(XB[y == 0].mean(axis=(0, 2)), label="normal (B)")
plt.plot(XB[y == 1].mean(axis=(0, 2)), label="anormal (B)")
plt.xlabel("Bande mel"); plt.ylabel("dB relatif"); plt.legend(); plt.grid(alpha=.3)
plt.savefig("../outputs/mean_spectrum_B.png", dpi=150)
plt.show()