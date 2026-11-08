import numpy as np
from extract_features import window_features
t = np.arange(16000) / 16000
x = (0.5 * np.sin(2 * np.pi * 1250 * t)).astype(np.float32)
f = window_features(x, ["wpt"]).reshape(16, 3)
print("bande dominante :", f[:, 0].argmax(), "(attendu : 2)")