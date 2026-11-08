import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
d = "../outputs/features/"

y = np.load(d + "y.npy")
names = np.load(d + "names.npy")

idx_normal = np.where(y == 0)[0]
idx_abn = np.where(y == 1)[0]

# 70 % train / 30 % reste
idx_train, idx_rest = train_test_split(idx_normal, test_size=0.30,
                                       random_state=SEED, shuffle=True)
# reste -> 1/3 validation (10 % du total) / 2/3 test normal (20 % du total)
idx_val, idx_test_norm = train_test_split(idx_rest, test_size=2/3,
                                          random_state=SEED, shuffle=True)

idx_test = np.concatenate([idx_test_norm, idx_abn])

np.savez(d + "split.npz",
         train=np.sort(idx_train),
         val=np.sort(idx_val),
         test_norm=np.sort(idx_test_norm),
         test_abn=np.sort(idx_abn),
         test=np.sort(idx_test))

# --- Contrôles ---
print("Train       :", len(idx_train))
print("Validation  :", len(idx_val))
print("Test normal :", len(idx_test_norm))
print("Test anormal:", len(idx_abn))
print("Total       :", len(idx_train) + len(idx_val) + len(idx_test))

# Aucun recouvrement
s_tr, s_va, s_te = set(idx_train), set(idx_val), set(idx_test)
print("\nRecouvrement train/val :", len(s_tr & s_va))
print("Recouvrement train/test:", len(s_tr & s_te))
print("Recouvrement val/test  :", len(s_va & s_te))

# Aucune anomalie dans train/val
print("Anomalies dans train   :", int(y[idx_train].sum()))
print("Anomalies dans val     :", int(y[idx_val].sum()))