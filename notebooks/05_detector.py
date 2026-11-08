import os, sys
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

# --- Paramètres ---
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "A"   # "A" ou "B"
K = 5                  # trames empilées
BOTTLENECK = 8
BATCH = 512
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
TAG = VARIANT if SEED == 0 else f"{VARIANT}_s{SEED}"
torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Variante :", VARIANT, "| device :", dev)

d = "../outputs/features/"
fname = {"A": "X_A_logmel.npy", "B": "X_B_logmel_nolevel.npy"}[VARIANT]
X = np.load(d + fname)                 # (N, 64, T)
y = np.load(d + "y.npy")
sp = np.load(d + "split.npz")
tr, va, te = sp["train"], sp["val"], sp["test"]

# --- Standardisation par bande, statistiques du TRAIN seul ---
mu = X[tr].mean(axis=(0, 2))
sd = X[tr].std(axis=(0, 2)) + 1e-8
Xs = ((X - mu[None, :, None]) / sd[None, :, None]).astype(np.float32)

# --- Empilement de K trames : (n, 64, T) -> (n, T-K+1, K*64) ---
def frames(Xf):
    n, F, T = Xf.shape
    out = np.stack([Xf[:, :, k:T - K + 1 + k] for k in range(K)], axis=1)  # (n,K,64,T')
    return out.transpose(0, 3, 1, 2).reshape(n, T - K + 1, K * F)

def flat(idx):
    return torch.from_numpy(frames(Xs[idx]).reshape(-1, K * 64))

Xtr, Xva = flat(tr).to(dev), flat(va).to(dev)
print("Trames train :", tuple(Xtr.shape), "| val :", tuple(Xva.shape))

# --- Modèle ---
def block(i, o):
    return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        D = K * 64
        self.enc = nn.Sequential(*block(D, 64), *block(64, 64), nn.Linear(64, BOTTLENECK))
        self.dec = nn.Sequential(*block(BOTTLENECK, 64), *block(64, 64), nn.Linear(64, D))
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z

model = AE().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# --- Entraînement avec arrêt précoce ---
best, best_state, wait = np.inf, None, 0
for ep in range(1, MAX_EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(Xtr), device=dev)
    tot = 0.0
    for i in range(0, len(Xtr), BATCH):
        b = Xtr[perm[i:i + BATCH]]
        if len(b) < 2:
            continue
        opt.zero_grad()
        rec, _ = model(b)
        loss = loss_fn(rec, b)
        loss.backward(); opt.step()
        tot += loss.item() * len(b)
    model.eval()
    with torch.no_grad():
        vl = loss_fn(model(Xva)[0], Xva).item()
    print(f"Epoch {ep:3d} | train {tot/len(Xtr):.4f} | val {vl:.4f}")
    if vl < best - 1e-5:
        best, wait = vl, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Arrêt précoce."); break

model.load_state_dict(best_state); model.eval()

# --- Scores, erreurs par bande et embeddings pour TOUS les fichiers ---
N = len(X)
scores = np.zeros(N); band_err = np.zeros((N, 64)); emb = np.zeros((N, BOTTLENECK))
with torch.no_grad():
    for i in range(N):
        f = torch.from_numpy(frames(Xs[i:i + 1])[0]).to(dev)   # (T', 320)
        rec, z = model(f)
        e = ((rec - f) ** 2).cpu().numpy()                      # (T', 320)
        scores[i] = e.mean()
        band_err[i] = e.reshape(-1, K, 64).mean(axis=(0, 1))
        emb[i] = z.cpu().numpy().mean(axis=0)

# --- Évaluation sur le test ---
yt, st = y[te], scores[te]
auc = roc_auc_score(yt, st)
pauc = roc_auc_score(yt, st, max_fpr=0.1)
print("\n=== Résultats (variante %s) ===" % VARIANT)
print(f"AUC  : {auc:.4f}")
print(f"pAUC : {pauc:.4f}  (FPR <= 0.1, version standardisée sklearn)")
print(f"Score moyen val (normal)      : {scores[va].mean():.4f}")
print(f"Score moyen test normal       : {scores[te][yt == 0].mean():.4f}")
print(f"Score moyen test anormal      : {scores[te][yt == 1].mean():.4f}")

# --- Sauvegardes ---
np.save(d + f"scores_{TAG}.npy", scores)
np.save(d + f"bandErr_{TAG}.npy", band_err)
np.save(d + f"emb_{TAG}.npy", emb)
np.savez(d + f"stdstats_{TAG}.npz", mu=mu, sd=sd)
torch.save(best_state, d + f"ae_{TAG}.pt")
print("Fichiers sauvegardés dans", d)