# Sonic AI : détection d'anomalie acoustique temps réel (STM32F469I-DISCO)

Détection normal / anormal, une décision par fenêtre de 1 s, entièrement sur la carte (sans cloud).
Dataset MIMII à −6 dB, métrique principale : F1 macro. Machine traitée pour l'instant : **slider** (ids 00, 02, 04, 06).

La partie 1 (prétraitement, features, diagnostic, méthode non supervisée) est la base du projet.
La partie 2 (auto-encodeur, K-means sur slider id_00) n'est qu'un test exploratoire.

## Organisation actuelle

```
<racine>/
├── venv/
├── dataset mimii/-6_dB_slider/        données brutes (ne jamais modifier)
├── windows/slider/                    windows.f32, meta.csv, features.f32, features_names.txt,
│                                      logmel.f32, logmel_names.txt
└── traitement/                        scripts + config.py
```

Toutes les commandes se lancent **depuis la racine** avec `.\venv\Scripts\python.exe`.

## Pipeline (ordre d'exécution)

| # | Script | Rôle | Sorties |
|---|--------|------|---------|
| 1 | `prepare_windows.py` | moyenne de 2 micros voisins (EGC), Butterworth 300–7000 Hz, fenêtres de 1 s (10 par clip) | `windows.f32`, `meta.csv` |
| 2a | `extract_features.py` | 65 features : 12 temporelles, 5 spectrales, 48 WPT (16 bandes × rms_rel, kurt, entropy) | `features.f32`, `features_names.txt` |
| 2b | `extract_logmel.py` | 128 valeurs : 64 moyennes + 64 écarts-types log-mel, niveau absolu retiré par défaut | `logmel.f32`, `logmel_names.txt` |
| 3 | `select_features.py` | sélection ANOVA vs JSD, leave-one-id-out (XGBoost) | tableaux de F1 |
| 4 | `diag_machine.py` | modèle commun vs modèle par machine (supervisé, XGBoost) | `diag_machine_<variante>.csv` |
| 5 | `diag_unsup.py` | détection non supervisée (Isolation Forest, Mahalanobis), entraînée sur le normal | `diag_unsup_features_<variante>.csv` |

Scripts de contrôle : `check_features.py`, `check_par_machine.py`, `test_bandes.py`. À clarifier : `par_machine.py`.

### Commandes (slider)

```powershell
.\venv\Scripts\python.exe .\traitement\prepare_windows.py --input ".\dataset mimii\-6_dB_slider" --output ".\windows\slider"
.\venv\Scripts\python.exe .\traitement\extract_features.py --dir ".\windows\slider" --groups temporal,spectral,wpt --out features
.\venv\Scripts\python.exe .\traitement\extract_logmel.py --dir ".\windows\slider" --out logmel
```

Variantes de `extract_logmel.py` : `--keep-level` (garde le niveau absolu, `--out logmel_level`), `--full` (log-mel complet N × 30 × 64).

Diagnostic (à compléter avec les arguments exacts) :

```powershell
.\venv\Scripts\python.exe .\traitement\diag_machine.py --variant nolevel     # ou --variant all
.\venv\Scripts\python.exe .\traitement\diag_unsup.py                         # nolevel par défaut ; --keep-level pour 65 features
```

## Décisions

- Détection **non supervisée**, calibrée sur site avec du normal seulement ; **une seule méthode** pour toutes les machines.
- Pas de modèle supervisé par machine : il sert seulement de **plafond** de référence.
- Variante par défaut : **`nolevel`** (58 features). Les 7 features dépendant du niveau sont exclues (`LEVEL` dans `config.py`) : `t_std`, `t_max`, `t_min`, `t_rms`, `t_energy`, `t_mean_abs`, `t_peak2peak`.
- Découpage train/test toujours par **id de machine**, jamais par fenêtre.

## Résultats (slider, −6 dB)

| Expérience | Résultat |
|---|---|
| Sélection ANOVA vs JSD | F1 macro 0,614 vs 0,611, écart-type ≈ 0,2 entre machines : pas de gagnant (variante à confirmer) |
| Modèle commun, `all` / `nolevel` (AUC cross) | id_00 0,87 / 0,83 ; id_02 0,76 / 0,77 ; id_04 0,75 / 0,76 ; id_06 0,62 / 0,64 |
| id_06, modèle commun | F1 0,34 / 0,33, 76 % de fenêtres prédites anormales (variante `all`) |
| Modèle par machine (within) | AUC 0,91 à 1,00, identique avec ou sans features de niveau |
| Non supervisé sur features, id_00 (cross) | Isolation Forest 0,85 (`all`) / 0,82 (`nolevel`) ; Mahalanobis 0,87 / 0,85 ; faible sur id_02, id_04, id_06 |
| Non supervisé log-mel + Mahalanobis, calibré par machine | AUC 0,71 à 0,92 (commande exacte à ajouter ci-dessous) |

Conclusion de robustesse : retirer les features de niveau ne change pas le diagnostic, sauf −0,02 à −0,04 d'AUC sur id_00.

## État

| Bloc | Statut |
|---|---|
| Prétraitement, features, log-mel | Fait |
| Diagnostic supervisé all vs nolevel | Fait |
| Non supervisé log-mel (méthode retenue) | Fait en partie : commande à documenter |
| `select_features.py` en `nolevel` | À relancer |
| Autres machines (pompe, ventilateur, vanne), 0 dB et 6 dB | À faire |
| Protocole de calibration sur site (durée de normal, seuil) | À faire |
| Export ONNX / X-CUBE-AI, acquisition PDM→PCM, firmware | À faire |
| Rapport (17 sections) | À jour avant cette relance, à compléter |
| Présentation (14 diapositives) | En cours |

## À compléter

- Commande exacte qui produit l'AUC log-mel + Mahalanobis.
- Arguments exacts de `diag_machine.py`, `diag_unsup.py` et `select_features.py`.
- Rôle de `par_machine.py`.