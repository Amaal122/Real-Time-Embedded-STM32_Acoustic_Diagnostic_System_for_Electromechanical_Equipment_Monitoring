# config.py : source unique des chemins et des choix du projet
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WINDOWS_DIR = ROOT / "windows"          # windows/<machine>/features.f32, meta.csv, ...
RESULTS_DIR = ROOT / "results"

MACHINES = ["slider"]                   # à étendre : pump, fan, valve
SNR_DB = -6
SAMPLE_RATE = 16000

# Features dépendantes du niveau (gain micro / carte) : exclues du modèle final
LEVEL = ["t_std", "t_max", "t_min", "t_rms",
         "t_energy", "t_mean_abs", "t_peak2peak"]

def machine_dir(machine):
    return WINDOWS_DIR / machine

def feature_columns(names, variant="nolevel"):
    """variant = 'all' (65) ou 'nolevel' (58). Par défaut : nolevel."""
    return [i for i, n in enumerate(names)
            if variant == "all" or n not in LEVEL]