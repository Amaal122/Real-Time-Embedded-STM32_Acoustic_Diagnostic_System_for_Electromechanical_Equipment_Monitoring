# Résultats — diagnostic audio automobile

## Ce qui a été exécuté

1. **Prétraitement audio** : WAV mono lu directement, converti en mono par moyenne si multicanal, rééchantillonné à 16 kHz, filtre passe-bande Butterworth ordre 4 (300–7 000 Hz), fenêtres de 1 seconde avec 50 % de recouvrement.
2. **Extraction** : mêmes 65 caractéristiques temporelles, spectrales et Wavelet Packet `db4`, niveau 4, que pour le projet MIMII.
3. **Détection supervisée** : XGBoost normal/anormal, avec sélecteurs ANOVA et JSD évalués séparément. Chaque sélecteur est ajusté sur les fichiers d’entraînement seulement et garde les 60 meilleures caractéristiques.
4. **Diagnostic du type de panne** : XGBoost multiclasse entraîné sur les fichiers anormaux pour identifier l’une des 10 catégories de panne.
5. **Détection non supervisée** : autoencodeur de reconstruction MLP entraîné seulement avec les enregistrements normaux d’entraînement. Le seuil de décision est calibré sur les enregistrements normaux de validation.
6. **Cascade** : d’abord la décision normal/anormal; si le fichier est déclaré anormal, le second modèle prédit le type de panne.

Les fenêtres d’un même WAV ont toujours été gardées dans le même split. Séparation stratifiée au niveau WAV : 886 entraînement, 222 validation et 278 test. Le test n’a pas servi à choisir les seuils.

## Résultats sur les WAV de test

### Détection normal / anormal

| Modèle | ROC AUC | Balanced accuracy | Précision anormal | Rappel anormal | F1 anormal | Faux positifs |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost + JSD | 0,9799 | 0,8979 | 0,9821 | 0,8333 | 0,9016 | 3 / 80 normaux |
| XGBoost + ANOVA | 0,9798 | 0,8980 | 0,9714 | 0,8586 | 0,9115 | 5 / 80 normaux |
| Autoencodeur normal-only | 0,8897 | 0,7176 | 0,9439 | 0,5101 | 0,6623 | 6 / 80 normaux |

JSD et ANOVA sont presque à égalité pour la détection binaire. L’autoencodeur obtient une AUC de 0,8897 et détecte 51,0 % des WAV anormaux au seuil retenu. Il est inférieur aux deux XGBoost sur ce split.

### Type de panne parmi les fichiers anormaux

| Modèle | WAV de test anormaux | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| XGBoost + JSD | 198 | 0,5960 | 0,6022 | 0,6147 |
| XGBoost + ANOVA | 198 | 0,6010 | 0,6066 | 0,6164 |

Plusieurs classifieurs ont ensuite été comparés sur validation uniquement pour remplacer XGBoost dans cette étape. Random Forest (65 features), sélectionné par macro-F1 de validation, a obtenu sur les mêmes 198 WAV anormaux de test : accuracy **0,6414**, balanced accuracy **0,6437**, précision macro **0,6793** et macro F1 **0,6507**. XGBoost + ANOVA sur le même test avait accuracy **0,6010**, balanced accuracy **0,6066** et macro F1 **0,6164**. Random Forest est donc un meilleur candidat pour la classification du type de panne sur ce split, sans garantir le même gain sur d’autres enregistrements.

Le meilleur SVM RBF a obtenu macro-F1 **0,6037** sur validation, très proche de Random Forest (**0,6061**). Les scores de validation ont servi à sélectionner le candidat; le modèle Random Forest final a ensuite été réentraîné sur train + validation avant l’évaluation test.

Un petit CNN a ensuite été essayé sur des spectrogrammes log-Mel (64 bandes), avec l’époque choisie sur validation. Sur le test : accuracy **0,4747**, balanced accuracy **0,5397**, précision macro **0,5132**, macro F1 **0,5041**. Il fait moins bien que Random Forest (**macro F1 0,6507**) et XGBoost + ANOVA (**0,6164**) sur ce split. Il n’est donc pas retenu pour l’instant. Le split a déjà servi à comparer d’autres modèles du projet; ces chiffres restent exploratoires.

La classification de panne est modeste, autour de 60 % en macro F1. ANOVA devance JSD de peu sur ce split.

### Cascade complète (normal ou type de panne)

| Modèle | Classes de sortie | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| XGBoost ANOVA en cascade | 11 (normal + 10 pannes) | 0,6223 | 0,5367 | 0,5595 |

Les pannes ratées par la première étape sont alors annoncées comme normales; les erreurs de diagnostic de la seconde étape s’ajoutent aussi.

### Cascade XGBoost + Random Forest et validation croisée

La chaîne combinée utilise XGBoost ANOVA top-60 pour le seuil normal/anormal, puis Random Forest avec 65 caractéristiques pour le type de panne. Sur le split existant, elle obtient accuracy **0,6583**, balanced accuracy **0,5780** et macro F1 **0,6027**. La cascade XGBoost + XGBoost avait macro F1 **0,5595** sur ce même split.

Pour vérifier la stabilité, une validation croisée stratifiée à 5 folds a gardé chaque WAV dans un seul fold de test. Le seuil du détecteur est calibré dans chaque fold avec un sous-ensemble normal de calibration : macro F1 moyen **0,5643 ± 0,0307**, balanced accuracy **0,5557 ± 0,0345**, accuracy **0,6429 ± 0,0235**. Le score pooled OOF macro F1 est **0,5704**. L’écart entre le split fixe et la CV confirme qu’il faut traiter le score du split fixe comme exploratoire.

## Seuils et autoencodeur

Pour comparer les détecteurs au même point de fonctionnement, le seuil de chacun est le 95e percentile des scores des WAV normaux de validation (cible : environ 5 % de fausses alarmes sur les données de calibration). Le taux réel peut varier sur le test.

TensorFlow et PyTorch n’étant pas installés, l’autoencodeur est une approximation compacte avec `sklearn.MLPRegressor` et couche centrale de 16 unités. Il est appris à reconstruire les 65 features normales standardisées. Ce n’est pas une reproduction d’un baseline automobile officiel.

## Limites à garder en tête

- Un seul split stratifié au niveau WAV; les classes les plus petites ont peu de fichiers de test.
- Les trois états (freinage, ralenti, démarrage) sont mélangés. Le modèle peut exploiter des différences d’état de fonctionnement autant que des signatures de panne.
- Le dataset ne fournit pas de groupes par véhicule, séance ou microphone. Le test mesure donc la généralisation à des fichiers tenus à l’écart, pas nécessairement à un nouveau véhicule ou à une nouvelle acquisition.
- L’autoencodeur est un premier repère non supervisé. Un détecteur séparé par état de fonctionnement ou une calibration plus large pourrait changer ses résultats.

## Fichiers

- Features : `car_diagnostics_features/features_65.csv`
- Inventaire des WAV et classes : `car_diagnostics_features/recording_inventory.csv`
- Répartition train/validation/test : `car_diagnostics_results/recording_split.csv`
- Synthèse des détecteurs : `car_diagnostics_results/binary_xgboost_summary.csv`, `car_diagnostics_results/autoencoder_metrics.json`
- Diagnostic multiclasse et cascade : `car_diagnostics_results/fault_diagnosis_summary.csv`, `car_diagnostics_results/cascade_summary.csv`
- Comparaison des classifieurs et modèle retenu : `car_diagnostics_results/fault_model_search/validation_model_comparison.csv`, `car_diagnostics_results/fault_model_search/selected_fault_classifier.joblib`
- CNN log-Mel : `car_diagnostics_results/cnn/metrics.json`, `car_diagnostics_results/cnn/fault_type_cnn.pt`
- Cascade XGBoost + Random Forest : `car_diagnostics_results/hybrid_cascade/holdout_metrics.json`, `car_diagnostics_results/hybrid_cascade/grouped_cv_metrics.json`, `car_diagnostics_results/hybrid_cascade/grouped_cv_confusion_matrix.csv`

## Détecteurs séparés par panne

Six Random Forest binaires ont été entraînés : `worn_out_brakes`, `bad_ignition`, `dead_battery`, `low_oil`, `power_steering` et `serpentine_belt`. Les quatre dossiers de pannes combinées sont annotés comme positifs pour chaque composant nommé. Un même WAV combiné peut donc déclencher plusieurs détecteurs.

Chaque modèle utilise les 65 features, `class_weight=balanced` et un seuil de décision initial de **0,5**. Les métriques sont calculées par WAV avec 5-fold stratifié; les fenêtres d’un fichier restent dans le même fold. Les modèles finaux sont ensuite réentraînés sur les 1 386 WAVs.

| Détecteur | WAV positifs | ROC AUC OOF | AP OOF | Précision à 0,5 | Rappel à 0,5 | F1 à 0,5 |
|---|---:|---:|---:|---:|---:|---:|
| Freins usés | 76 | 0,954 | 0,756 | 0,957 | 0,289 | 0,444 |
| Mauvais allumage | 62 | 0,975 | 0,808 | 1,000 | 0,210 | 0,347 |
| Batterie morte | 57 | 0,968 | 0,696 | 0,857 | 0,316 | 0,462 |
| Manque d’huile | 428 | 0,956 | 0,913 | 0,875 | 0,785 | 0,828 |
| Direction assistée | 459 | 0,896 | 0,785 | 0,762 | 0,621 | 0,684 |
| Courroie serpentine | 446 | 0,961 | 0,925 | 0,903 | 0,809 | 0,853 |

Les trois pannes rares ont un classement prometteur (AUC élevé), mais le seuil 0,5 est trop conservateur pour les détecter souvent. Les seuils devront être réglés par panne selon la priorité donnée au rappel ou aux fausses alarmes. L’AP (average precision) et l’AUC évaluent le classement des scores sans se limiter à ce seuil.

- Résumé OOF : `car_diagnostics_results/fault_specialists/specialist_summary.csv`
- Correspondance entre pannes combinées et détecteurs : `car_diagnostics_results/fault_specialists/mapping_and_run_metadata.json`
- Modèles finaux : `car_diagnostics_results/fault_specialists/detector_*.joblib`
- Script : `scripts/train_fault_specialists.py`
- Matrices de confusion et prédictions par WAV : fichiers CSV dans ce dossier.

## Relancer

Depuis la racine du projet :

```powershell
python scripts/extract_car_features.py --input "data/car diagnostics dataset" --output car_diagnostics_features --target-sr 16000 --window-sec 1 --overlap 0.5 --wavelet db4
python scripts/train_car_models.py --input car_diagnostics_features/features_65.csv --output car_diagnostics_results --top-k 60 --test-size 0.2 --validation-size 0.2 --target-fpr 0.05 --seed 42
```

Pour évaluer la chaîne XGBoost détection + Random Forest diagnostic et sa stabilité en CV :

```powershell
python scripts/evaluate_car_hybrid_cascade.py --input car_diagnostics_features/features_65.csv --split car_diagnostics_results/recording_split.csv --model-root car_diagnostics_results --output car_diagnostics_results/hybrid_cascade --folds 5 --top-k 60 --seed 42
```

Pour réentraîner les six détecteurs spécialisés :

```powershell
python scripts/train_fault_specialists.py --input car_diagnostics_features/features_65.csv --output car_diagnostics_results/fault_specialists --folds 5 --threshold 0.5 --seed 42
```


## Calibration des seuils par panne

Les six classifieurs Random Forest produisent un score entre 0 et 1 pour chaque panne. Le seuil 0,5 n'est pas toujours adapte, surtout pour les pannes rares. Une validation croisee imbriquee a 5 folds a ete executee au niveau WAV : dans chaque fold, un sous-ensemble de calibration choisit le seuil, puis les metriques sont calculees sur des WAV jamais utilises pour ce choix.

Deux regles sont comparees : **max-F1** (compromis precision/rappel sur les donnees de calibration) et **rappel cible 80 %** (seuil le plus eleve qui atteint ce rappel pendant la calibration). Les resultats ci-dessous sont les predictions pooled out-of-fold, avec un seuil choisi separement dans chaque fold. `Precision / rappel / F1` sont calcules par WAV.

| Panne | Seuil median max-F1* | Precision / rappel / F1 max-F1 | Seuil median cible 80 %* | Precision / rappel / F1 cible 80 % |
|---|---:|---:|---:|---:|
| Freins uses | 0,175 | 0,606 / 0,750 / 0,671 | 0,120 | 0,457 / 0,776 / 0,576 |
| Mauvais allumage | 0,200 | 0,586 / 0,661 / 0,621 | 0,133 | 0,505 / 0,758 / 0,606 |
| Batterie morte | 0,305 | 0,674 / 0,544 / 0,602 | 0,084 | 0,171 / 0,860 / 0,285 |
| Manque d'huile | 0,489 | 0,833 / 0,801 / 0,817 | 0,530 | 0,873 / 0,773 / 0,820 |
| Direction assistee | 0,364 | 0,678 / 0,769 / 0,720 | 0,388 | 0,694 / 0,706 / 0,700 |
| Courroie serpentine | 0,431 | 0,865 / 0,850 / 0,857 | 0,538 | 0,895 / 0,787 / 0,838 |

\* Les seuils medians sont des valeurs de depart pour les modeles entraines sur tous les WAV. Les metriques de la table viennent des seuils choisis separement dans chaque fold; elles ne garantissent donc pas exactement le meme resultat avec le seuil median final.

La strategie a rappel cible ne garantit pas 80 % sur les WAV de test : le rappel varie entre folds et peut baisser sur de nouveaux enregistrements. Pour la batterie morte, viser un rappel eleve a provoque beaucoup de fausses alarmes (precision 0,171). Le seuil max-F1 est plus equilibre dans cette experience. Ces resultats comparent des compromis et ne certifient pas un diagnostic automobile.

- Seuils medians proposes, seuils de chaque fold et regles : `car_diagnostics_results/fault_specialists/threshold_calibration/recommended_thresholds.json`
- Metriques test par fold : `car_diagnostics_results/fault_specialists/threshold_calibration/threshold_fold_metrics.csv`
- Predictions pooled OOF et metriques : `car_diagnostics_results/fault_specialists/threshold_calibration/threshold_oof_predictions.csv`, `threshold_pooled_oof_metrics.csv`
- Script : `scripts/calibrate_fault_thresholds.py`

Pour relancer la calibration :

```powershell
python scripts/calibrate_fault_thresholds.py --input car_diagnostics_features/features_65.csv --output car_diagnostics_results/fault_specialists/threshold_calibration --folds 5 --target-recall 0.80 --seed 42
```


## Predire les pannes d'un WAV

Le script applique le meme pretraitement que l'extraction des features, puis les six modeles specialises. Par defaut, il utilise les seuils medians max-F1 calibres; un WAV peut recevoir plusieurs pannes detectees. Le score d'un enregistrement est la moyenne des probabilites de ses fenetres. Le CSV de sortie contient aussi les scores et decisions par fenetre.

```powershell
python scripts/predict_wav_fault_specialists.py --input "chemin\vers\enregistrement.wav"
```

Pour choisir le profil qui vise un rappel de calibration de 80 % (avec potentiellement plus de fausses alarmes) :

```powershell
python scripts/predict_wav_fault_specialists.py --input "chemin\vers\enregistrement.wav" --threshold-strategy target_recall
```

Le CSV est cree a cote du WAV sous le nom `<nom>_fault_predictions.csv`. Le resultat signifie qu'un seuil du detecteur correspondant est depasse; il ne constitue pas a lui seul un diagnostic mecanique confirme.


## Tester avec le micro du PC

Le script enregistre d'abord quelques secondes depuis le microphone du PC qui execute le modele, puis lance la prediction sur le WAV capture. Pendant le message "Recording", faire jouer le son moteur sur l'autre PC et placer son haut-parleur pres du microphone. La prediction arrive apres la fin de l'enregistrement; ce n'est pas une prediction en continu.

Installer `sounddevice` une seule fois avec le meme Python que celui utilise pour lancer les scripts :

```powershell
& 'C:\Users\Eslem Ben Ameur\AppData\Local\Programs\Python\Python313\python.exe' -m pip install sounddevice
```

Depuis la racine du projet, lancer un enregistrement de 30 secondes (duree par defaut) :

```powershell
& 'C:\Users\Eslem Ben Ameur\AppData\Local\Programs\Python\Python313\python.exe' scripts\predict_microphone_fault_specialists.py --duration-sec 30
```

Pour afficher les micros disponibles en cas de probleme de peripherique :

```powershell
& 'C:\Users\Eslem Ben Ameur\AppData\Local\Programs\Python\Python313\python.exe' scripts\predict_microphone_fault_specialists.py --list-devices
```

Un son emis par un haut-parleur peut differer d'une mesure faite directement sur un moteur. La prediction reste donc un essai du pipeline, pas une validation sur des donnees reelles de terrain.


## Profil de seuils du tableau de reference

Le profil facultatif `lcd_table` applique directement les seuils fournis : `worn_out_brakes=0.90`, `bad_ignition=0.75`, `low_oil=0.45`, `power_steering=0.40` et `serpentine_belt=0.35`. Le tableau ne donne pas de seuil pour `dead_battery`, donc ce detecteur garde son seuil max-F1 median calibre (0.305). Il donne `normal=0.75`, mais le projet n'a pas de detecteur normal separe : aucun specialist ne depasse son seuil signifie seulement "aucune panne detectee". Deux pannes ou plus detectees sont affichees ensemble.

Ces valeurs viennent d'un tableau externe; elles ne sont pas calibrees pour les scores Random Forest de ce projet. Appliquees aux predictions OOF des specialistes, elles donnent un rappel de seulement 0,066 pour `worn_out_brakes` et 0,016 pour `bad_ignition` : elles sont donc inadaptees a ces modeles. Le profil `max_f1` calibre sur le dataset reste le choix par defaut. Pour essayer les valeurs du tableau avec le microphone :

```powershell
python scripts/predict_microphone_fault_specialists.py --duration-sec 30 --threshold-strategy lcd_table
```


