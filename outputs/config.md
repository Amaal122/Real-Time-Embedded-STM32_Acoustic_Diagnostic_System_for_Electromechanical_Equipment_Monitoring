# Configuration de l'étude

Machine type   : slider
Machine ID     : id_00
SNR level      : 6 dB (à confirmer à l'étape 2)
Normal files   : 1068
Abnormal files : 356
Anomaly ratio  : 0.25
Format         : 16 kHz, 8 canaux, 10 s (vérifié sur 1424 fichiers, 0 anomalie de format)
Labels         : binaire uniquement (normal / abnormal) ; type de défaut NON étiqueté
Justification  : plus grand nombre d'anomalies parmi les machines faciles à détecter
                 (356) ; un seul ID évite la variabilité inter-machines ;
                 trois scénarios de défaut décrits dans l'article
                 (rail endommagé, courroie desserrée, manque de graisse)
Source         : MIMII dataset (Purohit et al., 2019), Zenodo
Date           : 09/10/2026
SNR level        : 6 dB (confirmé par la provenance : 6_dB_slider.zip ;
                   le SNR n'est pas mesurable depuis le mélange final)
Niveau RMS normal : -48.7 dBFS (+/- 0.8)  [canal 0, 50 fichiers, graine 0]
Niveau RMS anormal: -40.9 dBFS (+/- 1.3)  [canal 0, 50 fichiers, graine 0]
Prétraitement : canal 0, 16 kHz, fichier entier (10 s), aucun filtrage,
                aucune normalisation du signal
Features      : log-mel 64 bandes, n_fft=1024, hop=512, power=2, ref=1.0
Variante A    : log-mel brut (niveau conservé)
Variante B    : A moins la moyenne globale par fichier (niveau retiré)
Fichiers      : outputs/features/{X_A_logmel, X_B_logmel_nolevel, y, names}.npy
Résultats     : X_A (1424, 64, 313) ; moyenne A normal -41.5 dB, anormal -33.4 dB
                (écart +8.1 dB) ; moyenne B = 0.0 dB
Observations  : signal stationnaire (spectrogrammes en bandes horizontales) ;
                écart normal/anormal croissant avec la fréquence (~5 dB vers
                bande 20, ~15 dB vers bande 60) : pas un simple décalage de gain
                Split         : graine 42, fichier outputs/features/split.npz
Train         : 747 normaux (70 %)
Validation    : 107 normaux (10 %)
Test          : 214 normaux + 356 anormaux
Contrôles     : 0 recouvrement, 0 anomalie dans train/val
Limite        : split aléatoire (pas chronologique)
Résultats A   : AUC = 1.0000, pAUC = 1.0000
                score val 0.3001 | test normal 0.3032 | test anormal 1.1919
Résultats B   : AUC = 0.9999, pAUC = 0.9993
                score val 0.3556 | test normal 0.3573 | test anormal 1.2291
Conclusion    : la détection repose sur la forme spectrale, pas uniquement
                sur le niveau global ; cas facile (slider id_00, 6 dB)
Sorties       : scores_{A,B}.npy, bandErr_{A,B}.npy, emb_{A,B}.npy, ae_{A,B}.pt
Stabilité     : 3 graines de modèle (0, 1, 2), split fixe (graine 42)
Résultats A   : AUC = 0.9999 ± 0.0001, pAUC = 0.9998 ± 0.0002
Résultats B   : AUC = 0.9998 ± 0.0002, pAUC = 0.9990 ± 0.0009
Conclusion    : détection stable ; A et B indiscernables ; cas facile (6 dB)
Référence     : graine 0 conservée pour la suite (scores_{A,B}.npy,
                bandErr_{A,B}.npy, emb_{A,B}.npy)
                Clustering    : 356 anomalies réelles, K-means k=2..8, PCA 95 %
Variante A    : k optimal = 3 pour les 3 représentations
                silhouette (k=3) : résidu 0.406 | bandErr 0.359 | emb 0.498
Variante B    : k optimal dispersé (2, 5, 8), silhouettes 0.25-0.47
ARI(A,B), k=3 : résidu 0.921 | bandErr 0.387 | emb 0.536
Conclusion    : structure à ~3 groupes, robuste au retrait du niveau pour le
                résidu spectral ; bandErr/emb plus dépendants du modèle ;
                correspondance avec les 3 défauts NON démontrée
                Référence clustering : résidu spectral, variante A, k=3, K-means (graine 0)
Tailles       : 80 / 96 / 180
Stabilité     : ARI graines = 1.000, sous-échantillons = 0.994 (min 0.982)
Accord        : ARI résidu/emb = 0.937, résidu/bandErr = 0.843
Ordre fichiers: Kruskal p = 1.6e-54, voisins même cluster = 0.941 (attendu 0.379)
                -> clusters en blocs consécutifs (défauts en séries OU sessions)
Niveau global : cluster 0 = -31.8 dB, 1 = -35.1, 2 = -33.2
Normaux/ordre : Spearman niveau rho=-0.015 (p=0.62) ; PC1 rho=+0.139 (faible)
                voisins même cluster = 0.630 (attendu 0.628), Kruskal p=0.93
Conclusion    : pas de dérive des normaux ; anomalies en blocs (0.941 vs 0.379)
                -> effet de session global peu probable ; structure liée à la
                condition de défaut OU à des séries d'enregistrement ;
                correspondance avec les 3 défauts NON démontrée
                Écoute        : 15 anomalies + 5 normaux, un auditeur, AVEUGLE (fiche remplie
                avant ouverture de key.csv), gain commun, ordre aléatoire
Normaux       : 5/5 reconnus ; 0/15 anomalies prises pour des normaux
Accord        : 13/15 anomalies bien placées ; ARI = 0.570 (indicatif, n=15)
Réponses doubles : L13 (A,C) et L16 (B,A), première lettre retenue
Descripteurs  : A (cl.0) irrégulier, grave ; B (cl.2) claquements ;
                C (cl.1) sifflement aigu
Limite        : un auditeur, n=15 ; sons différents, pas défauts nommés