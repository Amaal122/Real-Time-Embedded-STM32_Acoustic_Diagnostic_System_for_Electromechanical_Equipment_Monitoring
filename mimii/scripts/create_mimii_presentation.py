#!/usr/bin/env python3
"""Create an editable French PowerPoint recap of the MIMII XGBoost project."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET


OUT = Path("Presentation_MIMII_XGBoost_FR.pptx")
EMU = 914400
SW, SH = 13.333333 * EMU, 7.5 * EMU

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
XML = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("p", P)
ET.register_namespace("a", A)
ET.register_namespace("r", R)
ET.register_namespace("", REL)

NAVY = "0B1320"
NAVY2 = "142235"
TEAL = "17BEBB"
BLUE = "4F91FF"
BG = "F3F6FA"
WHITE = "FFFFFF"
INK = "17263A"
MUTED = "586A80"
PALE = "E5EEF5"
ORANGE = "F2A65A"
RED = "D85C62"
GREEN = "238A72"


def q(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


class Slide:
    def __init__(self, number: int, dark: bool = False):
        self.number = number
        self.dark = dark
        self.bg = NAVY if dark else BG
        self.root = ET.Element(q(P, "sld"), {"showMasterSp": "1"})
        c_sld = ET.SubElement(self.root, q(P, "cSld"))
        bg = ET.SubElement(c_sld, q(P, "bg"))
        bg_pr = ET.SubElement(bg, q(P, "bgPr"))
        solid = ET.SubElement(bg_pr, q(A, "solidFill"))
        ET.SubElement(solid, q(A, "srgbClr"), {"val": self.bg})
        ET.SubElement(bg_pr, q(A, "effectLst"))
        self.tree = ET.SubElement(c_sld, q(P, "spTree"))
        self._group_tree()

    def _group_tree(self) -> None:
        nv = ET.SubElement(self.tree, q(P, "nvGrpSpPr"))
        ET.SubElement(nv, q(P, "cNvPr"), {"id": "1", "name": ""})
        ET.SubElement(nv, q(P, "cNvGrpSpPr"))
        ET.SubElement(nv, q(P, "nvPr"))
        grp = ET.SubElement(self.tree, q(P, "grpSpPr"))
        xfrm = ET.SubElement(grp, q(A, "xfrm"))
        ET.SubElement(xfrm, q(A, "off"), {"x": "0", "y": "0"})
        ET.SubElement(xfrm, q(A, "ext"), {"cx": "0", "cy": "0"})
        ET.SubElement(xfrm, q(A, "chOff"), {"x": "0", "y": "0"})
        ET.SubElement(xfrm, q(A, "chExt"), {"cx": "0", "cy": "0"})

    @staticmethod
    def _emu(v: float) -> str:
        return str(int(v * EMU))

    def rect(self, x: float, y: float, w: float, h: float, fill: str,
             radius: bool = False, line: str | None = None) -> None:
        idx = len(self.tree) + 1
        sp = ET.SubElement(self.tree, q(P, "sp"))
        nv = ET.SubElement(sp, q(P, "nvSpPr"))
        ET.SubElement(nv, q(P, "cNvPr"), {"id": str(idx), "name": f"Shape {idx}"})
        ET.SubElement(nv, q(P, "cNvSpPr"))
        ET.SubElement(nv, q(P, "nvPr"))
        sppr = ET.SubElement(sp, q(P, "spPr"))
        xfrm = ET.SubElement(sppr, q(A, "xfrm"))
        ET.SubElement(xfrm, q(A, "off"), {"x": self._emu(x), "y": self._emu(y)})
        ET.SubElement(xfrm, q(A, "ext"), {"cx": self._emu(w), "cy": self._emu(h)})
        geom = ET.SubElement(sppr, q(A, "prstGeom"), {"prst": "roundRect" if radius else "rect"})
        ET.SubElement(geom, q(A, "avLst"))
        sf = ET.SubElement(sppr, q(A, "solidFill"))
        ET.SubElement(sf, q(A, "srgbClr"), {"val": fill})
        ln = ET.SubElement(sppr, q(A, "ln"), {"w": "9525"})
        if line:
            lsf = ET.SubElement(ln, q(A, "solidFill"))
            ET.SubElement(lsf, q(A, "srgbClr"), {"val": line})
        else:
            ET.SubElement(ln, q(A, "noFill"))

    def text(self, x: float, y: float, w: float, h: float, value: str,
             size: int = 16, color: str | None = None, bold: bool = False,
             align: str = "l", valign: str = "t", font: str = "Aptos",
             margin: float = 0.02, spacing: int = 100) -> None:
        color = color or (WHITE if self.dark else INK)
        idx = len(self.tree) + 1
        sp = ET.SubElement(self.tree, q(P, "sp"))
        nv = ET.SubElement(sp, q(P, "nvSpPr"))
        ET.SubElement(nv, q(P, "cNvPr"), {"id": str(idx), "name": f"Text {idx}"})
        ET.SubElement(nv, q(P, "cNvSpPr"), {"txBox": "1"})
        ET.SubElement(nv, q(P, "nvPr"))
        sppr = ET.SubElement(sp, q(P, "spPr"))
        xfrm = ET.SubElement(sppr, q(A, "xfrm"))
        ET.SubElement(xfrm, q(A, "off"), {"x": self._emu(x), "y": self._emu(y)})
        ET.SubElement(xfrm, q(A, "ext"), {"cx": self._emu(w), "cy": self._emu(h)})
        geom = ET.SubElement(sppr, q(A, "prstGeom"), {"prst": "rect"})
        ET.SubElement(geom, q(A, "avLst"))
        ET.SubElement(sppr, q(A, "noFill"))
        ln = ET.SubElement(sppr, q(A, "ln"))
        ET.SubElement(ln, q(A, "noFill"))
        body = ET.SubElement(sp, q(P, "txBody"))
        ET.SubElement(body, q(A, "bodyPr"), {
            "wrap": "square", "anchor": "ctr" if valign == "ctr" else "t",
            "lIns": self._emu(margin), "rIns": self._emu(margin),
            "tIns": self._emu(margin), "bIns": self._emu(margin),
        })
        ET.SubElement(body, q(A, "lstStyle"))
        for line_text in value.split("\n"):
            para = ET.SubElement(body, q(A, "p"))
            ppr = ET.SubElement(para, q(A, "pPr"), {"algn": {"l": "l", "c": "ctr", "r": "r"}.get(align, "l")})
            run = ET.SubElement(para, q(A, "r"))
            rpr = ET.SubElement(run, q(A, "rPr"), {
                "lang": "fr-FR", "sz": str(size * 100), "b": "1" if bold else "0",
                "dirty": "0",
            })
            solid = ET.SubElement(rpr, q(A, "solidFill"))
            ET.SubElement(solid, q(A, "srgbClr"), {"val": color})
            latin = ET.SubElement(rpr, q(A, "latin"), {"typeface": font})
            ET.SubElement(run, q(A, "t"), {q(XML, "space"): "preserve"}).text = line_text
            end = ET.SubElement(para, q(A, "endParaRPr"), {"lang": "fr-FR", "sz": str(size * 100)})
            end_solid = ET.SubElement(end, q(A, "solidFill"))
            ET.SubElement(end_solid, q(A, "srgbClr"), {"val": color})

    def header(self, section: str, title: str, subtitle: str | None = None) -> None:
        self.rect(0, 0, 13.333, 0.14, TEAL)
        self.text(0.68, 0.34, 6.0, 0.25, section.upper(), 9, TEAL, True)
        self.text(0.68, 0.67, 12.0, 0.52, title, 25, INK, True)
        if subtitle:
            self.text(0.7, 1.21, 11.9, 0.35, subtitle, 11, MUTED)
        self.rect(0.68, 7.08, 11.95, 0.012, "D8E1EA")
        self.text(0.7, 7.13, 10.7, 0.2, "MIMII • 0 dB • Slide Rail", 8, MUTED)
        self.text(12.0, 7.13, 0.6, 0.2, f"{self.number:02d}", 8, MUTED, True, "r")


def add_card(slide: Slide, x: float, y: float, w: float, h: float, title: str,
             body: str, accent: str = TEAL, body_size: int = 13) -> None:
    slide.rect(x, y, w, h, WHITE, radius=True, line="E2EAF1")
    slide.rect(x, y, 0.08, h, accent, radius=True)
    slide.text(x + 0.22, y + 0.18, w - 0.4, 0.35, title, 16, INK, True)
    slide.text(x + 0.22, y + 0.62, w - 0.42, h - 0.76, body, body_size, MUTED)


def bullet(slide: Slide, x: float, y: float, w: float, text: str, size: int = 14,
           color: str = INK, accent: str = TEAL) -> None:
    slide.rect(x, y + 0.14, 0.09, 0.09, accent, radius=True)
    slide.text(x + 0.22, y, w - 0.22, 0.5, text, size, color)


def build_slides() -> list[Slide]:
    slides: list[Slide] = []

    # 1 — cover
    s = Slide(1, dark=True)
    s.rect(0.78, 0.78, 0.13, 5.9, TEAL, radius=True)
    s.text(1.22, 0.82, 7.4, 0.35, "PROJET • MIMII 0 dB", 12, TEAL, True)
    s.text(1.2, 1.55, 10.9, 2.0, "Détection d’anomalies\nsur les machines Slide Rail", 34, WHITE, True)
    s.text(1.24, 3.9, 9.7, 0.9,
           "Chaîne audio • 65 caractéristiques • JSD / ANOVA • XGBoost par model ID", 18, "C9D6E3")
    s.rect(1.25, 5.37, 2.45, 0.48, NAVY2, radius=True)
    s.text(1.35, 5.46, 2.25, 0.24, "PRÉSENTATION DU PROJET", 9, WHITE, True, "c")
    s.text(1.25, 6.75, 5.0, 0.25, "29 octobre 2026", 10, "98AABD")
    s.text(10.65, 6.72, 1.9, 0.25, "MIMII • 0 dB", 10, TEAL, True, "r")
    slides.append(s)

    # 2 — dataset
    s = Slide(2)
    s.header("01 • Données", "Objectif et jeu de données", "Détecter un état normal ou anormal pour les enregistrements MIMII 0 dB Slide Rail.")
    add_card(s, 0.7, 1.9, 5.85, 1.55, "MIMII — Slide Rail", "Bruit de fond à 0 dB SNR.\nClasses : normal et abnormal.", TEAL)
    add_card(s, 6.78, 1.9, 5.85, 1.55, "Quatre model IDs", "id_00 • id_02 • id_04 • id_06\nCe sont des modèles de produit, pas des personnes.", BLUE)
    add_card(s, 0.7, 3.72, 5.85, 1.55, "Enregistrement", "8 microphones • 16 kHz • 16 bits\nSegments WAV de 10 secondes.", ORANGE)
    add_card(s, 6.78, 3.72, 5.85, 1.55, "But du modèle", "Classer les sons de chaque modèle connu.\nUn classifieur XGBoost est entraîné par ID.", GREEN)
    s.text(0.75, 5.75, 11.7, 0.7,
           "MIMII public 1.0 contient quatre model IDs disponibles (00, 02, 04, 06) pour chaque type de machine.",
           12, MUTED)
    s.text(0.75, 6.35, 11.7, 0.35, "Source : zenodo.org/records/3384388", 9, BLUE)
    slides.append(s)

    # 3 — processing
    s = Slide(3)
    s.header("02 • Prétraitement", "De l’audio aux fenêtres", "Les mêmes réglages sont appliqués à tous les WAV de ce run.")
    steps = [
        (0.72, "01", "Paire de micros", "Canaux 3 et 5\nMoyenne EGC", TEAL),
        (3.82, "02", "Filtrage", "Butterworth, ordre 4\n300–7 000 Hz", BLUE),
        (6.92, "03", "Fenêtrage", "1 seconde\n50 % de recouvrement", ORANGE),
        (10.02, "04", "Dataset", "4 094 WAV\n77 786 fenêtres", GREEN),
    ]
    for x, n, title, body, accent in steps:
        s.rect(x, 2.25, 2.55, 2.0, WHITE, radius=True, line="E0E8EF")
        s.text(x + 0.2, 2.42, 0.55, 0.35, n, 17, accent, True)
        s.text(x + 0.2, 2.92, 2.18, 0.55, title, 16, INK, True)
        s.text(x + 0.2, 3.52, 2.2, 0.55, body, 12, MUTED)
    for x in (3.37, 6.47, 9.57):
        s.text(x, 3.02, 0.35, 0.45, "›", 24, TEAL, True, "c")
    s.rect(0.75, 4.82, 11.8, 0.9, NAVY, radius=True)
    s.text(0.98, 5.02, 11.3, 0.45,
           "La paire 3–5 a été tirée au sort une fois puis gardée identique pour tous les enregistrements.",
           13, WHITE, True, "c")
    s.text(0.8, 6.0, 11.6, 0.45,
           "Fréquence vérifiée : 16 kHz, compatible avec la coupure haute de 7 kHz.", 11, MUTED)
    slides.append(s)

    # 4 — features
    s = Slide(4)
    s.header("03 • Caractéristiques", "65 caractéristiques par fenêtre", "Les caractéristiques sont extraites après filtrage, sur chaque fenêtre d’une seconde.")
    cards = [
        (0.72, 1.95, 3.82, "12 temporelles", "Moyenne, écart-type, RMS,\nénergie, ZCR, kurtosis,\nskewness, crest factor…", TEAL, "12"),
        (4.76, 1.95, 3.82, "5 spectrales", "Centroid, bandwidth, roll-off 85 %,\nflatness et peak ratio.", BLUE, "05"),
        (8.80, 1.95, 3.82, "48 Wavelet Packet", "db4 • niveau 4 • 16 sous-bandes\nPour chaque bande : RMS relatif,\nkurtosis et entropie.", ORANGE, "48"),
    ]
    for x, y, w, title, body, accent, number in cards:
        s.rect(x, y, w, 2.65, WHITE, radius=True, line="E0E8EF")
        s.text(x + 0.23, y + 0.22, 1.0, 0.52, number, 25, accent, True)
        s.text(x + 0.23, y + 0.88, w - 0.44, 0.45, title, 16, INK, True)
        s.text(x + 0.23, y + 1.48, w - 0.45, 0.95, body, 12, MUTED)
    s.rect(0.75, 5.05, 11.8, 0.75, PALE, radius=True)
    s.text(0.95, 5.22, 11.4, 0.35, "12 + 5 + 48 = 65 caractéristiques", 17, INK, True, "c")
    s.text(0.8, 6.05, 11.6, 0.45, "Fichier produit : features_0dB/features_65.csv", 11, MUTED)
    slides.append(s)

    # 5 — selectors
    s = Slide(5)
    s.header("04 • Sélection", "JSD et ANOVA : classement des variables", "Les deux méthodes notent les caractéristiques; XGBoost apprend ensuite la frontière de décision.")
    add_card(s, 0.78, 1.95, 5.7, 2.08, "JSD", "Divergence entre les distributions\nnormal et abnormal.\n\nLes 60 meilleures variables ont été\nretenues pour les modèles finaux.", TEAL, 13)
    add_card(s, 6.78, 1.95, 5.7, 2.08, "ANOVA", "Score F entre les classes.\n\nMême nombre de variables pour une\ncomparaison comparable : top 60.", BLUE, 13)
    s.rect(0.8, 4.5, 11.65, 0.9, NAVY, radius=True)
    s.text(1.0, 4.7, 11.25, 0.45,
           "Le classement des variables est recalculé uniquement dans les données d’entraînement de chaque fold.",
           13, WHITE, True, "c")
    s.text(0.85, 5.8, 11.2, 0.62,
           "Le nombre 60 a été gardé après avoir comparé plusieurs tailles de sous-ensemble sur validation.",
           12, MUTED)
    slides.append(s)

    # 6 — model strategy
    s = Slide(6)
    s.header("05 • Modélisation", "Un XGBoost par model ID", "Les modèles de produit ont des signatures acoustiques différentes.")
    s.rect(0.8, 1.95, 3.65, 3.8, NAVY, radius=True)
    s.text(1.05, 2.25, 3.1, 0.65, "4 modèles", 24, WHITE, True, "c")
    s.text(1.1, 3.05, 3.0, 1.55, "id_00\nid_02\nid_04\nid_06", 19, "D5E2EF", True, "c")
    s.text(4.9, 2.05, 7.4, 0.42, "Réglages communs", 17, INK, True)
    bullet(s, 4.92, 2.7, 7.0, "XGBoost : 200 arbres, profondeur maximale 4.", 13)
    bullet(s, 4.92, 3.35, 7.0, "JSD : 60 caractéristiques sélectionnées par ID.", 13, INK, BLUE)
    bullet(s, 4.92, 4.0, 7.0, "Seuil de décision estimé sur les scores out-of-fold.", 13, INK, ORANGE)
    bullet(s, 4.92, 4.65, 7.0, "Prédiction finale au niveau WAV : moyenne des probabilités des fenêtres.", 13, INK, GREEN)
    s.text(4.95, 5.45, 7.2, 0.4, "Les IDs sont des modèles de produit MIMII, pas des sujets humains.", 11, MUTED)
    slides.append(s)

    # 7 — CV results
    s = Slide(7)
    s.header("06 • Résultats", "Validation croisée groupée par WAV", "5 folds par model ID • toutes les fenêtres d’un WAV restent dans le même fold.")
    # table header
    cols = [(0.95, "Méthode", 2.5), (4.25, "ROC AUC macro", 2.5), (7.2, "Balanced accuracy", 3.0), (10.45, "Rappel abnormal", 2.0)]
    s.rect(0.82, 2.05, 11.75, 0.65, NAVY, radius=True)
    for x, label, w in cols:
        s.text(x, 2.2, w, 0.32, label, 12, WHITE, True)
    data = [
        ("JSD • top 60", "0,9982", "0,9627", "0,9260", TEAL),
        ("ANOVA • top 60", "0,9983", "0,9654", "0,9316", BLUE),
    ]
    for i, (method, auc, bal, rec, accent) in enumerate(data):
        y = 2.95 + i * 1.05
        s.rect(0.82, y, 11.75, 0.82, WHITE, radius=True, line="E1E8EF")
        s.rect(0.82, y, 0.08, 0.82, accent, radius=True)
        s.text(1.03, y + 0.2, 2.8, 0.38, method, 15, INK, True)
        s.text(4.32, y + 0.2, 2.3, 0.38, auc, 16, INK, True)
        s.text(7.28, y + 0.2, 2.5, 0.38, bal, 16, INK, True)
        s.text(10.5, y + 0.2, 1.6, 0.38, rec, 16, INK, True)
    s.rect(0.85, 5.45, 11.7, 0.8, "E5F3F2", radius=True)
    s.text(1.05, 5.64, 11.2, 0.36,
           "Conclusion : résultats pratiquement équivalents; JSD reste le choix retenu, sans supériorité démontrée.",
           12, INK, True, "c")
    s.text(0.9, 6.45, 11.4, 0.3, "Estimation exploratoire : le même dataset avait déjà été utilisé dans les étapes précédentes.", 9, MUTED)
    slides.append(s)

    # 8 — cross ID vs per ID
    s = Slide(8)
    s.header("07 • Interprétation", "Pourquoi entraîner un modèle par ID ?", "Un modèle entraîné sur plusieurs IDs puis testé sur un ID inédit répond à une question plus difficile.")
    s.rect(0.82, 2.0, 5.55, 2.65, WHITE, radius=True, line="E1E8EF")
    s.text(1.08, 2.25, 5.0, 0.42, "Transfert vers un ID inédit", 16, INK, True)
    s.text(1.1, 2.95, 4.9, 0.56, "ROC AUC 0,693", 22, RED, True)
    s.text(1.12, 3.72, 4.8, 0.6, "Balanced accuracy 0,662\nPrecision abnormal 0,216", 12, MUTED)
    s.rect(6.85, 2.0, 5.55, 2.65, WHITE, radius=True, line="E1E8EF")
    s.text(7.1, 2.25, 5.0, 0.42, "Modèles séparés par ID connu", 16, INK, True)
    s.text(7.13, 2.95, 4.9, 0.56, "ROC AUC macro ≈ 0,998", 22, GREEN, True)
    s.text(7.14, 3.72, 4.8, 0.6, "CV groupée par WAV\nJSD et ANOVA presque à égalité", 12, MUTED)
    s.text(0.9, 5.15, 11.3, 0.75,
           "Ces scores ne mesurent pas la même capacité : le premier teste le transfert entre produits; le second teste de nouveaux WAV d’un produit connu.",
           13, INK, True, "c")
    s.text(0.95, 6.25, 11.2, 0.35, "Dans les deux cas, la réutilisation antérieure des données rend les résultats exploratoires.", 10, MUTED)
    slides.append(s)

    # 9 — inference
    s = Slide(9)
    s.header("08 • Utilisation", "Lancer une prédiction", "Le model ID du fichier doit être connu pour sélectionner le bon classifieur.")
    s.rect(0.82, 2.05, 11.7, 1.25, NAVY, radius=True)
    s.text(1.1, 2.32, 11.15, 0.7,
           "python scripts/predict_wav_xgboost.py --input recording.wav --model-id id_00",
           14, WHITE, True, font="Consolas", valign="ctr")
    add_card(s, 0.85, 3.75, 3.68, 1.65, "Entrée", "WAV multicanal\nModel ID connu", TEAL, 12)
    add_card(s, 4.82, 3.75, 3.68, 1.65, "Sorties", "Probabilité abnormal\npar fenêtre", BLUE, 12)
    add_card(s, 8.79, 3.75, 3.68, 1.65, "Résumé", "CSV de prédictions\net décision au niveau WAV", ORANGE, 12)
    s.text(0.9, 5.95, 11.4, 0.55,
           "Le test d’exécution effectué sur un WAV normal d’entraînement a confirmé que le script fonctionne; ce n’est pas un test indépendant.",
           11, MUTED)
    slides.append(s)

    # 10 — limitations
    s = Slide(10)
    s.header("09 • Limites", "Ce que les résultats permettent de conclure", "La validation doit correspondre au scénario de déploiement visé.")
    bullet(s, 0.9, 1.95, 11.3, "Les modèles par ID fonctionnent bien sur les WAV du même modèle produit dans la CV groupée.", 14)
    bullet(s, 0.9, 2.75, 11.3, "Le transfert vers un ID nouveau reste plus difficile; la precision sur le test cross-ID était faible.", 14, INK, RED)
    bullet(s, 0.9, 3.55, 11.3, "Les 4 model IDs publics ont déjà participé à l’exploration; il faut de nouvelles données pour un test externe indépendant.", 14, INK, ORANGE)
    bullet(s, 0.9, 4.35, 11.3, "Le baseline autoencodeur essayé est une approximation scikit-learn, pas une reproduction exacte de l’implémentation officielle.", 14, INK, BLUE)
    s.rect(0.86, 5.48, 11.58, 0.85, NAVY, radius=True)
    s.text(1.08, 5.68, 11.1, 0.4,
           "Modèle retenu : JSD top 60 + XGBoost par model ID, à utiliser pour un ID connu.",
           13, WHITE, True, "c")
    slides.append(s)

    # 11 — references
    s = Slide(11)
    s.header("Références", "Sources et livrables", "Sources officielles et fichiers générés dans le projet.")
    s.text(0.9, 1.9, 11.3, 0.36, "MIMII Dataset, public 1.0 — Zenodo", 15, INK, True)
    s.text(0.92, 2.38, 11.2, 0.32, "https://zenodo.org/records/3384388", 11, BLUE)
    s.text(0.9, 3.05, 11.3, 0.36, "Baseline officiel MIMII (autoencodeur normal-only)", 15, INK, True)
    s.text(0.92, 3.53, 11.2, 0.32, "https://github.com/MIMII-hitachi/mimii_baseline", 11, BLUE)
    s.text(0.9, 4.25, 11.3, 0.36, "Modèles finaux par ID", 15, INK, True)
    s.text(0.92, 4.72, 11.2, 0.32, "xgboost_comparison/final_per_model_id/", 11, MUTED, font="Consolas")
    s.text(0.9, 5.4, 11.3, 0.36, "CV groupée par WAV — comparaison JSD / ANOVA", 15, INK, True)
    s.text(0.92, 5.87, 11.2, 0.32, "xgboost_comparison/per_model_id_cv/", 11, MUTED, font="Consolas")
    slides.append(s)
    return slides


def xml_bytes(element: ET.Element) -> bytes:
    return ET.tostring(element, encoding="utf-8", xml_declaration=True)


def relationships(items: list[tuple[str, str, str]]) -> bytes:
    root = ET.Element(q(REL, "Relationships"))
    for rid, rel_type, target in items:
        ET.SubElement(root, q(REL, "Relationship"), {"Id": rid, "Type": rel_type, "Target": target})
    return xml_bytes(root)


def package_pptx(slides: list[Slide], out: Path) -> None:
    content_types = ET.Element(q(CT, "Types"))
    ET.SubElement(content_types, q(CT, "Default"), {"Extension": "rels", "ContentType": "application/vnd.openxmlformats-package.relationships+xml"})
    ET.SubElement(content_types, q(CT, "Default"), {"Extension": "xml", "ContentType": "application/xml"})
    overrides = [
        ("/ppt/presentation.xml", "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"),
        ("/ppt/slideMasters/slideMaster1.xml", "application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"),
        ("/ppt/slideLayouts/slideLayout1.xml", "application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"),
        ("/ppt/theme/theme1.xml", "application/vnd.openxmlformats-officedocument.theme+xml"),
        ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml"),
        ("/docProps/app.xml", "application/vnd.openxmlformats-officedocument.extended-properties+xml"),
    ]
    overrides.extend((f"/ppt/slides/slide{i}.xml", "application/vnd.openxmlformats-officedocument.presentationml.slide+xml") for i in range(1, len(slides) + 1))
    for part, ctype in overrides:
        ET.SubElement(content_types, q(CT, "Override"), {"PartName": part, "ContentType": ctype})

    pres = ET.Element(q(P, "presentation"), {"saveSubsetFonts": "1"})
    master_ids = ET.SubElement(pres, q(P, "sldMasterIdLst"))
    ET.SubElement(master_ids, q(P, "sldMasterId"), {"id": "2147483648", q(R, "id"): "rId1"})
    sld_ids = ET.SubElement(pres, q(P, "sldIdLst"))
    pres_rels = [("rId1", f"{R}/slideMaster", "slideMasters/slideMaster1.xml")]
    for i in range(1, len(slides) + 1):
        rid = f"rId{i+1}"
        ET.SubElement(sld_ids, q(P, "sldId"), {"id": str(255 + i), q(R, "id"): rid})
        pres_rels.append((rid, f"{R}/slide", f"slides/slide{i}.xml"))
    ET.SubElement(pres, q(P, "sldSz"), {"cx": str(int(SW)), "cy": str(int(SH)), "type": "screen16x9"})
    ET.SubElement(pres, q(P, "notesSz"), {"cx": str(int(6858000)), "cy": str(int(9144000))})

    master = ET.Element(q(P, "sldMaster"))
    mc = ET.SubElement(master, q(P, "cSld"))
    mtree = ET.SubElement(mc, q(P, "spTree"))
    _add_empty_group(mtree)
    ET.SubElement(master, q(P, "clrMap"), {"bg1": "lt1", "tx1": "dk1", "bg2": "lt2", "tx2": "dk2",
                                              "accent1": "accent1", "accent2": "accent2", "accent3": "accent3",
                                              "accent4": "accent4", "accent5": "accent5", "accent6": "accent6",
                                              "hlink": "hlink", "folHlink": "folHlink"})
    layout_ids = ET.SubElement(master, q(P, "sldLayoutIdLst"))
    ET.SubElement(layout_ids, q(P, "sldLayoutId"), {"id": "1", q(R, "id"): "rId1"})
    tx = ET.SubElement(master, q(P, "txStyles"))
    for tag in ("titleStyle", "bodyStyle", "otherStyle"):
        sty = ET.SubElement(tx, q(P, tag))
        ET.SubElement(sty, q(A, "lvl1pPr"))

    layout = ET.Element(q(P, "sldLayout"), {"type": "blank", "preserve": "1"})
    lc = ET.SubElement(layout, q(P, "cSld"), {"name": "Blank"})
    ltree = ET.SubElement(lc, q(P, "spTree"))
    _add_empty_group(ltree)
    clr = ET.SubElement(layout, q(P, "clrMapOvr"))
    ET.SubElement(clr, q(A, "masterClrMapping"))

    theme = _theme_xml()
    core = ET.Element("{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}coreProperties")
    ET.SubElement(core, "{http://purl.org/dc/elements/1.1/}title").text = "Détection d’anomalies MIMII 0 dB — Slide Rail"
    ET.SubElement(core, "{http://purl.org/dc/terms/}created", {"{http://www.w3.org/2001/XMLSchema-instance}type": "dcterms:W3CDTF"}).text = "2026-10-29T00:00:00Z"
    app = ET.Element("{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Properties")
    ET.SubElement(app, "{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Application").text = "Microsoft Office PowerPoint"
    ET.SubElement(app, "{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}PresentationFormat").text = "On-screen Show (16:9)"
    ET.SubElement(app, "{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Slides").text = str(len(slides))

    with ZipFile(out, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", xml_bytes(content_types))
        zf.writestr("_rels/.rels", relationships([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties", "docProps/app.xml"),
        ]))
        zf.writestr("ppt/presentation.xml", xml_bytes(pres))
        zf.writestr("ppt/_rels/presentation.xml.rels", relationships(pres_rels))
        zf.writestr("ppt/slideMasters/slideMaster1.xml", xml_bytes(master))
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", relationships([
            ("rId1", f"{R}/slideLayout", "../slideLayouts/slideLayout1.xml"),
            ("rId2", f"{R}/theme", "../theme/theme1.xml"),
        ]))
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", xml_bytes(layout))
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", relationships([
            ("rId1", f"{R}/slideMaster", "../slideMasters/slideMaster1.xml")
        ]))
        zf.writestr("ppt/theme/theme1.xml", xml_bytes(theme))
        for i, slide in enumerate(slides, start=1):
            clr_ovr = ET.SubElement(slide.root, q(P, "clrMapOvr"))
            ET.SubElement(clr_ovr, q(A, "masterClrMapping"))
            zf.writestr(f"ppt/slides/slide{i}.xml", xml_bytes(slide.root))
            zf.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", relationships([
                ("rId1", f"{R}/slideLayout", "../slideLayouts/slideLayout1.xml")
            ]))
        zf.writestr("docProps/core.xml", xml_bytes(core))
        zf.writestr("docProps/app.xml", xml_bytes(app))


def _add_empty_group(parent: ET.Element) -> None:
    nv = ET.SubElement(parent, q(P, "nvGrpSpPr"))
    ET.SubElement(nv, q(P, "cNvPr"), {"id": "1", "name": ""})
    ET.SubElement(nv, q(P, "cNvGrpSpPr"))
    ET.SubElement(nv, q(P, "nvPr"))
    grp = ET.SubElement(parent, q(P, "grpSpPr"))
    xfrm = ET.SubElement(grp, q(A, "xfrm"))
    ET.SubElement(xfrm, q(A, "off"), {"x": "0", "y": "0"})
    ET.SubElement(xfrm, q(A, "ext"), {"cx": "0", "cy": "0"})
    ET.SubElement(xfrm, q(A, "chOff"), {"x": "0", "y": "0"})
    ET.SubElement(xfrm, q(A, "chExt"), {"cx": "0", "cy": "0"})


def _theme_xml() -> ET.Element:
    a = A
    theme = ET.Element(q(a, "theme"), {"name": "MIMII Project"})
    elements = ET.SubElement(theme, q(a, "themeElements"))
    colors = ET.SubElement(elements, q(a, "clrScheme"), {"name": "MIMII"})
    for name, tag, val in [
        ("dk1", "srgbClr", NAVY), ("lt1", "srgbClr", WHITE), ("dk2", "srgbClr", INK),
        ("lt2", "srgbClr", BG), ("accent1", "srgbClr", TEAL), ("accent2", "srgbClr", BLUE),
        ("accent3", "srgbClr", ORANGE), ("accent4", "srgbClr", GREEN), ("accent5", "srgbClr", RED),
        ("accent6", "srgbClr", MUTED), ("hlink", "srgbClr", BLUE), ("folHlink", "srgbClr", "7952A8")
    ]:
        node = ET.SubElement(colors, q(a, name))
        ET.SubElement(node, q(a, tag), {"val": val})
    fonts = ET.SubElement(elements, q(a, "fontScheme"), {"name": "Aptos"})
    for kind, typeface in (("majorFont", "Aptos Display"), ("minorFont", "Aptos")):
        node = ET.SubElement(fonts, q(a, kind))
        ET.SubElement(node, q(a, "latin"), {"typeface": typeface})
        ET.SubElement(node, q(a, "ea"), {"typeface": ""})
        ET.SubElement(node, q(a, "cs"), {"typeface": ""})
    fmt = ET.SubElement(elements, q(a, "fmtScheme"), {"name": "Simple"})
    fill_list = ET.SubElement(fmt, q(a, "fillStyleLst"))
    for _ in range(3):
        sf = ET.SubElement(fill_list, q(a, "solidFill"))
        ET.SubElement(sf, q(a, "schemeClr"), {"val": "phClr"})
    line_list = ET.SubElement(fmt, q(a, "lnStyleLst"))
    for width in (6350, 12700, 19050):
        ln = ET.SubElement(line_list, q(a, "ln"), {"w": str(width), "cap": "flat", "cmpd": "sng", "algn": "ctr"})
        solid = ET.SubElement(ln, q(a, "solidFill"))
        ET.SubElement(solid, q(a, "schemeClr"), {"val": "phClr"})
        ET.SubElement(ln, q(a, "prstDash"), {"val": "solid"})
    effect = ET.SubElement(fmt, q(a, "effectStyleLst"))
    for _ in range(3):
        es = ET.SubElement(effect, q(a, "effectStyle"))
        ET.SubElement(es, q(a, "effectLst"))
    bg = ET.SubElement(fmt, q(a, "bgFillStyleLst"))
    for _ in range(3):
        sf = ET.SubElement(bg, q(a, "solidFill"))
        ET.SubElement(sf, q(a, "schemeClr"), {"val": "phClr"})
    return theme


def main() -> None:
    slides = build_slides()
    package_pptx(slides, OUT)
    print(f"Created {OUT} ({len(slides)} slides)")


if __name__ == "__main__":
    main()
