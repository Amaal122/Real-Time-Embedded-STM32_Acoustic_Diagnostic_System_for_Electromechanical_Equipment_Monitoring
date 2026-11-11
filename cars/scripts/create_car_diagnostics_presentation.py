#!/usr/bin/env python3
"""Build an editable French PowerPoint summarizing the car-audio experiments."""

from __future__ import annotations

import html
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "Presentation_MIMII_XGBoost_FR.pptx"
OUTPUT = ROOT / "Presentation_Diagnostic_Acoustique_Vehicule_FR.pptx"
HTML_OUTPUT = ROOT / "Presentation_Diagnostic_Acoustique_Vehicule_FR.html"
EMU = 914400
WIDTH = 12191999
HEIGHT = 6858000

NAVY = "13243A"
INK = "1B2A3A"
MUTED = "627386"
TEAL = "08B8B0"
BLUE = "3D78C5"
ORANGE = "F0A34A"
RED = "D65D63"
GREEN = "42A58B"
PALE = "F3F6FA"
PALE_BLUE = "EAF1F8"
WHITE = "FFFFFF"
LINE = "DBE3EB"


def ex(value: str) -> str:
    return html.escape(str(value), quote=True)


def ix(value: float) -> int:
    return round(value * EMU)


def rect(x: float, y: float, w: float, h: float, fill: str,
         radius: bool = False, line: str | None = None) -> str:
    geom = "roundRect" if radius else "rect"
    outline = f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>' if line else '<a:ln><a:noFill/></a:ln>'
    return (
        '<p:sp><p:nvSpPr><p:cNvPr id="{id}" name="Shape"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
        '<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
        '<a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>'
        '<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>{outline}</p:spPr></p:sp>'
    ).format(id="{id}", x=ix(x), y=ix(y), w=ix(w), h=ix(h), geom=geom, fill=fill, outline=outline)


def text_box(x: float, y: float, w: float, h: float, text: str,
             size: int = 18, color: str = INK, bold: bool = False,
             align: str = "l", valign: str = "t", font: str = "Aptos",
             italic: bool = False, margin: float = 0.02) -> str:
    paragraphs = []
    for line in str(text).split("\n"):
        b = ' b="1"' if bold else ""
        i = ' i="1"' if italic else ""
        paragraphs.append(
            f'<a:p><a:pPr algn="{align}"/><a:r><a:rPr lang="fr-FR" sz="{size * 100}"{b}{i}>'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:latin typeface="{font}"/></a:rPr><a:t xml:space="preserve">{ex(line)}</a:t></a:r>'
            f'<a:endParaRPr lang="fr-FR" sz="{size * 100}"/></a:p>'
        )
    return (
        '<p:sp><p:nvSpPr><p:cNvPr id="{id}" name="Text"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        '<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
        '<p:txBody><a:bodyPr wrap="square" lIns="{m}" rIns="{m}" tIns="{m}" bIns="{m}" anchor="{valign}"/><a:lstStyle/>{paras}</p:txBody></p:sp>'
    ).format(id="{id}", x=ix(x), y=ix(y), w=ix(w), h=ix(h), m=ix(margin), valign=valign, paras="".join(paragraphs))


class Slide:
    def __init__(self, bg: str = PALE):
        self.bg = bg
        self.parts: list[str] = []

    def add(self, xml: str) -> None:
        self.parts.append(xml)

    def box(self, *args, **kwargs) -> None:
        self.add(text_box(*args, **kwargs))

    def rect(self, *args, **kwargs) -> None:
        self.add(rect(*args, **kwargs))

    def card(self, x: float, y: float, w: float, h: float, title: str,
             body: str, accent: str = TEAL, body_size: int = 15) -> None:
        self.rect(x, y, w, h, WHITE, True, LINE)
        self.rect(x, y, 0.075, h, accent, True)
        self.box(x + 0.25, y + 0.19, w - 0.45, 0.4, title, 17, INK, True)
        self.box(x + 0.25, y + 0.68, w - 0.45, h - 0.82, body, body_size, MUTED)

    def heading(self, section: str, title: str, subtitle: str = "") -> None:
        self.box(0.62, 0.28, 5.8, 0.25, section.upper(), 10, TEAL, True)
        self.box(0.62, 0.62, 12.05, 0.55, title, 28, NAVY, True)
        if subtitle:
            self.box(0.64, 1.22, 12.0, 0.4, subtitle, 13, MUTED)

    def footer(self, page: int) -> None:
        self.rect(0.62, 7.08, 12.1, 0.012, LINE)
        self.box(0.65, 7.15, 10.5, 0.18, "DIAGNOSTIC ACOUSTIQUE AUTOMOBILE  •  SYNTHÈSE DES EXPÉRIMENTATIONS", 8, MUTED)
        self.box(12.05, 7.12, 0.6, 0.22, f"{page:02d}", 9, MUTED, True, "r")

    def xml(self, page: int) -> bytes:
        shapes = []
        for index, shape in enumerate(self.parts, 2):
            shapes.append(shape.replace('id="{id}"', f'id="{index}"').replace('id="{{id}}"', f'id="{index}"'))
        xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" showMasterSp="0">'
            '<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="' + self.bg + '"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>'
            '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            + "".join(shapes) + '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
        )
        return xml.encode("utf-8")


def bar(slide: Slide, x: float, y: float, label: str, value: float, max_value: float,
        color: str, suffix: str = "", width: float = 4.1) -> None:
    slide.box(x, y, width, 0.28, label, 12, INK, True)
    slide.rect(x, y + 0.35, width, 0.18, PALE_BLUE, True)
    slide.rect(x, y + 0.35, max(0.08, width * value / max_value), 0.18, color, True)
    slide.box(x + width + 0.12, y + 0.26, 1.0, 0.3, f"{value:.3f}{suffix}", 12, INK, True)


def build_slides() -> list[Slide]:
    slides: list[Slide] = []

    s = Slide(NAVY)
    s.rect(0.0, 0.0, 0.18, 7.5, TEAL)
    s.box(0.82, 0.76, 7.4, 0.35, "PROJET DE DIAGNOSTIC AUDIO", 13, TEAL, True)
    s.box(0.82, 1.48, 10.8, 1.7, "Diagnostic acoustique\ndes pannes automobiles", 34, WHITE, True)
    s.box(0.86, 3.48, 9.4, 0.82, "Traitement du signal, extraction de caractéristiques\net comparaison de modèles supervisés et non supervisés", 19, "D4E0EB")
    s.rect(0.86, 5.25, 6.9, 0.72, "20374F", True)
    s.box(1.12, 5.45, 6.5, 0.28, "1 386 WAV   •   2 772 fenêtres   •   6 familles de pannes", 14, WHITE, True)
    s.box(0.86, 6.76, 8.0, 0.23, "Présentation des travaux réalisés — résultats expérimentaux", 10, "AFC0D0")
    slides.append(s)

    s = Slide(); s.heading("01  •  Données", "Objectif et jeu de données", "Construire une chaîne audio qui détecte une anomalie puis aide à identifier sa famille.")
    s.card(0.7, 1.95, 3.75, 1.7, "1 386 enregistrements", "WAV mono\n13 catégories : normal, panne simple ou combinée.", TEAL, 14)
    s.card(4.78, 1.95, 3.75, 1.7, "3 états du moteur", "Freinage : 153 WAV\nRalenti : 1 053 WAV\nDémarrage : 180 WAV", BLUE)
    s.card(8.86, 1.95, 3.75, 1.7, "Labels du dataset", "worn_out_brakes · bad_ignition\ndead_battery · low_oil\npower_steering · serpentine_belt", ORANGE, 12)
    s.box(0.86, 4.12, 11.5, 0.5, "Point important", 16, NAVY, True)
    s.box(0.88, 4.62, 11.4, 1.2, "Le dataset contient aussi des enregistrements avec plusieurs défauts simultanés.\nLa formulation du problème change donc selon l’objectif : détection binaire, classe de panne, ou détection indépendante de chaque panne.", 18, INK)
    s.footer(2); slides.append(s)

    s = Slide(); s.heading("02  •  Prétraitement", "De l’enregistrement aux fenêtres audio", "Les mêmes réglages de signal sont appliqués avant l’extraction des caractéristiques.")
    steps = [
        (0.78, "01", "Rééchantillonnage", "16 kHz", BLUE),
        (3.22, "02", "Filtre passe-bande", "Butterworth, ordre 4\n300–7 000 Hz", TEAL),
        (6.10, "03", "Fenêtrage", "1 seconde\n50 % de recouvrement", ORANGE),
        (9.03, "04", "Représentation", "WPT db4\nniveau 4", GREEN),
    ]
    for x, num, title, body, color in steps:
        s.rect(x, 2.25, 2.1, 2.15, WHITE, True, LINE)
        s.rect(x + 0.2, 2.5, 0.48, 0.48, color, True)
        s.box(x + 0.2, 2.61, 0.48, 0.18, num, 11, WHITE, True, "ctr")
        title_size = 13 if title == "Rééchantillonnage" else 16
        s.box(x + 0.2, 3.15, 1.75, 0.45, title, title_size, NAVY, True)
        s.box(x + 0.2, 3.67, 1.75, 0.68, body, 12, MUTED)
    s.box(0.9, 5.0, 11.4, 0.48, "Résultat : 2 772 fenêtres d’une seconde, issues de 1 386 WAV.", 20, NAVY, True, "ctr")
    s.box(1.1, 5.66, 11.0, 0.62, "La séparation train / validation / test se fait par enregistrement : les fenêtres d’un même WAV restent ensemble afin de limiter les fuites de données.", 14, MUTED, False, "ctr")
    s.footer(3); slides.append(s)

    s = Slide(); s.heading("03  •  Caractéristiques", "65 caractéristiques audio par fenêtre", "Combinaison de statistiques temporelles, spectrales et wavelet packet.")
    s.card(0.8, 2.0, 3.7, 2.75, "12 temporelles", "Moyenne, écart-type, RMS\nÉnergie, ZCR, crête-à-crête\nKurtosis, skewness, crest factor\nMaximum, minimum, moyenne absolue", TEAL, 14)
    s.card(4.82, 2.0, 3.7, 2.75, "5 spectrales", "Centroïde spectral\nBandwidth\nRoll-off 85 %\nFlatness\nPeak ratio", BLUE, 14)
    s.card(8.84, 2.0, 3.7, 2.75, "48 wavelet packet", "Ondelette db4 au niveau 4\n16 sous-bandes\nPour chaque bande : RMS relatif, kurtosis et entropie", ORANGE, 14)
    s.rect(3.0, 5.45, 7.3, 0.72, NAVY, True)
    s.box(3.2, 5.66, 6.9, 0.28, "12 + 5 + (16 × 3) = 65 caractéristiques", 18, WHITE, True, "ctr")
    s.footer(4); slides.append(s)

    s = Slide(); s.heading("04  •  Formulations", "Trois objectifs de classification", "Les scores ne sont comparables que si la tâche et les classes évaluées sont identiques.")
    s.card(0.78, 2.0, 3.75, 3.45, "A — Détection binaire", "Question : normal ou anormal ?\n\nXGBoost avec sélection ANOVA ou JSD.\nAutoencodeur MLP entraîné seulement sur les sons normaux.", BLUE, 15)
    s.card(4.78, 2.0, 3.75, 3.45, "B — Type de panne", "Question : quelle catégorie du dataset ?\n\nXGBoost JSD / ANOVA, Random Forest et CNN.\nLes pannes combinées constituent des classes distinctes.", TEAL, 15)
    s.card(8.78, 2.0, 3.75, 3.45, "C — Spécialistes par panne", "Un détecteur binaire par famille de panne.\n\nLes dossiers de pannes combinées comptent comme positifs pour chaque défaut présent.", ORANGE, 15)
    s.box(1.0, 6.0, 11.3, 0.45, "JSD et ANOVA classent les caractéristiques; ce ne sont pas des classifieurs de panne.", 15, NAVY, True, "ctr")
    s.footer(5); slides.append(s)

    s = Slide(); s.heading("05  •  Détection", "Normal / anormal : XGBoost devant l’autoencodeur", "Évaluation sur 278 enregistrements du jeu de test — seuils choisis sur validation.")
    s.rect(0.8, 1.9, 11.7, 0.56, NAVY, True)
    cols = [(1.05,"Modèle"),(5.1,"Accuracy"),(6.75,"Balanced acc."),(8.7,"Précision anorm."),(10.65,"F1 anormal")]
    for x,t in cols: s.box(x,2.08,1.7,0.2,t,11,WHITE,True)
    rows = [
        ("XGBoost ANOVA", "0,881", "0,898", "0,971", "0,912", TEAL),
        ("XGBoost JSD", "0,871", "0,898", "0,982", "0,902", BLUE),
        ("Autoencodeur MLP", "0,629", "0,718", "0,944", "0,662", ORANGE),
    ]
    for i,(name,a,b,c,d,color) in enumerate(rows):
        y=2.58+i*0.78
        s.rect(0.8,y,11.7,0.67, WHITE if i%2==0 else PALE_BLUE, True)
        vals=[(1.05,name),(5.1,a),(6.75,b),(8.7,c),(10.65,d)]
        for x,t in vals: s.box(x,y+0.2,1.7,0.22,t,14,INK, x==1.05)
    s.card(0.85, 5.2, 5.6, 1.2, "Lecture", "ANOVA : F1 anormal supérieur; JSD : précision un peu plus haute.", TEAL, 11)
    s.card(6.75, 5.2, 5.6, 1.2, "Autoencodeur", "Moins performant ici; peut aider si seules les données normales sont disponibles.", ORANGE, 11)
    s.footer(6); slides.append(s)

    s = Slide(); s.heading("06  •  Diagnostic", "Classification du type de panne", "10 catégories anormales, incluant plusieurs configurations combinées — 198 WAV de test.")
    s.card(0.82, 2.0, 3.75, 1.55, "XGBoost ANOVA", "Accuracy 0,601\nBalanced accuracy 0,607\nMacro-F1 0,616", TEAL)
    s.card(4.8, 2.0, 3.75, 1.55, "XGBoost JSD", "Accuracy 0,596\nBalanced accuracy 0,602\nMacro-F1 0,615", BLUE)
    s.card(8.78, 2.0, 3.75, 1.55, "CNN log-Mel", "Accuracy 0,475\nBalanced accuracy 0,540\nMacro-F1 0,504", ORANGE)
    s.box(0.92, 3.78, 11.0, 0.32, "Cascade XGBoost ANOVA (détection puis diagnostic) : accuracy 0,622 ; macro-F1 0,559.", 13, MUTED, True)
    s.box(0.92, 4.15, 11.0, 0.32, "Sélection de modèle sur validation : Random Forest (65 variables)", 15, NAVY, True)
    bar(s, 0.98, 4.55, "Random Forest — macro-F1 test", 0.651, 1.0, GREEN, width=8.8)
    s.box(1.0, 5.35, 11.0, 0.52, "Le Random Forest dépasse légèrement les XGBoost multiclasses dans cette expérience. Les classes combinées et le déséquilibre rendent la tâche difficile.", 13, MUTED)
    s.box(1.0, 6.05, 11.0, 0.32, "Comparaison rétrospective : le jeu de test a déjà été utilisé dans des comparaisons antérieures.", 11, RED, True)
    s.footer(7); slides.append(s)

    s = Slide(); s.heading("07  •  Détecteurs spécialisés", "Random Forest : résultats par famille de panne", "OOF groupé par WAV, avec seuil max-F1 choisi dans la validation interne de chaque fold.")
    specialists = [
        ("serpentine_belt", .857, GREEN),
        ("low_oil", .817, TEAL),
        ("worn_out_brakes", .671, BLUE),
        ("bad_ignition", .621, ORANGE),
        ("dead_battery", .602, RED),
        ("power_steering", .720, "6C83B5"),
    ]
    for i,(name,value,color) in enumerate(specialists):
        col=i//3; row=i%3
        bar(s, 0.95 + col*6.3, 2.0 + row*1.12, name, value, 1.0, color, width=3.65)
    s.box(0.95, 5.75, 11.3, 0.65, "Meilleurs F1 : serpentine_belt et low_oil. Les seuils influencent le compromis entre fausses alertes et pannes manquées.", 15, NAVY, True, "ctr")
    s.footer(8); slides.append(s)

    s = Slide(); s.heading("08  •  Prototype microphone", "Du fichier audio à la prédiction", "Capture de 30 secondes, fenêtrage, scores des détecteurs, puis CSV par fenêtre.")
    flow = [(1.0,"PC source","Lecture du son"),(4.0,"Microphone","Capture WAV"),(7.0,"Random Forest","6 spécialistes"),(10.0,"Résultat","Scores + seuils")]
    for x,title,body in flow:
        s.rect(x,2.25,2.1,1.35,WHITE,True,LINE)
        s.box(x+0.12,2.5,1.86,0.3,title,16,NAVY,True,"ctr")
        s.box(x+0.12,2.98,1.86,0.35,body,12,MUTED,False,"ctr")
    for x in [3.2,6.2,9.2]:
        s.box(x,2.7,.55,.28,"→",22,TEAL,True,"ctr")
    s.card(0.9,4.35,5.55,1.4,"Ce qui fonctionne", "Le script enregistre puis prédit; les seuils max-F1 sont appliqués par détecteur.", TEAL, 13)
    s.card(6.8,4.35,5.55,1.4,"Limite observée", "Le son diffusé par haut-parleur puis capté par un micro diffère du WAV source; des faux positifs ont été observés.", RED, 13)
    s.box(1.0,6.15,11.2,.4,"Le prototype micro doit être validé sur plusieurs captures connues avant toute interprétation réelle.",14,NAVY,True,"ctr")
    s.footer(9); slides.append(s)

    s = Slide(); s.heading("09  •  Synthèse", "Quel modèle choisir selon l’objectif ?", "Les résultats mesurent des tâches différentes; il n’existe pas un classement unique.")
    s.card(0.85,2.0,3.72,3.0,"Détecter une anomalie", "Choix expérimental : XGBoost ANOVA.\n\nF1 anormal : 0,912\nBalanced accuracy : 0,898\n\nRésultat sur 278 WAV de test.", BLUE, 14)
    s.card(4.8,2.0,3.72,3.0,"Nommer une panne", "Random Forest sélectionné par validation.\n\nMacro-F1 test : 0,651\n\nLes spécialistes donnent un F1 par défaut et permettent plusieurs pannes.", GREEN, 14)
    s.card(8.75,2.0,3.72,3.0,"Autoencodeur normal-only", "Autoencodeur MLP.\n\nF1 anormal : 0,662\n\nMoins bon que XGBoost dans cette expérience, mais adapté à un autre scénario de données.", ORANGE, 14)
    s.box(1.0,5.62,11.2,.65,"La détection binaire répond « panne ou non »; le diagnostic répond « quelle panne ? » — les métriques ne se comparent pas directement.",15,NAVY,True,"ctr")
    s.footer(10); slides.append(s)

    s = Slide(NAVY)
    s.rect(0.0,0.0,0.18,7.5,TEAL)
    s.box(0.78,0.52,5.5,.28,"10  •  CONCLUSION ET SUITE",10,TEAL,True)
    s.box(0.78,.96,11.5,.65,"Bilan et prochaines étapes",30,WHITE,True)
    s.card(0.82,2.0,5.55,2.75,"Acquis", "Chaîne audio complète : prétraitement et 65 caractéristiques.\n\nComparaison binaire, multiclasses, autoencodeur, CNN et spécialistes Random Forest.\n\nSéparation par WAV pour l’évaluation groupée des spécialistes.", TEAL, 14)
    s.card(6.72,2.0,5.55,2.75,"À consolider", "Évaluer sur un nouveau jeu indépendant.\n\nAjouter des captures micro étiquetées et variées.\n\nRenforcer les classes rares et analyser les erreurs par panne.\n\nRecalibrer les seuils sans utiliser le jeu de test.", ORANGE, 14)
    s.box(1.0,5.5,11.1,.72,"Conclusion : résultats prometteurs pour la détection binaire; l’identification des pannes et l’usage au microphone demandent encore une validation plus robuste.",18,WHITE,True,"ctr")
    s.box(.82,6.8,9.0,.23,"PROJET DE DIAGNOSTIC ACOUSTIQUE AUTOMOBILE  •  FIN",9,"AFC0D0")
    slides.append(s)
    return slides


def update_properties(data: bytes) -> bytes:
    root = ET.fromstring(data)
    ns = {"dc": "http://purl.org/dc/elements/1.1/", "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties/"}
    title = root.find("{http://purl.org/dc/elements/1.1/}title")
    if title is not None:
        title.text = "Diagnostic acoustique des pannes automobiles"
    subject = root.find("{http://purl.org/dc/elements/1.1/}subject")
    if subject is not None:
        subject.text = "Synthese des experimentations de traitement du signal et classification"
    creator = root.find("{http://purl.org/dc/elements/1.1/}creator")
    if creator is not None:
        creator.text = "Projet de diagnostic acoustique automobile"
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_html_preview() -> None:
    ns = {
        "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    }
    pxw, pxh = 1280, 720
    pages: list[str] = []
    with zipfile.ZipFile(OUTPUT, "r") as package:
        slide_names = sorted(
            (n for n in package.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=lambda n: int(re.search(r"slide(\d+)", n).group(1)),
        )
        for name in slide_names:
            root = ET.fromstring(package.read(name))
            bg_node = root.find("p:cSld/p:bg/p:bgPr/a:solidFill/a:srgbClr", ns)
            bg = bg_node.attrib.get("val", PALE) if bg_node is not None else PALE
            items = [f'<section class="page" style="background:#{bg}">']
            for shape in root.findall(".//p:sp", ns):
                sppr = shape.find("p:spPr", ns)
                if sppr is None:
                    continue
                xfrm = sppr.find("a:xfrm", ns)
                if xfrm is None:
                    continue
                off, ext = xfrm.find("a:off", ns), xfrm.find("a:ext", ns)
                if off is None or ext is None:
                    continue
                x, y = int(off.attrib["x"]), int(off.attrib["y"])
                w, h = int(ext.attrib["cx"]), int(ext.attrib["cy"])
                styles = [
                    "position:absolute", f"left:{100*x/WIDTH:.5f}%", f"top:{100*y/HEIGHT:.5f}%",
                    f"width:{100*w/WIDTH:.5f}%", f"height:{100*h/HEIGHT:.5f}%", "box-sizing:border-box",
                    "overflow:hidden", "display:flex",
                ]
                geom = sppr.find("a:prstGeom", ns)
                if geom is not None and geom.attrib.get("prst") == "roundRect":
                    styles.append("border-radius:12px")
                fill = sppr.find("a:solidFill/a:srgbClr", ns)
                if fill is not None:
                    styles.append(f"background-color:#{fill.attrib['val']}")
                line = sppr.find("a:ln/a:solidFill/a:srgbClr", ns)
                if line is not None:
                    styles.append(f"border:1px solid #{line.attrib['val']}")
                body = shape.find("p:txBody/a:bodyPr", ns)
                anchor = body.attrib.get("anchor", "t") if body is not None else "t"
                styles.append("flex-direction:column")
                styles.append("justify-content:center" if anchor == "ctr" else "justify-content:flex-start")
                styles.append("align-items:stretch")
                styles.append("padding:0")
                paragraphs = []
                for paragraph in shape.findall("p:txBody/a:p", ns):
                    txt = "".join(t.text or "" for t in paragraph.findall(".//a:t", ns))
                    if not txt:
                        continue
                    run = paragraph.find("a:r/a:rPr", ns)
                    size = int(run.attrib.get("sz", "1400")) / 100 if run is not None else 14
                    bold = run is not None and run.attrib.get("b") == "1"
                    italic = run is not None and run.attrib.get("i") == "1"
                    color_node = run.find("a:solidFill/a:srgbClr", ns) if run is not None else None
                    color = color_node.attrib.get("val", INK) if color_node is not None else INK
                    ppr = paragraph.find("a:pPr", ns)
                    alignment = ppr.attrib.get("algn", "l") if ppr is not None else "l"
                    align_css = {"ctr": "center", "r": "right", "l": "left"}.get(alignment, "left")
                    paragraphs.append(
                        f'<div style="font-family:Aptos,Arial,sans-serif;font-size:{size}pt;color:#{color};'
                        f'font-weight:{700 if bold else 400};font-style:{"italic" if italic else "normal"};'
                        f'text-align:{align_css};line-height:1.16;white-space:pre-wrap">{html.escape(txt)}</div>'
                    )
                if paragraphs:
                    items.append(f'<div style="{";".join(styles)}">{"".join(paragraphs)}</div>')
                else:
                    items.append(f'<div style="{";".join(styles)}"></div>')
            items.append("</section>")
            pages.append("".join(items))
    doc = (
        '<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Diagnostic acoustique automobile</title>'
        '<style>@page{size:13.333in 7.5in;margin:0}*{box-sizing:border-box}html,body{margin:0;padding:0}'
        '.page{width:13.333in;height:7.5in;position:relative;overflow:hidden;page-break-after:always}'
        '.page:last-child{page-break-after:auto}</style></head><body>' + "".join(pages) + "</body></html>"
    )
    HTML_OUTPUT.write_text(doc, encoding="utf-8")


def main() -> None:
    slides = build_slides()
    tmp = OUTPUT.with_suffix(".tmp.pptx")
    with zipfile.ZipFile(TEMPLATE, "r") as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            name = info.filename
            if name.startswith("ppt/slides/slide") and name.endswith(".xml") and "/_rels/" not in name:
                match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
                if match:
                    index = int(match.group(1))
                    if index <= len(slides):
                        dst.writestr(info, slides[index - 1].xml(index))
                        continue
            if name == "ppt/slides/_rels/slide1.xml.rels":
                rels = (
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
                    '</Relationships>'
                ).encode("utf-8")
                dst.writestr(info, rels)
                continue
            rel_match = re.fullmatch(r"ppt/slides/_rels/slide(\d+)\.xml\.rels", name)
            if rel_match:
                rels = (
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
                    '</Relationships>'
                ).encode("utf-8")
                dst.writestr(info, rels)
                continue
            if name == "docProps/core.xml":
                dst.writestr(info, update_properties(src.read(name)))
                continue
            if name == "docProps/app.xml":
                app = src.read(name).replace(b"MIMII XGBoost", b"Diagnostic acoustique automobile")
                dst.writestr(info, app)
                continue
            dst.writestr(info, src.read(name))
    tmp.replace(OUTPUT)
    build_html_preview()
    print(f"Presentation creee : {OUTPUT}")
    print(f"Apercu HTML cree : {HTML_OUTPUT}")
    print(f"Nombre de diapositives : {len(slides)}")


if __name__ == "__main__":
    main()
