"""
generate_pdf_report.py - Génération du rapport PDF complet du projet ML Retail
Auteur: Adam Amara
Module: Machine Learning - GI2
Année Académique: 2025-2026
"""

import os
import sys
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
OUTPUT_PDF = os.path.join(REPORTS_DIR, 'ML_Retail_Analysis_Report_Adam_Amara.pdf')

# Informations du projet
AUTHOR = "Adam Amara"
MODULE = "Machine Learning - GI2"
ACADEMIC_YEAR = "2025-2026"
PROJECT_TITLE = "Analyse des Clients Retail par Machine Learning"
SUBTITLE = "Segmentation, Prédiction du Churn et Estimation des Dépenses"


def get_styles():
    """Crée et retourne les styles personnalisés pour le PDF"""
    styles = getSampleStyleSheet()
    
    # Style pour le titre principal
    styles.add(ParagraphStyle(
        name='MainTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a5276')
    ))
    
    # Style pour le sous-titre
    styles.add(ParagraphStyle(
        name='SubTitle',
        parent=styles['Normal'],
        fontSize=14,
        spaceAfter=6,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2874a6')
    ))
    
    # Style pour les sections
    styles.add(ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=12,
        textColor=colors.HexColor('#1a5276'),
        borderPadding=5,
        borderColor=colors.HexColor('#3498db'),
        borderWidth=0,
        leftIndent=0
    ))
    
    # Style pour les sous-sections
    styles.add(ParagraphStyle(
        name='SubSectionTitle',
        parent=styles['Heading2'],
        fontSize=13,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#2874a6')
    ))
    
    # Style pour le texte normal justifié
    styles.add(ParagraphStyle(
        name='BodyTextCustom',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        leading=14
    ))
    
    # Style pour les informations de l'auteur
    styles.add(ParagraphStyle(
        name='AuthorInfo',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=4,
        alignment=TA_CENTER
    ))
    
    # Style pour les légendes d'images
    styles.add(ParagraphStyle(
        name='Caption',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12,
        alignment=TA_CENTER,
        textColor=colors.grey,
        fontName='Helvetica-Oblique'
    ))
    
    # Style pour les points clés
    styles.add(ParagraphStyle(
        name='KeyPoint',
        parent=styles['Normal'],
        fontSize=11,
        spaceBefore=4,
        spaceAfter=4,
        leftIndent=20,
        bulletIndent=10
    ))
    
    return styles


def add_title_page(story, styles):
    """Ajoute la page de titre"""
    story.append(Spacer(1, 2*inch))
    
    # Titre principal
    story.append(Paragraph(PROJECT_TITLE, styles['MainTitle']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(SUBTITLE, styles['SubTitle']))
    
    story.append(Spacer(1, 1.5*inch))
    
    # Ligne de séparation
    story.append(Table([['']], colWidths=[5*inch], rowHeights=[2],
                       style=TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#3498db'))])))
    
    story.append(Spacer(1, 0.5*inch))
    
    # Informations de l'auteur
    story.append(Paragraph(f"<b>Réalisé par:</b> {AUTHOR}", styles['AuthorInfo']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"<b>Module:</b> {MODULE}", styles['AuthorInfo']))
    story.append(Paragraph(f"<b>Année Académique:</b> {ACADEMIC_YEAR}", styles['AuthorInfo']))
    
    story.append(Spacer(1, 0.5*inch))
    
    # Ligne de séparation
    story.append(Table([['']], colWidths=[5*inch], rowHeights=[2],
                       style=TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#3498db'))])))
    
    story.append(Spacer(1, 1*inch))
    
    # Date
    date_str = datetime.now().strftime("%d %B %Y")
    story.append(Paragraph(f"Date de génération: {date_str}", styles['AuthorInfo']))
    
    story.append(PageBreak())


def add_table_of_contents(story, styles):
    """Ajoute la table des matières"""
    story.append(Paragraph("Table des Matières", styles['MainTitle']))
    story.append(Spacer(1, 0.5*inch))
    
    toc_items = [
        ("1. Introduction et Contexte", "3"),
        ("2. Description du Dataset", "4"),
        ("3. Prétraitement des Données", "5"),
        ("4. Analyse Exploratoire (EDA)", "7"),
        ("5. Réduction de Dimensionnalité (PCA)", "9"),
        ("6. Clustering (K-Means)", "11"),
        ("7. Classification (Prédiction du Churn)", "13"),
        ("8. Régression (Prédiction MonetaryAvg)", "17"),
        ("9. Application Web", "19"),
        ("10. Conclusions et Perspectives", "20"),
    ]
    
    toc_data = []
    for title, page in toc_items:
        toc_data.append([title, page])
    
    toc_table = Table(toc_data, colWidths=[5*inch, 0.5*inch])
    toc_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.lightgrey),
    ]))
    
    story.append(toc_table)
    story.append(PageBreak())


def add_introduction(story, styles):
    """Ajoute la section introduction"""
    story.append(Paragraph("1. Introduction et Contexte", styles['SectionTitle']))
    
    story.append(Paragraph("<b>1.1 Contexte du Projet</b>", styles['SubSectionTitle']))
    intro_text = """
    Ce projet s'inscrit dans le cadre du module Machine Learning du programme GI2 pour l'année 
    académique 2025-2026. L'objectif est d'appliquer des techniques avancées de Machine Learning 
    pour analyser et prédire le comportement des clients d'une entreprise de vente en ligne 
    spécialisée dans les cadeaux.
    """
    story.append(Paragraph(intro_text, styles['BodyTextCustom']))
    
    story.append(Paragraph("<b>1.2 Objectifs du Projet</b>", styles['SubSectionTitle']))
    objectives = [
        "Prétraitement et nettoyage des données clients",
        "Analyse exploratoire approfondie (EDA)",
        "Segmentation des clients par clustering (K-Means)",
        "Prédiction du churn (départ des clients) par classification",
        "Estimation des dépenses moyennes par régression",
        "Développement d'une application web de prédiction"
    ]
    
    for obj in objectives:
        story.append(Paragraph(f"• {obj}", styles['KeyPoint']))
    
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>1.3 Technologies Utilisées</b>", styles['SubSectionTitle']))
    tech_text = """
    <b>Langages:</b> Python 3.x<br/>
    <b>Analyse de données:</b> Pandas, NumPy<br/>
    <b>Visualisation:</b> Matplotlib, Seaborn<br/>
    <b>Machine Learning:</b> Scikit-learn<br/>
    <b>Application Web:</b> Flask, Bootstrap<br/>
    <b>Sérialisation:</b> Joblib
    """
    story.append(Paragraph(tech_text, styles['BodyTextCustom']))
    
    story.append(PageBreak())


def add_dataset_description(story, styles):
    """Ajoute la description du dataset"""
    story.append(Paragraph("2. Description du Dataset", styles['SectionTitle']))
    
    story.append(Paragraph("<b>2.1 Vue d'ensemble</b>", styles['SubSectionTitle']))
    dataset_text = """
    Le dataset utilisé provient d'une entreprise de e-commerce basée au Royaume-Uni, 
    spécialisée dans la vente de cadeaux uniques. Il contient des informations détaillées 
    sur les transactions et comportements des clients.
    """
    story.append(Paragraph(dataset_text, styles['BodyTextCustom']))
    
    # Tableau des statistiques du dataset
    stats_data = [
        ["Métrique", "Valeur"],
        ["Nombre total de clients", "4,372"],
        ["Nombre de features initiales", "52"],
        ["Nombre de features après ingénierie", "84"],
        ["Période couverte", "2009-2011"],
        ["Variable cible (Classification)", "Churn (0/1)"],
        ["Variable cible (Régression)", "MonetaryAvg (£)"],
    ]
    
    stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>2.2 Features Principales (RFM)</b>", styles['SubSectionTitle']))
    rfm_text = """
    L'analyse RFM (Recency, Frequency, Monetary) est au cœur de ce projet:
    """
    story.append(Paragraph(rfm_text, styles['BodyTextCustom']))
    
    rfm_features = [
        "<b>Recency:</b> Nombre de jours depuis le dernier achat",
        "<b>Frequency:</b> Nombre total de commandes",
        "<b>MonetaryTotal:</b> Montant total dépensé (£)",
        "<b>MonetaryAvg:</b> Panier moyen (£)"
    ]
    for feat in rfm_features:
        story.append(Paragraph(f"• {feat}", styles['KeyPoint']))
    
    story.append(PageBreak())


def add_preprocessing_section(story, styles):
    """Ajoute la section prétraitement"""
    story.append(Paragraph("3. Prétraitement des Données", styles['SectionTitle']))
    
    story.append(Paragraph("<b>3.1 Gestion des Valeurs Manquantes</b>", styles['SubSectionTitle']))
    missing_text = """
    Le dataset contenait des valeurs manquantes significatives, notamment pour la variable 
    Age (environ 30% de valeurs manquantes). Les stratégies suivantes ont été appliquées:
    """
    story.append(Paragraph(missing_text, styles['BodyTextCustom']))
    
    strategies = [
        "Variables numériques: Imputation par la médiane (robuste aux outliers)",
        "Variables catégorielles: Imputation par le mode",
        "Age: Imputation médiane par segment RFM pour plus de précision"
    ]
    for s in strategies:
        story.append(Paragraph(f"• {s}", styles['KeyPoint']))
    
    # Image des valeurs manquantes
    missing_img_path = os.path.join(REPORTS_DIR, 'missing_values.png')
    if os.path.exists(missing_img_path):
        story.append(Spacer(1, 0.2*inch))
        img = Image(missing_img_path, width=5.5*inch, height=2.8*inch)
        story.append(img)
        story.append(Paragraph("Figure 3.1: Distribution des valeurs manquantes par colonne", styles['Caption']))
    
    story.append(Paragraph("<b>3.2 Traitement des Outliers</b>", styles['SubSectionTitle']))
    outliers_text = """
    Les outliers ont été traités par la méthode IQR (Interquartile Range) avec 
    clipping aux bornes [Q1 - 1.5*IQR, Q3 + 1.5*IQR] pour les variables numériques 
    sensibles comme MonetaryTotal et TotalQuantity.
    """
    story.append(Paragraph(outliers_text, styles['BodyTextCustom']))
    
    story.append(Paragraph("<b>3.3 Encodage des Variables</b>", styles['SubSectionTitle']))
    encoding_text = """
    Les variables catégorielles ont été encodées selon leur nature:
    """
    story.append(Paragraph(encoding_text, styles['BodyTextCustom']))
    
    encoding_methods = [
        "<b>Label Encoding:</b> Variables ordinales (RFMSegment, SpendingCategory)",
        "<b>One-Hot Encoding:</b> Variables nominales (Region, Country)",
        "<b>Frequency Encoding:</b> Variables à haute cardinalité"
    ]
    for e in encoding_methods:
        story.append(Paragraph(f"• {e}", styles['KeyPoint']))
    
    story.append(Paragraph("<b>3.4 Feature Engineering</b>", styles['SubSectionTitle']))
    fe_text = """
    Plusieurs nouvelles features ont été créées pour enrichir l'analyse:
    """
    story.append(Paragraph(fe_text, styles['BodyTextCustom']))
    
    new_features = [
        "Ratios: AvgItemsPerOrder, MonetaryPerProduct",
        "Indicateurs temporels: IsWeekendShopper, SeasonalPreference",
        "Scores composites: CLV (Customer Lifetime Value), EngagementScore"
    ]
    for f in new_features:
        story.append(Paragraph(f"• {f}", styles['KeyPoint']))
    
    story.append(PageBreak())


def add_eda_section(story, styles):
    """Ajoute la section analyse exploratoire"""
    story.append(Paragraph("4. Analyse Exploratoire (EDA)", styles['SectionTitle']))
    
    story.append(Paragraph("<b>4.1 Matrice de Corrélation</b>", styles['SubSectionTitle']))
    corr_text = """
    L'analyse des corrélations révèle les relations entre les différentes variables du dataset. 
    Les features les plus corrélées avec le Churn sont Recency, Frequency et les variables 
    d'engagement client.
    """
    story.append(Paragraph(corr_text, styles['BodyTextCustom']))
    
    # Image de la matrice de corrélation
    corr_img_path = os.path.join(REPORTS_DIR, 'correlation_matrix.png')
    if os.path.exists(corr_img_path):
        img = Image(corr_img_path, width=5.5*inch, height=4.5*inch)
        story.append(img)
        story.append(Paragraph("Figure 4.1: Matrice de corrélation des features numériques", styles['Caption']))
    
    story.append(Paragraph("<b>4.2 Observations Clés</b>", styles['SubSectionTitle']))
    observations = [
        "Forte corrélation positive entre Frequency et MonetaryTotal (r=0.85)",
        "Corrélation négative entre Recency et la fidélité client",
        "Les clients à haut MonetaryAvg ont tendance à avoir un Churn plus faible",
        "La variable Age montre une corrélation modérée avec les habitudes d'achat"
    ]
    for obs in observations:
        story.append(Paragraph(f"• {obs}", styles['KeyPoint']))
    
    story.append(PageBreak())


def add_pca_section(story, styles):
    """Ajoute la section PCA"""
    story.append(Paragraph("5. Réduction de Dimensionnalité (PCA)", styles['SectionTitle']))
    
    story.append(Paragraph("<b>5.1 Objectif et Méthodologie</b>", styles['SubSectionTitle']))
    pca_text = """
    L'Analyse en Composantes Principales (PCA) a été appliquée pour réduire la dimensionnalité 
    du dataset tout en préservant un maximum de variance. Cette étape est cruciale pour 
    améliorer les performances du clustering.
    """
    story.append(Paragraph(pca_text, styles['BodyTextCustom']))
    
    # Résultats PCA
    pca_results = [
        ["Métrique", "Valeur"],
        ["Features originales", "83"],
        ["Composantes retenues", "36"],
        ["Variance expliquée", "95%"],
        ["Critère de sélection", "Variance cumulative ≥ 95%"],
    ]
    
    pca_table = Table(pca_results, colWidths=[3*inch, 2*inch])
    pca_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(pca_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Image variance PCA
    pca_var_path = os.path.join(REPORTS_DIR, 'pca_variance.png')
    if os.path.exists(pca_var_path):
        story.append(Paragraph("<b>5.2 Variance Expliquée</b>", styles['SubSectionTitle']))
        img = Image(pca_var_path, width=5.5*inch, height=3*inch)
        story.append(img)
        story.append(Paragraph("Figure 5.1: Variance expliquée par les composantes principales", styles['Caption']))
    
    # Image PCA 2D
    pca_2d_path = os.path.join(REPORTS_DIR, 'pca_2d.png')
    if os.path.exists(pca_2d_path):
        story.append(Paragraph("<b>5.3 Projection 2D</b>", styles['SubSectionTitle']))
        img = Image(pca_2d_path, width=5*inch, height=3.5*inch)
        story.append(img)
        story.append(Paragraph("Figure 5.2: Projection des clients sur les deux premières composantes", styles['Caption']))
    
    story.append(PageBreak())


def add_clustering_section(story, styles):
    """Ajoute la section clustering"""
    story.append(Paragraph("6. Clustering (K-Means)", styles['SectionTitle']))
    
    story.append(Paragraph("<b>6.1 Méthodologie</b>", styles['SubSectionTitle']))
    clustering_text = """
    L'algorithme K-Means a été utilisé pour segmenter les clients en groupes homogènes. 
    Le nombre optimal de clusters a été déterminé par la méthode du coude (Elbow) et 
    le score Silhouette.
    """
    story.append(Paragraph(clustering_text, styles['BodyTextCustom']))
    
    # Image optimal clusters
    optimal_path = os.path.join(REPORTS_DIR, 'optimal_clusters.png')
    if os.path.exists(optimal_path):
        img = Image(optimal_path, width=5.5*inch, height=2.8*inch)
        story.append(img)
        story.append(Paragraph("Figure 6.1: Méthode du coude pour déterminer K optimal", styles['Caption']))
    
    story.append(Paragraph("<b>6.2 Résultats du Clustering</b>", styles['SubSectionTitle']))
    
    cluster_results = [
        ["Métrique", "Valeur"],
        ["Nombre de clusters (K)", "2"],
        ["Score Silhouette", "0.16"],
        ["Algorithme", "K-Means"],
        ["Données utilisées", "PCA (36 composantes)"],
    ]
    
    cluster_table = Table(cluster_results, colWidths=[3*inch, 2*inch])
    cluster_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(cluster_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Image distribution clusters
    cluster_dist_path = os.path.join(REPORTS_DIR, 'cluster_distribution_KMeans_K2.png')
    if os.path.exists(cluster_dist_path):
        img = Image(cluster_dist_path, width=5*inch, height=3.5*inch)
        story.append(img)
        story.append(Paragraph("Figure 6.2: Distribution des clients par cluster", styles['Caption']))
    
    story.append(Paragraph("<b>6.3 Interprétation des Clusters</b>", styles['SubSectionTitle']))
    interpretation = """
    Les deux clusters identifiés représentent deux profils de clients distincts:
    """
    story.append(Paragraph(interpretation, styles['BodyTextCustom']))
    
    clusters_desc = [
        "<b>Cluster 0:</b> Clients réguliers avec fréquence d'achat élevée et bon engagement",
        "<b>Cluster 1:</b> Clients occasionnels avec achats moins fréquents mais potentiel de conversion"
    ]
    for c in clusters_desc:
        story.append(Paragraph(f"• {c}", styles['KeyPoint']))
    
    story.append(PageBreak())


def add_classification_section(story, styles):
    """Ajoute la section classification"""
    story.append(Paragraph("7. Classification (Prédiction du Churn)", styles['SectionTitle']))
    
    story.append(Paragraph("<b>7.1 Objectif</b>", styles['SubSectionTitle']))
    class_text = """
    L'objectif est de prédire si un client va quitter l'entreprise (Churn = 1) ou rester fidèle 
    (Churn = 0). Plusieurs algorithmes de classification ont été évalués et comparés.
    """
    story.append(Paragraph(class_text, styles['BodyTextCustom']))
    
    story.append(Paragraph("<b>7.2 Modèles Évalués</b>", styles['SubSectionTitle']))
    
    # Tableau comparatif des modèles
    models_data = [
        ["Modèle", "Accuracy", "Precision", "Recall", "F1-Score"],
        ["Logistic Regression", "100%", "100%", "100%", "100%"],
        ["Random Forest", "100%", "100%", "100%", "100%"],
        ["Gradient Boosting", "100%", "100%", "100%", "100%"],
        ["KNN", "99%", "99%", "99%", "99%"],
        ["Decision Tree", "100%", "100%", "100%", "100%"],
    ]
    
    models_table = Table(models_data, colWidths=[1.8*inch, 1*inch, 1*inch, 0.9*inch, 0.9*inch])
    models_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#d5f5e3')),  # Highlight best
    ]))
    story.append(models_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>7.3 Meilleur Modèle: Logistic Regression</b>", styles['SubSectionTitle']))
    best_text = """
    La Régression Logistique a été sélectionnée comme modèle final pour sa simplicité, 
    son interprétabilité et ses excellentes performances (F1-Score de 100%).
    """
    story.append(Paragraph(best_text, styles['BodyTextCustom']))
    
    # Matrices de confusion
    story.append(Paragraph("<b>7.4 Matrices de Confusion</b>", styles['SubSectionTitle']))
    
    confusion_models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
    for model in confusion_models:
        img_path = os.path.join(REPORTS_DIR, f'confusion_matrix_{model}.png')
        if os.path.exists(img_path):
            img = Image(img_path, width=3.5*inch, height=2.8*inch)
            story.append(img)
            story.append(Paragraph(f"Figure 7.x: Matrice de confusion - {model}", styles['Caption']))
    
    story.append(PageBreak())
    
    # Courbes ROC
    story.append(Paragraph("<b>7.5 Courbes ROC</b>", styles['SubSectionTitle']))
    roc_text = """
    Les courbes ROC (Receiver Operating Characteristic) montrent les performances des 
    classificateurs en termes de taux de vrais positifs vs faux positifs.
    """
    story.append(Paragraph(roc_text, styles['BodyTextCustom']))
    
    for model in confusion_models:
        img_path = os.path.join(REPORTS_DIR, f'roc_curve_{model}.png')
        if os.path.exists(img_path):
            img = Image(img_path, width=3.5*inch, height=2.8*inch)
            story.append(img)
            story.append(Paragraph(f"Figure 7.x: Courbe ROC - {model}", styles['Caption']))
    
    story.append(PageBreak())


def add_regression_section(story, styles):
    """Ajoute la section régression"""
    story.append(Paragraph("8. Régression (Prédiction MonetaryAvg)", styles['SectionTitle']))
    
    story.append(Paragraph("<b>8.1 Objectif</b>", styles['SubSectionTitle']))
    reg_text = """
    L'objectif est de prédire le panier moyen (MonetaryAvg) des clients afin d'estimer 
    leur valeur potentielle et personnaliser les stratégies marketing.
    """
    story.append(Paragraph(reg_text, styles['BodyTextCustom']))
    
    story.append(Paragraph("<b>8.2 Modèles Évalués</b>", styles['SubSectionTitle']))
    
    # Tableau comparatif régression
    reg_data = [
        ["Modèle", "R²", "RMSE", "MAE"],
        ["Gradient Boosting", "0.49", "85.2", "45.3"],
        ["Random Forest", "0.45", "88.7", "48.1"],
        ["Ridge Regression", "0.32", "98.4", "52.6"],
    ]
    
    reg_table = Table(reg_data, colWidths=[2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    reg_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f39c12')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#fef9e7')),  # Highlight best
    ]))
    story.append(reg_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>8.3 Meilleur Modèle: Gradient Boosting</b>", styles['SubSectionTitle']))
    best_reg_text = """
    Le Gradient Boosting Regressor offre les meilleures performances avec un R² de 0.49, 
    expliquant près de 50% de la variance du panier moyen. Ce résultat est acceptable 
    compte tenu de la complexité du comportement d'achat client.
    """
    story.append(Paragraph(best_reg_text, styles['BodyTextCustom']))
    
    story.append(Paragraph("<b>8.4 Visualisations des Performances</b>", styles['SubSectionTitle']))
    
    reg_models = ['Gradient Boosting', 'Random Forest', 'Ridge Regression']
    for model in reg_models:
        img_path = os.path.join(REPORTS_DIR, f'regression_eval_{model}.png')
        if os.path.exists(img_path):
            img = Image(img_path, width=5*inch, height=3*inch)
            story.append(img)
            story.append(Paragraph(f"Figure 8.x: Évaluation régression - {model}", styles['Caption']))
    
    story.append(PageBreak())


def add_webapp_section(story, styles):
    """Ajoute la section application web"""
    story.append(Paragraph("9. Application Web", styles['SectionTitle']))
    
    story.append(Paragraph("<b>9.1 Description</b>", styles['SubSectionTitle']))
    webapp_text = """
    Une application web Flask a été développée pour permettre aux utilisateurs de faire 
    des prédictions en temps réel. L'interface utilise Bootstrap pour un design moderne 
    et responsive.
    """
    story.append(Paragraph(webapp_text, styles['BodyTextCustom']))
    
    story.append(Paragraph("<b>9.2 Fonctionnalités</b>", styles['SubSectionTitle']))
    features = [
        "Saisie des caractéristiques client via formulaire intuitif",
        "Prédiction du segment (Cluster K-Means)",
        "Prédiction de la probabilité de Churn",
        "Estimation du panier moyen (MonetaryAvg)",
        "Affichage des résultats avec indicateurs visuels"
    ]
    for f in features:
        story.append(Paragraph(f"• {f}", styles['KeyPoint']))
    
    story.append(Paragraph("<b>9.3 Architecture Technique</b>", styles['SubSectionTitle']))
    arch_text = """
    <b>Backend:</b> Flask (Python)<br/>
    <b>Frontend:</b> HTML5, CSS3, Bootstrap 5<br/>
    <b>Modèles:</b> Chargés via Joblib depuis le dossier models/<br/>
    <b>API:</b> Endpoint POST /predict pour les prédictions
    """
    story.append(Paragraph(arch_text, styles['BodyTextCustom']))
    
    story.append(PageBreak())


def add_conclusion(story, styles):
    """Ajoute la conclusion"""
    story.append(Paragraph("10. Conclusions et Perspectives", styles['SectionTitle']))
    
    story.append(Paragraph("<b>10.1 Résumé des Résultats</b>", styles['SubSectionTitle']))
    summary = """
    Ce projet a permis d'analyser en profondeur les données clients d'une entreprise 
    de e-commerce et de développer des modèles prédictifs performants:
    """
    story.append(Paragraph(summary, styles['BodyTextCustom']))
    
    results = [
        "<b>Prétraitement:</b> Nettoyage complet avec 84 features finales",
        "<b>PCA:</b> Réduction de 83 à 36 dimensions (95% variance conservée)",
        "<b>Clustering:</b> Segmentation en 2 groupes distincts de clients",
        "<b>Classification:</b> Prédiction du Churn avec 100% de précision",
        "<b>Régression:</b> Estimation du panier moyen avec R² = 0.49"
    ]
    for r in results:
        story.append(Paragraph(f"• {r}", styles['KeyPoint']))
    
    story.append(Paragraph("<b>10.2 Points Forts</b>", styles['SubSectionTitle']))
    strengths = [
        "Pipeline ML complet et reproductible",
        "Excellentes performances en classification",
        "Application web fonctionnelle pour l'inférence",
        "Code modulaire et bien documenté"
    ]
    for s in strengths:
        story.append(Paragraph(f"• {s}", styles['KeyPoint']))
    
    story.append(Paragraph("<b>10.3 Perspectives d'Amélioration</b>", styles['SubSectionTitle']))
    improvements = [
        "Tester des algorithmes de deep learning (Neural Networks)",
        "Améliorer le modèle de régression (R² > 0.5)",
        "Ajouter des features temporelles plus sophistiquées",
        "Déployer l'application sur un serveur cloud (AWS, GCP)",
        "Implémenter un système de monitoring des prédictions"
    ]
    for i in improvements:
        story.append(Paragraph(f"• {i}", styles['KeyPoint']))
    
    story.append(Spacer(1, 0.5*inch))
    
    # Note finale
    final_note = """
    <b>Note:</b> Ce projet démontre l'application pratique des techniques de Machine Learning 
    pour résoudre des problématiques business réelles dans le domaine du retail et de 
    l'e-commerce.
    """
    story.append(Paragraph(final_note, styles['BodyTextCustom']))
    
    story.append(Spacer(1, 1*inch))
    
    # Signature
    story.append(Paragraph(f"<i>Rapport réalisé par {AUTHOR}</i>", styles['AuthorInfo']))
    story.append(Paragraph(f"<i>{MODULE} - {ACADEMIC_YEAR}</i>", styles['AuthorInfo']))


def generate_pdf():
    """Génère le rapport PDF complet"""
    print("="*60)
    print("GÉNÉRATION DU RAPPORT PDF")
    print("="*60)
    print(f"\nAuteur: {AUTHOR}")
    print(f"Module: {MODULE}")
    print(f"Année: {ACADEMIC_YEAR}")
    print(f"\nFichier de sortie: {OUTPUT_PDF}")
    
    # Créer le document
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
        title=PROJECT_TITLE,
        author=AUTHOR
    )
    
    # Obtenir les styles
    styles = get_styles()
    
    # Construire le contenu
    story = []
    
    print("\nGénération des sections...")
    
    print("  [1/10] Page de titre...")
    add_title_page(story, styles)
    
    print("  [2/10] Table des matières...")
    add_table_of_contents(story, styles)
    
    print("  [3/10] Introduction...")
    add_introduction(story, styles)
    
    print("  [4/10] Description du dataset...")
    add_dataset_description(story, styles)
    
    print("  [5/10] Prétraitement...")
    add_preprocessing_section(story, styles)
    
    print("  [6/10] Analyse exploratoire...")
    add_eda_section(story, styles)
    
    print("  [7/10] PCA...")
    add_pca_section(story, styles)
    
    print("  [8/10] Clustering...")
    add_clustering_section(story, styles)
    
    print("  [9/10] Classification...")
    add_classification_section(story, styles)
    
    print("  [10/10] Régression...")
    add_regression_section(story, styles)
    
    print("  [11/10] Application web...")
    add_webapp_section(story, styles)
    
    print("  [12/10] Conclusion...")
    add_conclusion(story, styles)
    
    # Générer le PDF
    print("\nCompilation du PDF...")
    doc.build(story)
    
    print("\n" + "="*60)
    print("RAPPORT PDF GÉNÉRÉ AVEC SUCCÈS!")
    print("="*60)
    print(f"\nFichier: {OUTPUT_PDF}")
    
    # Vérifier la taille du fichier
    if os.path.exists(OUTPUT_PDF):
        size_mb = os.path.getsize(OUTPUT_PDF) / (1024 * 1024)
        print(f"Taille: {size_mb:.2f} MB")


if __name__ == "__main__":
    generate_pdf()
