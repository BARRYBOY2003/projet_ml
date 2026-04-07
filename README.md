# Projet ML Retail - Analyse Comportementale Clientele

## Description
Ce projet de Machine Learning vise a analyser le comportement des clients d'une entreprise e-commerce de cadeaux. L'objectif est de:
- Personnaliser les strategies marketing
- Reduire le taux de depart des clients (churn)
- Optimiser le chiffre d'affaires

Le projet utilise une base de donnees de 52 features issues de transactions reelles pour appliquer:
- **Clustering** (segmentation client)
- **Classification** (prediction du churn)
- **Regression** (prediction des depenses)
- **ACP** (reduction de dimension)

## Structure du Projet

```
projet_ml_retail/
|-- data/                    # Base de donnees
|   |-- raw/                 # Donnees brutes originales
|   |-- processed/           # Donnees nettoyees
|   \-- train_test/          # Donnees splittees (train/test)
|-- notebooks/               # Notebooks Jupyter (prototypage)
|-- src/                     # Scripts Python (production)
|   |-- preprocessing.py     # Nettoyage et preparation des donnees
|   |-- train_model.py       # Entrainement des modeles
|   |-- predict.py           # Predictions
|   \-- utils.py             # Fonctions utilitaires
|-- models/                  # Modeles sauvegardes (.pkl, .joblib)
|-- app/                     # Application web (Flask)
|-- reports/                 # Rapports et visualisations
|-- requirements.txt         # Dependances
|-- README.md                # Documentation
\-- .gitignore
```

## Installation

### 1. Creer l'environnement virtuel
```bash
# Creation
python -m venv venv

# Activation (Windows)
venv\Scripts\activate

# Activation (Linux/Mac)
source venv/bin/activate
```

### 2. Installer les dependances
```bash
pip install -r requirements.txt
```

## Utilisation

### Preprocessing des donnees
```bash
python src/preprocessing.py
```
Ce script:
- Nettoie les donnees (valeurs manquantes, aberrantes)
- Encode les variables categorielles
- Normalise les features numeriques
- Sauvegarde les donnees traitees

### Entrainement des modeles
```bash
python src/train_model.py
```
Ce script entraine:
- K-Means pour la segmentation client
- Random Forest pour la prediction du churn
- Modele de regression pour les depenses

### Predictions
```bash
python src/predict.py
```

### Application Web Flask
```bash
cd app
python app.py
```
Ouvrir http://localhost:5000 dans le navigateur.

## Features du Dataset

Le dataset contient 52 features reparties en:
- **Features numeriques (1-34)**: RFM metrics, comportement d'achat, statistiques
- **Features categorielles (35-52)**: Segments, categories, preferences

### Problemes de qualite resolus:
- Valeurs manquantes (Age: 30%, SupportTickets: 5-8%)
- Valeurs aberrantes (SupportTickets, Satisfaction)
- Formats inconsistants (RegistrationDate)
- Features inutiles (NewsletterSubscribed)
- Desequilibre de classes (Churn)

## Technologies Utilisees
- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- Flask
- Matplotlib, Seaborn

## Auteur
Projet realise dans le cadre du Module Machine Learning - GI2

## Licence
Projet pedagogique - Annee Universitaire 2025-2026
