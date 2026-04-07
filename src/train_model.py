"""
train_model.py - Script d'entrainement des modeles ML
Inclut: Clustering (K-Means), Classification (Churn), Regression (MonetaryTotal)
"""

import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Ajouter le repertoire src au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_train_test_data, save_model, load_model,
    apply_pca, plot_pca_2d,
    evaluate_classification, evaluate_regression, evaluate_clustering,
    MODELS_DIR, REPORTS_DIR, DATA_PROCESSED
)

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import silhouette_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


# ============== CLUSTERING ==============

def find_optimal_clusters(X, max_k=10):
    """
    Trouve le nombre optimal de clusters avec la methode du coude et silhouette
    """
    print("\n" + "="*60)
    print("RECHERCHE DU NOMBRE OPTIMAL DE CLUSTERS")
    print("="*60)
    
    inertias = []
    silhouettes = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, kmeans.labels_))
        print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.4f}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Methode du coude
    axes[0].plot(K_range, inertias, 'bo-')
    axes[0].set_xlabel('Nombre de clusters (K)')
    axes[0].set_ylabel('Inertie')
    axes[0].set_title('Methode du Coude')
    
    # Silhouette Score
    axes[1].plot(K_range, silhouettes, 'go-')
    axes[1].set_xlabel('Nombre de clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Score Silhouette par K')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'optimal_clusters.png'), dpi=150)
    plt.show()
    
    # Meilleur K selon silhouette
    best_k = K_range[np.argmax(silhouettes)]
    print(f"\nMeilleur K (silhouette): {best_k}")
    
    return best_k


def train_clustering(X, n_clusters=4):
    """
    Entraine un modele K-Means pour la segmentation client
    """
    print("\n" + "="*60)
    print(f"ENTRAINEMENT K-MEANS (K={n_clusters})")
    print("="*60)
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Evaluation
    evaluate_clustering(X, clusters, f'KMeans_K{n_clusters}')
    
    # Sauvegarde
    save_model(kmeans, 'kmeans_model.joblib')
    
    return kmeans, clusters


def analyze_clusters(X_original, clusters, feature_names):
    """
    Analyse les caracteristiques de chaque cluster
    """
    print("\n" + "="*60)
    print("ANALYSE DES CLUSTERS")
    print("="*60)
    
    # Creer un DataFrame avec les clusters
    df_clusters = pd.DataFrame(X_original, columns=feature_names)
    df_clusters['Cluster'] = clusters
    
    # Statistiques par cluster
    cluster_stats = df_clusters.groupby('Cluster').mean()
    
    # Afficher les caracteristiques principales
    print("\nMoyenne des features par cluster:")
    
    # Selectionner quelques features importantes
    important_features = ['Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg', 
                         'TotalQuantity', 'UniqueProducts', 'CustomerTenureDays',
                         'ReturnRatio', 'EngagementScore']
    
    available_features = [f for f in important_features if f in cluster_stats.columns]
    
    if available_features:
        print(cluster_stats[available_features].round(2))
    
    # Taille des clusters
    cluster_sizes = df_clusters['Cluster'].value_counts().sort_index()
    print("\nTaille des clusters:")
    for idx, size in cluster_sizes.items():
        print(f"  Cluster {idx}: {size} clients ({size/len(df_clusters)*100:.1f}%)")
    
    return cluster_stats


# ============== CLASSIFICATION (CHURN) ==============

def train_classification_models(X_train, X_test, y_train, y_test, use_smote=True):
    """
    Entraine plusieurs modeles de classification pour predire le churn
    """
    print("\n" + "="*60)
    print("ENTRAINEMENT DES MODELES DE CLASSIFICATION (CHURN)")
    print("="*60)
    
    # Gestion du desequilibre des classes avec SMOTE
    if use_smote:
        print("\nApplication de SMOTE pour equilibrer les classes...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"  Avant SMOTE: {len(y_train)} samples")
        print(f"  Apres SMOTE: {len(y_train_balanced)} samples")
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Dictionnaire des modeles
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    best_model = None
    best_f1 = 0
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Entrainement
        model.fit(X_train_balanced, y_train_balanced)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Probabilites pour ROC (si disponible)
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except:
            y_prob = None
        
        # Evaluation
        metrics = evaluate_classification(y_test, y_pred, y_prob, name)
        results[name] = metrics
        
        # Garder le meilleur modele
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = (name, model)
    
    # Afficher le resume
    print("\n" + "="*60)
    print("RESUME DES PERFORMANCES")
    print("="*60)
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    print(f"\nMeilleur modele: {best_model[0]} (F1={best_f1:.4f})")
    
    # Sauvegarder le meilleur modele
    save_model(best_model[1], 'best_classifier_churn.joblib')
    
    return best_model, results


def tune_random_forest(X_train, y_train):
    """
    Optimisation des hyperparametres du Random Forest avec GridSearchCV
    """
    print("\n" + "="*60)
    print("OPTIMISATION HYPERPARAMETRES - RANDOM FOREST")
    print("="*60)
    
    # Grille de parametres
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    # GridSearchCV
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=5, 
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    grid_search.fit(X_balanced, y_balanced)
    
    print(f"\nMeilleurs parametres: {grid_search.best_params_}")
    print(f"Meilleur score F1 (CV): {grid_search.best_score_:.4f}")
    
    # Sauvegarder le meilleur modele
    save_model(grid_search.best_estimator_, 'random_forest_tuned.joblib')
    
    return grid_search.best_estimator_


def get_feature_importance(model, feature_names, top_n=20):
    """
    Affiche l'importance des features pour les modeles arborescents
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Features Importance')
        plt.barh(range(top_n), importances[indices][::-1], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'feature_importance.png'), dpi=150)
        plt.show()
        
        print(f"\nTop {top_n} features:")
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


# ============== REGRESSION ==============

def train_regression_models(X_train, X_test, y_train, y_test, target_name='MonetaryTotal'):
    """
    Entraine des modeles de regression pour predire les depenses
    """
    print("\n" + "="*60)
    print(f"ENTRAINEMENT DES MODELES DE REGRESSION ({target_name})")
    print("="*60)
    
    # Modeles
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model = None
    best_r2 = -float('inf')
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Entrainement
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Evaluation
        metrics = evaluate_regression(y_test, y_pred, name)
        results[name] = metrics
        
        # Garder le meilleur modele
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model = (name, model)
    
    # Resume
    print("\n" + "="*60)
    print("RESUME DES PERFORMANCES")
    print("="*60)
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    print(f"\nMeilleur modele: {best_model[0]} (R2={best_r2:.4f})")
    
    # Sauvegarder
    save_model(best_model[1], 'best_regressor.joblib')
    
    return best_model, results


# ============== MAIN ==============

def main():
    """
    Pipeline complet d'entrainement des modeles
    """
    print("\n" + "="*60)
    print("CHARGEMENT DES DONNEES")
    print("="*60)
    
    # Charger les donnees preprocessees
    X_train, X_test, y_train, y_test = load_train_test_data()
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train distribution:\n{pd.Series(y_train).value_counts()}")
    
    feature_names = X_train.columns.tolist()
    
    # Convertir en array numpy
    X_train_arr = X_train.values
    X_test_arr = X_test.values
    
    # ========== 1. ACP ==========
    print("\n" + "="*60)
    print("1. ANALYSE EN COMPOSANTES PRINCIPALES (ACP)")
    print("="*60)
    
    # Combiner train et test pour ACP (juste pour visualisation)
    X_combined = np.vstack([X_train_arr, X_test_arr])
    X_pca, pca = apply_pca(X_combined, n_components=0.95)
    
    # Visualisation 2D
    plot_pca_2d(X_pca[:len(X_train_arr)], y_train, 'ACP - Colore par Churn')
    
    # Sauvegarder le PCA
    save_model(pca, 'pca_model.joblib')
    
    # ========== 2. CLUSTERING ==========
    print("\n" + "="*60)
    print("2. CLUSTERING (SEGMENTATION CLIENT)")
    print("="*60)
    
    # Trouver le nombre optimal de clusters
    optimal_k = find_optimal_clusters(X_train_arr, max_k=8)
    
    # Entrainer K-Means
    kmeans, clusters = train_clustering(X_train_arr, n_clusters=optimal_k)
    
    # Analyser les clusters
    cluster_stats = analyze_clusters(X_train_arr, clusters, feature_names)
    
    # Visualiser les clusters avec ACP
    X_train_pca = pca.transform(X_train_arr)
    plot_pca_2d(X_train_pca, clusters, f'Clusters K-Means (K={optimal_k})')
    
    # ========== 3. CLASSIFICATION (CHURN) ==========
    print("\n" + "="*60)
    print("3. CLASSIFICATION - PREDICTION DU CHURN")
    print("="*60)
    
    best_classifier, clf_results = train_classification_models(
        X_train_arr, X_test_arr, y_train, y_test, use_smote=True
    )
    
    # Feature importance pour le meilleur modele
    if hasattr(best_classifier[1], 'feature_importances_'):
        get_feature_importance(best_classifier[1], feature_names)
    
    # Optionnel: Optimisation hyperparametres
    # rf_tuned = tune_random_forest(X_train_arr, y_train)
    
    # ========== 4. REGRESSION ==========
    print("\n" + "="*60)
    print("4. REGRESSION - PREDICTION DES DEPENSES")
    print("="*60)
    
    # Charger les donnees pour la regression
    # On utilise MonetaryAvg comme target au lieu de MonetaryTotal
    # car MonetaryTotal est deja utilise comme feature
    df_processed = pd.read_csv(os.path.join(DATA_PROCESSED, 'retail_customers_processed.csv'))
    
    # Verifier si on peut faire de la regression
    if 'MonetaryAvg' in df_processed.columns:
        y_reg = df_processed['MonetaryAvg']
        X_reg = df_processed.drop(['MonetaryAvg', 'Churn'], axis=1)
        
        # Split
        from sklearn.model_selection import train_test_split
        X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        # Entrainer les modeles de regression
        best_regressor, reg_results = train_regression_models(
            X_reg_train.values, X_reg_test.values, 
            y_reg_train.values, y_reg_test.values,
            target_name='MonetaryAvg'
        )
    else:
        print("MonetaryAvg non disponible pour la regression")
    
    # ========== RESUME FINAL ==========
    print("\n" + "="*60)
    print("RESUME FINAL")
    print("="*60)
    print("\nModeles sauvegardes:")
    print("  - models/pca_model.joblib")
    print("  - models/kmeans_model.joblib")
    print("  - models/best_classifier_churn.joblib")
    print("  - models/best_regressor.joblib")
    
    print("\nRapports generes dans reports/")
    
    return {
        'pca': pca,
        'kmeans': kmeans,
        'classifier': best_classifier,
        'regressor': best_regressor if 'best_regressor' in dir() else None
    }


if __name__ == "__main__":
    models = main()
