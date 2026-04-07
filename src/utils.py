"""
utils.py - Fonctions utilitaires pour le projet ML Retail
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score
)
import joblib
import os
from datetime import datetime

# Paths configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
DATA_TRAIN_TEST = os.path.join(BASE_DIR, 'data', 'train_test')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')


def load_raw_data(filename='retail_customers_COMPLETE_CATEGORICAL.csv'):
    """Charge les donnees brutes depuis data/raw/"""
    filepath = os.path.join(DATA_RAW, filename)
    df = pd.read_csv(filepath)
    print(f"Donnees chargees: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def save_processed_data(df, filename):
    """Sauvegarde les donnees traitees dans data/processed/"""
    filepath = os.path.join(DATA_PROCESSED, filename)
    df.to_csv(filepath, index=False)
    print(f"Donnees sauvegardees: {filepath}")


def save_train_test_data(X_train, X_test, y_train, y_test, prefix=''):
    """Sauvegarde les donnees train/test"""
    X_train.to_csv(os.path.join(DATA_TRAIN_TEST, f'{prefix}X_train.csv'), index=False)
    X_test.to_csv(os.path.join(DATA_TRAIN_TEST, f'{prefix}X_test.csv'), index=False)
    y_train.to_csv(os.path.join(DATA_TRAIN_TEST, f'{prefix}y_train.csv'), index=False)
    y_test.to_csv(os.path.join(DATA_TRAIN_TEST, f'{prefix}y_test.csv'), index=False)
    print(f"Donnees train/test sauvegardees avec prefix '{prefix}'")


def load_train_test_data(prefix=''):
    """Charge les donnees train/test"""
    X_train = pd.read_csv(os.path.join(DATA_TRAIN_TEST, f'{prefix}X_train.csv'))
    X_test = pd.read_csv(os.path.join(DATA_TRAIN_TEST, f'{prefix}X_test.csv'))
    y_train = pd.read_csv(os.path.join(DATA_TRAIN_TEST, f'{prefix}y_train.csv'))
    y_test = pd.read_csv(os.path.join(DATA_TRAIN_TEST, f'{prefix}y_test.csv'))
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()


def save_model(model, filename):
    """Sauvegarde un modele dans models/"""
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, filepath)
    print(f"Modele sauvegarde: {filepath}")


def load_model(filename):
    """Charge un modele depuis models/"""
    filepath = os.path.join(MODELS_DIR, filename)
    model = joblib.load(filepath)
    print(f"Modele charge: {filepath}")
    return model


# ============== EXPLORATION DES DONNEES ==============

def explore_data(df):
    """Exploration complete des donnees"""
    print("=" * 60)
    print("EXPLORATION DES DONNEES")
    print("=" * 60)
    
    print(f"\nDimensions: {df.shape[0]} lignes x {df.shape[1]} colonnes")
    
    print("\n--- Types de donnees ---")
    print(df.dtypes.value_counts())
    
    print("\n--- Valeurs manquantes ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing'] > 0].sort_values('Percentage', ascending=False))
    
    print("\n--- Statistiques descriptives (numeriques) ---")
    print(df.describe())
    
    return missing_df


def plot_missing_values(df, figsize=(12, 6)):
    """Visualise les valeurs manquantes"""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("Aucune valeur manquante!")
        return
    
    plt.figure(figsize=figsize)
    missing_pct = (missing / len(df) * 100)
    missing_pct.plot(kind='bar', color='coral')
    plt.title('Pourcentage de valeurs manquantes par colonne')
    plt.ylabel('Pourcentage (%)')
    plt.xlabel('Colonnes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'missing_values.png'), dpi=150)
    plt.show()


def plot_correlation_matrix(df, figsize=(16, 14), threshold=0.8):
    """Affiche la matrice de correlation avec heatmap"""
    # Selectionner uniquement les colonnes numeriques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1)
    plt.title('Matrice de Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'correlation_matrix.png'), dpi=150)
    plt.show()
    
    # Identifier les paires fortement correlees
    print(f"\nPaires avec |correlation| > {threshold}:")
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr:
        high_corr_df = pd.DataFrame(high_corr).sort_values('Correlation', ascending=False)
        print(high_corr_df)
        return high_corr_df
    return None


def plot_distribution(df, column, figsize=(10, 4)):
    """Visualise la distribution d'une variable"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogramme
    df[column].hist(bins=50, ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title(f'Distribution de {column}')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequence')
    
    # Boxplot
    df.boxplot(column=column, ax=axes[1])
    axes[1].set_title(f'Boxplot de {column}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f'distribution_{column}.png'), dpi=150)
    plt.show()


# ============== PREPROCESSING ==============

def parse_registration_date(df, column='RegistrationDate'):
    """Parse la colonne RegistrationDate avec differents formats"""
    def parse_date(date_str):
        if pd.isna(date_str):
            return pd.NaT
        
        date_str = str(date_str).strip()
        formats = [
            '%d/%m/%y', '%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', 
            '%d-%m-%Y', '%Y/%m/%d', '%m/%d/%y'
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        
        try:
            return pd.to_datetime(date_str, dayfirst=True)
        except:
            return pd.NaT
    
    df[column] = df[column].apply(parse_date)
    
    # Extraction de features
    df['RegYear'] = df[column].dt.year
    df['RegMonth'] = df[column].dt.month
    df['RegDay'] = df[column].dt.day
    df['RegWeekday'] = df[column].dt.weekday
    
    return df


def parse_ip_address(df, column='LastLoginIP'):
    """Extrait des features depuis l'adresse IP"""
    def get_ip_class(ip):
        if pd.isna(ip):
            return 'Unknown'
        try:
            first_octet = int(str(ip).split('.')[0])
            if first_octet < 128:
                return 'A'
            elif first_octet < 192:
                return 'B'
            elif first_octet < 224:
                return 'C'
            else:
                return 'D/E'
        except:
            return 'Unknown'
    
    def is_private_ip(ip):
        if pd.isna(ip):
            return -1
        try:
            parts = str(ip).split('.')
            first = int(parts[0])
            second = int(parts[1])
            
            if first == 10:
                return 1
            if first == 172 and 16 <= second <= 31:
                return 1
            if first == 192 and second == 168:
                return 1
            return 0
        except:
            return -1
    
    df['IPClass'] = df[column].apply(get_ip_class)
    df['IsPrivateIP'] = df[column].apply(is_private_ip)
    
    return df


def handle_outliers(df, column, method='iqr', factor=1.5):
    """Traite les valeurs aberrantes"""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        
        outliers_count = ((df[column] < lower) | (df[column] > upper)).sum()
        print(f"{column}: {outliers_count} outliers detectes (IQR method)")
        
        # Clip les valeurs
        df[column] = df[column].clip(lower=lower, upper=upper)
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers_count = (z_scores > 3).sum()
        print(f"{column}: {outliers_count} outliers detectes (Z-score method)")
        
        mean = df[column].mean()
        std = df[column].std()
        df[column] = df[column].clip(lower=mean - 3*std, upper=mean + 3*std)
    
    return df


def handle_special_values(df):
    """Traite les valeurs speciales (-1, 999, 99) dans SupportTickets et Satisfaction"""
    # SupportTickets: -1 et 999 sont des valeurs speciales
    df['SupportTickets_Missing'] = (df['SupportTicketsCount'].isin([-1, 999])).astype(int)
    df.loc[df['SupportTicketsCount'].isin([-1, 999]), 'SupportTicketsCount'] = np.nan
    
    # Satisfaction: -1 et 99 sont des valeurs speciales
    df['Satisfaction_Missing'] = (df['SatisfactionScore'].isin([-1, 99])).astype(int)
    df.loc[df['SatisfactionScore'].isin([-1, 99]), 'SatisfactionScore'] = np.nan
    
    return df


def impute_missing_values(df, numeric_strategy='median', categorical_strategy='most_frequent'):
    """Impute les valeurs manquantes"""
    # Colonnes numeriques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Colonnes categorielles
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Imputation numerique
    if numeric_cols:
        imputer_num = SimpleImputer(strategy=numeric_strategy)
        df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
        print(f"Imputation numerique ({numeric_strategy}): {len(numeric_cols)} colonnes")
    
    # Imputation categorielle
    if categorical_cols:
        imputer_cat = SimpleImputer(strategy=categorical_strategy)
        df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
        print(f"Imputation categorielle ({categorical_strategy}): {len(categorical_cols)} colonnes")
    
    return df


def encode_ordinal_features(df):
    """Encode les features ordinales"""
    ordinal_mappings = {
        'AgeCategory': {'18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55-64': 5, '65+': 6, 'Inconnu': 0},
        'SpendingCategory': {'Low': 1, 'Medium': 2, 'High': 3, 'VIP': 4},
        'LoyaltyLevel': {'Nouveau': 1, 'Jeune': 2, 'Etabli': 3, 'Ancien': 4, 'Inconnu': 0, 'Établi': 3},
        'ChurnRiskCategory': {'Faible': 1, 'Moyen': 2, 'Eleve': 3, 'Critique': 4, 'Élevé': 3},
        'BasketSizeCategory': {'Petit': 1, 'Moyen': 2, 'Grand': 3, 'Inconnu': 0},
        'PreferredTimeOfDay': {'Matin': 1, 'Midi': 2, 'Apres-midi': 3, 'Soir': 4, 'Nuit': 5, 'Après-midi': 3},
        'RFMSegment': {'Dormants': 1, 'Potentiels': 2, 'Fideles': 3, 'Champions': 4, 'Fidèles': 3}
    }
    
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col + '_encoded'] = df[col].map(mapping).fillna(0).astype(int)
            print(f"Encoded ordinal: {col}")
    
    return df


def encode_onehot_features(df, columns):
    """Encode les features nominales avec One-Hot Encoding"""
    for col in columns:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            print(f"One-Hot encoded: {col} -> {len(dummies.columns)} colonnes")
    
    return df


def create_feature_engineering(df):
    """Cree de nouvelles features"""
    # Ratio depenses/recency
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    
    # Panier moyen
    df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)
    
    # Ratio anciennete vs activite recente
    df['TenureRecencyRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)
    
    # Score d'engagement (combine plusieurs metriques)
    df['EngagementScore'] = (
        df['Frequency'] * 0.3 + 
        (1 - df['Recency'] / df['Recency'].max()) * 0.3 +
        df['UniqueProducts'] / df['UniqueProducts'].max() * 0.2 +
        df['TotalTransactions'] / df['TotalTransactions'].max() * 0.2
    )
    
    # Valeur vie client estimee (simplifiee)
    df['CLV_Estimate'] = df['MonetaryAvg'] * df['Frequency'] * (1 - df['ReturnRatio'])
    
    print("Feature engineering: 5 nouvelles features creees")
    return df


# ============== ACP ==============

def apply_pca(X, n_components=0.95, random_state=42):
    """Applique l'ACP pour reduire la dimension"""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    print(f"\nACP - Reduction de {X.shape[1]} features a {X_pca.shape[1]} composantes")
    print(f"Variance expliquee totale: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Plot variance expliquee
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_, alpha=0.7, label='Individuelle')
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), 'r-o', label='Cumulee')
    plt.xlabel('Composante Principale')
    plt.ylabel('Variance Expliquee')
    plt.title('Variance expliquee par composante')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), 'b-o')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.xlabel('Nombre de composantes')
    plt.ylabel('Variance cumulee')
    plt.title('Variance cumulee')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'pca_variance.png'), dpi=150)
    plt.show()
    
    return X_pca, pca


def plot_pca_2d(X_pca, labels=None, title='Projection ACP 2D'):
    """Visualise les 2 premieres composantes principales"""
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster/Label')
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    
    plt.xlabel('Composante Principale 1')
    plt.ylabel('Composante Principale 2')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'pca_2d.png'), dpi=150)
    plt.show()


# ============== EVALUATION DES MODELES ==============

def evaluate_classification(y_true, y_pred, y_prob=None, model_name='Model'):
    """Evalue un modele de classification"""
    print(f"\n{'='*60}")
    print(f"EVALUATION - {model_name}")
    print(f"{'='*60}")
    
    print("\nRapport de classification:")
    print(classification_report(y_true, y_pred))
    
    # Metriques
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Matrice de confusion
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.ylabel('Vrai')
    plt.xlabel('Predit')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f'confusion_matrix_{model_name}.png'), dpi=150)
    plt.show()
    
    # Courbe ROC si probabilites disponibles
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Random')
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title(f'Courbe ROC - {model_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, f'roc_curve_{model_name}.png'), dpi=150)
        plt.show()
        
        print(f"AUC-ROC: {auc:.4f}")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_regression(y_true, y_pred, model_name='Model'):
    """Evalue un modele de regression"""
    print(f"\n{'='*60}")
    print(f"EVALUATION REGRESSION - {model_name}")
    print(f"{'='*60}")
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nMSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Valeurs Reelles')
    plt.ylabel('Predictions')
    plt.title('Predictions vs Valeurs Reelles')
    
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.hist(residuals, bins=50, edgecolor='black')
    plt.xlabel('Residus')
    plt.ylabel('Frequence')
    plt.title('Distribution des Residus')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f'regression_eval_{model_name}.png'), dpi=150)
    plt.show()
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def evaluate_clustering(X, labels, model_name='Clustering'):
    """Evalue un modele de clustering"""
    print(f"\n{'='*60}")
    print(f"EVALUATION CLUSTERING - {model_name}")
    print(f"{'='*60}")
    
    # Silhouette score
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(X, labels)
        print(f"\nSilhouette Score: {silhouette:.4f}")
    else:
        silhouette = 0
        print("Un seul cluster - Silhouette non calculable")
    
    # Distribution des clusters
    unique, counts = np.unique(labels, return_counts=True)
    print("\nDistribution des clusters:")
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} ({c/len(labels)*100:.1f}%)")
    
    # Visualisation
    plt.figure(figsize=(8, 6))
    plt.bar(unique, counts, color='steelblue', edgecolor='black')
    plt.xlabel('Cluster')
    plt.ylabel('Nombre de clients')
    plt.title(f'Distribution des Clusters - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f'cluster_distribution_{model_name}.png'), dpi=150)
    plt.show()
    
    return {'silhouette': silhouette, 'n_clusters': len(unique)}


if __name__ == "__main__":
    # Test des fonctions
    print("Test du module utils.py")
    df = load_raw_data()
    explore_data(df)
