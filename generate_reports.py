"""
generate_reports.py - Script pour generer les rapports et visualisations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Ajouter le repertoire src au path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.utils import DATA_RAW, DATA_PROCESSED, REPORTS_DIR, load_raw_data
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def generate_missing_values_report(df):
    """Genere un rapport sur les valeurs manquantes"""
    print("Generating missing values report...")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_cols = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing_cols) > 0:
        plt.figure(figsize=(12, 6))
        missing_pct_plot = (missing_cols / len(df) * 100)
        colors = ['coral' if pct > 10 else 'steelblue' for pct in missing_pct_plot]
        
        plt.bar(range(len(missing_cols)), missing_pct_plot.values, color=colors)
        plt.xticks(range(len(missing_cols)), missing_cols.index, rotation=45, ha='right')
        plt.xlabel('Colonnes')
        plt.ylabel('Pourcentage manquant (%)')
        plt.title('Valeurs Manquantes par Colonne')
        plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Seuil 10%')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'missing_values.png'), dpi=150)
        plt.close()
        print(f"  Saved: {REPORTS_DIR}/missing_values.png")
    else:
        print("  No missing values found")


def generate_target_distribution(df, target='Churn'):
    """Genere la distribution de la variable cible"""
    print("Generating target distribution report...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot
    churn_counts = df[target].value_counts()
    colors = ['steelblue', 'coral']
    axes[0].bar(['Fidele (0)', 'Parti (1)'], churn_counts.values, color=colors)
    axes[0].set_ylabel('Nombre de clients')
    axes[0].set_title('Distribution du Churn')
    
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie(churn_counts.values, labels=['Fidele (0)', 'Parti (1)'],
                autopct='%1.1f%%', colors=colors, explode=[0, 0.05])
    axes[1].set_title('Proportion du Churn')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'churn_distribution.png'), dpi=150)
    plt.close()
    print(f"  Saved: {REPORTS_DIR}/churn_distribution.png")


def generate_rfm_analysis(df):
    """Genere l'analyse RFM"""
    print("Generating RFM analysis report...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Recency vs Churn
    df.boxplot(column='Recency', by='Churn', ax=axes[0, 0])
    axes[0, 0].set_title('Recency par Churn')
    axes[0, 0].set_xlabel('Churn')
    axes[0, 0].set_ylabel('Recency (jours)')
    
    # Frequency vs Churn
    df.boxplot(column='Frequency', by='Churn', ax=axes[0, 1])
    axes[0, 1].set_title('Frequency par Churn')
    axes[0, 1].set_xlabel('Churn')
    axes[0, 1].set_ylabel('Frequency')
    
    # MonetaryTotal vs Churn
    df.boxplot(column='MonetaryTotal', by='Churn', ax=axes[1, 0])
    axes[1, 0].set_title('MonetaryTotal par Churn')
    axes[1, 0].set_xlabel('Churn')
    axes[1, 0].set_ylabel('MonetaryTotal (£)')
    
    # RFM Segment distribution
    if 'RFMSegment' in df.columns:
        segment_counts = df['RFMSegment'].value_counts()
        segment_counts.plot(kind='bar', ax=axes[1, 1], color='steelblue')
        axes[1, 1].set_title('Distribution des Segments RFM')
        axes[1, 1].set_xlabel('Segment')
        axes[1, 1].set_ylabel('Nombre de clients')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Analyse RFM', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'rfm_analysis.png'), dpi=150)
    plt.close()
    print(f"  Saved: {REPORTS_DIR}/rfm_analysis.png")


def generate_correlation_heatmap(df):
    """Genere la matrice de correlation"""
    print("Generating correlation heatmap...")
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True)
    plt.title('Matrice de Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'correlation_matrix.png'), dpi=150)
    plt.close()
    print(f"  Saved: {REPORTS_DIR}/correlation_matrix.png")
    
    # Top correlations with Churn
    if 'Churn' in corr_matrix.columns:
        churn_corr = corr_matrix['Churn'].abs().sort_values(ascending=False)[1:11]
        
        plt.figure(figsize=(10, 6))
        churn_corr.plot(kind='barh', color='steelblue')
        plt.xlabel('Correlation absolue avec Churn')
        plt.title('Top 10 Features Correlees avec Churn')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'churn_correlations.png'), dpi=150)
        plt.close()
        print(f"  Saved: {REPORTS_DIR}/churn_correlations.png")


def generate_categorical_analysis(df):
    """Genere l'analyse des variables categorielles"""
    print("Generating categorical analysis report...")
    
    categorical_cols = ['RFMSegment', 'SpendingCategory', 'CustomerType', 
                       'ChurnRiskCategory', 'Region']
    available_cols = [c for c in categorical_cols if c in df.columns]
    
    if not available_cols:
        print("  No categorical columns found")
        return
    
    n_cols = min(len(available_cols), 6)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, col in enumerate(available_cols[:6]):
        # Churn rate by category
        churn_rate = df.groupby(col)['Churn'].mean().sort_values(ascending=False)
        
        colors = ['coral' if rate > 0.3 else 'steelblue' for rate in churn_rate]
        churn_rate.plot(kind='bar', ax=axes[i], color=colors)
        axes[i].set_title(f'Taux de Churn par {col}')
        axes[i].set_ylabel('Taux de Churn')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].axhline(y=df['Churn'].mean(), color='red', linestyle='--', 
                        alpha=0.7, label='Moyenne globale')
    
    # Remove empty subplots
    for j in range(len(available_cols), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'categorical_analysis.png'), dpi=150)
    plt.close()
    print(f"  Saved: {REPORTS_DIR}/categorical_analysis.png")


def generate_distribution_plots(df):
    """Genere les distributions des variables numeriques"""
    print("Generating distribution plots...")
    
    numeric_features = ['Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg',
                       'TotalQuantity', 'CustomerTenureDays', 'UniqueProducts', 'Age']
    available_features = [f for f in numeric_features if f in df.columns]
    
    n_cols = 4
    n_rows = (len(available_features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(available_features):
        df[col].hist(bins=50, ax=axes[i], color='steelblue', edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Distribution: {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequence')
    
    # Remove empty subplots
    for j in range(len(available_features), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'distributions.png'), dpi=150)
    plt.close()
    print(f"  Saved: {REPORTS_DIR}/distributions.png")


def generate_summary_statistics(df):
    """Genere un fichier de statistiques resumees"""
    print("Generating summary statistics...")
    
    # Numeric summary
    numeric_summary = df.describe()
    numeric_summary.to_csv(os.path.join(REPORTS_DIR, 'numeric_summary.csv'))
    print(f"  Saved: {REPORTS_DIR}/numeric_summary.csv")
    
    # Categorical summary
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        cat_summary = []
        for col in cat_cols:
            summary = {
                'Column': col,
                'Unique Values': df[col].nunique(),
                'Most Common': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                'Most Common %': df[col].value_counts(normalize=True).iloc[0] * 100 if len(df[col]) > 0 else 0
            }
            cat_summary.append(summary)
        
        cat_df = pd.DataFrame(cat_summary)
        cat_df.to_csv(os.path.join(REPORTS_DIR, 'categorical_summary.csv'), index=False)
        print(f"  Saved: {REPORTS_DIR}/categorical_summary.csv")


def main():
    """Fonction principale"""
    print("="*60)
    print("GENERATION DES RAPPORTS ET VISUALISATIONS")
    print("="*60)
    
    # Creer le repertoire reports s'il n'existe pas
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Charger les donnees
    print("\nChargement des donnees...")
    df = load_raw_data()
    print(f"Dataset: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Generer tous les rapports
    print("\n" + "-"*40)
    generate_missing_values_report(df)
    
    print("-"*40)
    generate_target_distribution(df)
    
    print("-"*40)
    generate_rfm_analysis(df)
    
    print("-"*40)
    generate_correlation_heatmap(df)
    
    print("-"*40)
    generate_categorical_analysis(df)
    
    print("-"*40)
    generate_distribution_plots(df)
    
    print("-"*40)
    generate_summary_statistics(df)
    
    print("\n" + "="*60)
    print("RAPPORTS GENERES AVEC SUCCES!")
    print("="*60)
    print(f"\nTous les rapports sont disponibles dans: {REPORTS_DIR}")
    
    # Liste des fichiers generes
    print("\nFichiers generes:")
    for f in os.listdir(REPORTS_DIR):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
