import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_raw_data, save_processed_data, save_train_test_data,
    explore_data, plot_missing_values, plot_correlation_matrix,
    parse_registration_date, parse_ip_address,
    handle_outliers, handle_special_values, impute_missing_values,
    encode_ordinal_features, encode_onehot_features,
    create_feature_engineering, save_model,
    DATA_PROCESSED, REPORTS_DIR
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def preprocess_data(df):
    print("\n" + "="*60)
    print("DEBUT DU PREPROCESSING")
    print("="*60)
    
    df_clean = df.copy()
    print(f"\nShape initial: {df_clean.shape}")
    
    if 'CustomerID' in df_clean.columns:
        customer_ids = df_clean['CustomerID'].copy()
        df_clean = df_clean.drop('CustomerID', axis=1)
        print("CustomerID supprime (garde en memoire)")
    
    if 'NewsletterSubscribed' in df_clean.columns:
        df_clean = df_clean.drop('NewsletterSubscribed', axis=1)
        print("NewsletterSubscribed supprime (variance nulle)")
    
    if 'RegistrationDate' in df_clean.columns:
        df_clean = parse_registration_date(df_clean, 'RegistrationDate')
        df_clean = df_clean.drop('RegistrationDate', axis=1)
        print("RegistrationDate parse et features extraites")
    
    if 'LastLoginIP' in df_clean.columns:
        df_clean = parse_ip_address(df_clean, 'LastLoginIP')
        df_clean = df_clean.drop('LastLoginIP', axis=1)
        print("LastLoginIP parse et features extraites")
    
    df_clean = handle_special_values(df_clean)
    print("Valeurs speciales (-1, 999, 99) traitees")
    
    outlier_cols = ['MonetaryTotal', 'TotalQuantity', 'MonetaryMin', 'MonetaryMax']
    for col in outlier_cols:
        if col in df_clean.columns:
            df_clean = handle_outliers(df_clean, col, method='iqr', factor=3)
    
    df_clean = create_feature_engineering(df_clean)
    df_clean = encode_ordinal_features(df_clean)
    
    nominal_cols = ['CustomerType', 'FavoriteSeason', 'Region', 
                    'WeekendPreference', 'ProductDiversity', 'Gender', 
                    'AccountStatus', 'IPClass']
    df_clean = encode_onehot_features(df_clean, nominal_cols)
    
    if 'Country' in df_clean.columns:
        le_country = LabelEncoder()
        df_clean['Country_encoded'] = le_country.fit_transform(df_clean['Country'].astype(str))
        save_model(le_country, 'label_encoder_country.joblib')
        print(f"Country encode: {len(le_country.classes_)} categories")
    
    cols_to_drop = [
        'RFMSegment', 'AgeCategory', 'SpendingCategory', 'CustomerType',
        'FavoriteSeason', 'PreferredTimeOfDay', 'Region', 'LoyaltyLevel',
        'ChurnRiskCategory', 'WeekendPreference', 'BasketSizeCategory',
        'ProductDiversity', 'Gender', 'AccountStatus', 'Country', 'IPClass'
    ]
    for col in cols_to_drop:
        if col in df_clean.columns:
            df_clean = df_clean.drop(col, axis=1)
    
    print(f"\nShape apres encodage: {df_clean.shape}")
    
    df_clean = impute_missing_values(df_clean, numeric_strategy='median')
    
    missing_count = df_clean.isnull().sum().sum()
    print(f"\nValeurs manquantes restantes: {missing_count}")
    
    print(f"\nShape final: {df_clean.shape}")
    print("="*60)
    print("PREPROCESSING TERMINE")
    print("="*60)
    
    return df_clean, customer_ids


def prepare_datasets(df_processed, target_col='Churn'):
    print("\n" + "="*60)
    print("PREPARATION DES DATASETS")
    print("="*60)
    
    if target_col in df_processed.columns:
        y = df_processed[target_col]
        X = df_processed.drop(target_col, axis=1)
    else:
        raise ValueError(f"Colonne target '{target_col}' non trouvee")
    
    print(f"\nFeatures (X): {X.shape}")
    print(f"Target (y): {y.shape}")
    print(f"\nDistribution du target ({target_col}):")
    print(y.value_counts(normalize=True))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    save_model(scaler, 'standard_scaler.joblib')
    print("\nStandardScaler sauvegarde")
    
    save_model(X_train.columns.tolist(), 'feature_names.joblib')
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def main():
    print("\n" + "="*60)
    print("CHARGEMENT DES DONNEES")
    print("="*60)
    df = load_raw_data()
    
    explore_data(df)
    
    try:
        plot_missing_values(df)
    except Exception as e:
        print(f"Erreur visualisation: {e}")
    
    df_processed, customer_ids = preprocess_data(df)
    save_processed_data(df_processed, 'retail_customers_processed.csv')
    
    try:
        plot_correlation_matrix(df_processed)
    except Exception as e:
        print(f"Erreur correlation matrix: {e}")
    
    X_train, X_test, y_train, y_test = prepare_datasets(df_processed, target_col='Churn')
    save_train_test_data(X_train, X_test, pd.Series(y_train), pd.Series(y_test))
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print("\nFichiers generes:")
    print("  - data/processed/retail_customers_processed.csv")
    print("  - data/train_test/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print("  - models/standard_scaler.joblib, label_encoder_country.joblib, feature_names.joblib")
    
    return df_processed, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df_processed, X_train, X_test, y_train, y_test = main()
