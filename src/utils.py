# Music Genre Classification - Utility Functions
# This module contains helper functions for data loading and preprocessing

"""
Utility Functions Module

This module implements:
- Data loading from CSV files
- Label encoding
- Train/validation/test split
- Data preprocessing
- Helper functions

Functions:
- load_features()
- encode_labels()
- split_data()
- preprocess_audio()
- save_preprocessed_data()
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the 10 music genres
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def load_features(file_path, verbose=True):
    """
    Load features from CSV file and handle missing/corrupted data.
    
    Args:
        file_path (str): Path to the CSV file
        verbose (bool): Whether to print loading information
    
    Returns:
        tuple: (features_df, features, labels) where:
            - features_df: Original DataFrame
            - features: Feature matrix (X)
            - labels: Label array (y)
    """
    try:
        if verbose:
            logger.info(f"Loading features from: {file_path}")
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        if verbose:
            logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values in the dataset")
            # Remove rows with missing values
            df = df.dropna()
            logger.info(f"Removed rows with missing values. Remaining samples: {len(df)}")
        
        # Separate features and labels
        feature_columns = [col for col in df.columns if col not in ['filename', 'label', 'length']]
        features = df[feature_columns].values
        labels = df['label'].values
        
        if verbose:
            logger.info(f"Feature matrix shape: {features.shape}")
            logger.info(f"Unique labels: {np.unique(labels)}")
            logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df, features, labels
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading features from {file_path}: {str(e)}")
        raise

def encode_labels(labels, encoder=None, fit_encoder=True):
    """
    Encode string labels to numerical values.
    
    Args:
        labels (array-like): String labels
        encoder (LabelEncoder, optional): Pre-fitted encoder
        fit_encoder (bool): Whether to fit the encoder
    
    Returns:
        tuple: (encoded_labels, label_encoder, label_mapping)
    """
    if encoder is None:
        encoder = LabelEncoder()
    
    if fit_encoder:
        encoded_labels = encoder.fit_transform(labels)
        label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    else:
        encoded_labels = encoder.transform(labels)
        label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    
    logger.info(f"Label encoding completed. Mapping: {label_mapping}")
    
    return encoded_labels, encoder, label_mapping

def split_data(features, labels, test_size=0.2, val_size=0.2, random_state=42, stratify=True):
    """
    Split data into train, validation, and test sets.
    
    Args:
        features (array): Feature matrix
        labels (array): Label array
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of remaining data for validation set
        random_state (int): Random state for reproducibility
        stratify (bool): Whether to stratify the split
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info(f"Splitting data with test_size={test_size}, val_size={val_size}")
    
    # First split: separate test set
    stratify_param = labels if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_param
    )
    
    # Second split: separate train and validation from remaining data
    if stratify:
        stratify_param = y_temp
    else:
        stratify_param = None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size/(1-test_size),  # Adjust val_size for remaining data
        random_state=random_state,
        stratify=stratify_param
    )
    
    logger.info(f"Data split completed:")
    logger.info(f"  Train set: {X_train.shape[0]} samples")
    logger.info(f"  Validation set: {X_val.shape[0]} samples")
    logger.info(f"  Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test, scaler=None, fit_scaler=True):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train (array): Training features
        X_val (array): Validation features
        X_test (array): Test features
        scaler (StandardScaler, optional): Pre-fitted scaler
        fit_scaler (bool): Whether to fit the scaler
    
    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if fit_scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        logger.info("Scaler fitted on training data")
    else:
        X_train_scaled = scaler.transform(X_train)
    
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Feature scaling completed")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def preprocess_audio(file_path, sr=22050):
    """
    Basic audio preprocessing function (placeholder for future audio processing).
    
    Args:
        file_path (str): Path to audio file
        sr (int): Sample rate
    
    Returns:
        array: Audio data
    """
    # This is a placeholder function for future audio preprocessing
    # For now, we're using pre-extracted features from CSV files
    logger.info(f"Audio preprocessing for {file_path} (placeholder)")
    return None

def save_preprocessed_data(data_dict, save_dir='models'):
    """
    Save preprocessed data and scalers.
    
    Args:
        data_dict (dict): Dictionary containing data to save
        save_dir (str): Directory to save files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for name, data in data_dict.items():
        file_path = os.path.join(save_dir, f"{name}.pkl")
        joblib.dump(data, file_path)
        logger.info(f"Saved {name} to {file_path}")

def load_preprocessed_data(load_dir='models', files=None):
    """
    Load preprocessed data and scalers.
    
    Args:
        load_dir (str): Directory to load files from
        files (list): List of file names to load (without .pkl extension)
    
    Returns:
        dict: Dictionary containing loaded data
    """
    if files is None:
        files = os.listdir(load_dir)
        files = [f.replace('.pkl', '') for f in files if f.endswith('.pkl')]
    
    loaded_data = {}
    for file_name in files:
        file_path = os.path.join(load_dir, f"{file_name}.pkl")
        if os.path.exists(file_path):
            loaded_data[file_name] = joblib.load(file_path)
            logger.info(f"Loaded {file_name} from {file_path}")
        else:
            logger.warning(f"File not found: {file_path}")
    
    return loaded_data

def get_feature_info(file_path):
    """
    Get information about the features in a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        dict: Information about the features
    """
    df = pd.read_csv(file_path)
    
    info = {
        'total_samples': len(df),
        'total_features': len(df.columns) - 3,  # Exclude filename, length, label
        'feature_columns': [col for col in df.columns if col not in ['filename', 'label', 'length']],
        'genres': df['label'].unique().tolist(),
        'genre_counts': df['label'].value_counts().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    return info

def validate_data_integrity(file_path):
    """
    Validate the integrity of the dataset.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        dict: Validation results
    """
    df = pd.read_csv(file_path)
    
    validation_results = {
        'file_path': file_path,
        'total_samples': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'expected_genres': set(GENRES),
        'actual_genres': set(df['label'].unique()),
        'genre_mismatch': set(GENRES) != set(df['label'].unique()),
        'samples_per_genre': df['label'].value_counts().to_dict(),
        'is_valid': True
    }
    
    # Check for issues
    if validation_results['missing_values'] > 0:
        logger.warning(f"Found {validation_results['missing_values']} missing values")
        validation_results['is_valid'] = False
    
    if validation_results['duplicate_rows'] > 0:
        logger.warning(f"Found {validation_results['duplicate_rows']} duplicate rows")
        validation_results['is_valid'] = False
    
    if validation_results['genre_mismatch']:
        logger.warning(f"Genre mismatch: expected {validation_results['expected_genres']}, got {validation_results['actual_genres']}")
        validation_results['is_valid'] = False
    
    return validation_results
