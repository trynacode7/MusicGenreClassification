"""
Improved model training with better parameters and cross-validation.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_features, encode_labels, split_data, scale_features,
    save_preprocessed_data, load_preprocessed_data, GENRES
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_improved_random_forest(X_train, y_train, X_val, y_val, verbose=True):
    """Train Random Forest with hyperparameter tuning."""
    if verbose:
        logger.info("Training IMPROVED Random Forest with hyperparameter tuning...")
    
    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [300, 500, 800],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create base model
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Grid search with cross-validation
    if verbose:
        logger.info("Performing grid search for Random Forest...")
    
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=5, 
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1 if verbose else 0
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = grid_search.best_estimator_
    
    if verbose:
        logger.info(f"Best Random Forest parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    val_pred = best_rf.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    
    if verbose:
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation F1-Score: {val_f1:.4f}")
    
    return best_rf, val_accuracy, val_f1

def train_improved_svm(X_train, y_train, X_val, y_val, verbose=True):
    """Train SVM with hyperparameter tuning."""
    if verbose:
        logger.info("Training IMPROVED SVM with hyperparameter tuning...")
    
    # Define parameter grid for tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    # Create base model
    svm = SVC(
        random_state=42,
        probability=True,
        class_weight='balanced'
    )
    
    # Grid search with cross-validation
    if verbose:
        logger.info("Performing grid search for SVM...")
    
    grid_search = GridSearchCV(
        svm, 
        param_grid, 
        cv=3,  # Reduced CV for faster training
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1 if verbose else 0
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_svm = grid_search.best_estimator_
    
    if verbose:
        logger.info(f"Best SVM parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    val_pred = best_svm.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    
    if verbose:
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation F1-Score: {val_f1:.4f}")
    
    return best_svm, val_accuracy, val_f1

def save_improved_model(model, model_name, scaler, label_encoder):
    """Save improved model with metadata."""
    model_path = f"models/{model_name}_improved.pkl"
    scaler_path = "models/scaler_improved.pkl"
    encoder_path = "models/label_encoder_improved.pkl"
    
    # Save model
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    
    logger.info(f"Improved {model_name} saved to: {model_path}")
    logger.info(f"Improved scaler saved to: {scaler_path}")
    logger.info(f"Improved label encoder saved to: {encoder_path}")

def main():
    """Train improved models with better parameters."""
    logger.info("=" * 60)
    logger.info("Training IMPROVED Models with Hyperparameter Tuning")
    logger.info("=" * 60)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    _, X, y = load_features('Data/features_30_sec.csv')
    y_encoded, label_encoder, _ = encode_labels(y)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_encoded)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    logger.info(f"Training set: {X_train_scaled.shape}")
    logger.info(f"Validation set: {X_val_scaled.shape}")
    logger.info(f"Test set: {X_test_scaled.shape}")
    
    # Train improved Random Forest
    logger.info("\n" + "=" * 50)
    logger.info("Training IMPROVED Random Forest")
    logger.info("=" * 50)
    
    rf_model, rf_val_acc, rf_val_f1 = train_improved_random_forest(
        X_train_scaled, y_train, X_val_scaled, y_val
    )
    
    # Train improved SVM
    logger.info("\n" + "=" * 50)
    logger.info("Training IMPROVED SVM")
    logger.info("=" * 50)
    
    svm_model, svm_val_acc, svm_val_f1 = train_improved_svm(
        X_train_scaled, y_train, X_val_scaled, y_val
    )
    
    # Evaluate on test set
    logger.info("\n" + "=" * 50)
    logger.info("Evaluating IMPROVED Models on Test Set")
    logger.info("=" * 50)
    
    # Random Forest test evaluation
    rf_test_pred = rf_model.predict(X_test_scaled)
    rf_test_acc = accuracy_score(y_test, rf_test_pred)
    rf_test_f1 = f1_score(y_test, rf_test_pred, average='weighted')
    
    logger.info(f"IMPROVED Random Forest Test Results:")
    logger.info(f"  Accuracy: {rf_test_acc:.4f}")
    logger.info(f"  F1-Score: {rf_test_f1:.4f}")
    
    # SVM test evaluation
    svm_test_pred = svm_model.predict(X_test_scaled)
    svm_test_acc = accuracy_score(y_test, svm_test_pred)
    svm_test_f1 = f1_score(y_test, svm_test_pred, average='weighted')
    
    logger.info(f"IMPROVED SVM Test Results:")
    logger.info(f"  Accuracy: {svm_test_acc:.4f}")
    logger.info(f"  F1-Score: {svm_test_f1:.4f}")
    
    # Save improved models
    logger.info("\n" + "=" * 50)
    logger.info("Saving IMPROVED Models")
    logger.info("=" * 50)
    
    save_improved_model(rf_model, "random_forest", scaler, label_encoder)
    save_improved_model(svm_model, "svm", scaler, label_encoder)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("IMPROVED Training Summary")
    logger.info("=" * 60)
    logger.info(f"Random Forest - Test Accuracy: {rf_test_acc:.4f}, F1-Score: {rf_test_f1:.4f}")
    logger.info(f"SVM - Test Accuracy: {svm_test_acc:.4f}, F1-Score: {svm_test_f1:.4f}")
    logger.info("=" * 60)
    
    return rf_model, svm_model, scaler, label_encoder

if __name__ == "__main__":
    main()
