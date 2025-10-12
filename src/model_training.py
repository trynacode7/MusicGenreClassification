# Music Genre Classification - Model Training
# This module contains functions for training various ML models on the GTZAN dataset

"""
Model Training Module

This module implements training for:
- Random Forest Classifier
- SVM Classifier  
- CNN for spectrogram images

Functions:
- train_random_forest()
- train_svm()
- train_cnn()
- save_model()
- load_model()
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import glob

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_features, encode_labels, split_data, scale_features,
    save_preprocessed_data, load_preprocessed_data, GENRES
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_random_forest(X_train, y_train, X_val, y_val, n_estimators=100, random_state=42, verbose=True):
    """
    Train a Random Forest Classifier.
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        X_val (array): Validation features
        y_val (array): Validation labels
        n_estimators (int): Number of trees in the forest
        random_state (int): Random state for reproducibility
        verbose (bool): Whether to print training information
    
    Returns:
        tuple: (trained_model, train_accuracy, val_accuracy, train_f1, val_f1)
    """
    if verbose:
        logger.info("Training Random Forest Classifier...")
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Validation set shape: {X_val.shape}")
    
    # Initialize Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        verbose=1 if verbose else 0
    )
    
    # Train the model
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    if verbose:
        logger.info(f"Random Forest Training Results:")
        logger.info(f"  Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"  Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"  Training F1-Score: {train_f1:.4f}")
        logger.info(f"  Validation F1-Score: {val_f1:.4f}")
    
    return rf, train_accuracy, val_accuracy, train_f1, val_f1

def train_svm(X_train, y_train, X_val, y_val, kernel='rbf', C=1.0, gamma='scale', random_state=42, verbose=True):
    """
    Train a Support Vector Machine Classifier.
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        X_val (array): Validation features
        y_val (array): Validation labels
        kernel (str): Kernel type ('rbf', 'linear', 'poly')
        C (float): Regularization parameter
        gamma (str or float): Kernel coefficient
        random_state (int): Random state for reproducibility
        verbose (bool): Whether to print training information
    
    Returns:
        tuple: (trained_model, train_accuracy, val_accuracy, train_f1, val_f1)
    """
    if verbose:
        logger.info("Training Support Vector Machine...")
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Validation set shape: {X_val.shape}")
        logger.info(f"SVM Parameters: kernel={kernel}, C={C}, gamma={gamma}")
    
    # Initialize SVM
    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        random_state=random_state,
        verbose=1 if verbose else 0
    )
    
    # Train the model
    svm.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = svm.predict(X_train)
    y_val_pred = svm.predict(X_val)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    if verbose:
        logger.info(f"SVM Training Results:")
        logger.info(f"  Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"  Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"  Training F1-Score: {train_f1:.4f}")
        logger.info(f"  Validation F1-Score: {val_f1:.4f}")
    
    return svm, train_accuracy, val_accuracy, train_f1, val_f1

def hyperparameter_tuning_rf(X_train, y_train, X_val, y_val, cv=3, verbose=True):
    """
    Perform hyperparameter tuning for Random Forest.
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        X_val (array): Validation features
        y_val (array): Validation labels
        cv (int): Number of cross-validation folds
        verbose (bool): Whether to print tuning information
    
    Returns:
        tuple: (best_model, best_params, best_score)
    """
    if verbose:
        logger.info("Performing hyperparameter tuning for Random Forest...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1 if verbose else 0
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    if verbose:
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def hyperparameter_tuning_svm(X_train, y_train, X_val, y_val, cv=3, verbose=True):
    """
    Perform hyperparameter tuning for SVM.
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        X_val (array): Validation features
        y_val (array): Validation labels
        cv (int): Number of cross-validation folds
        verbose (bool): Whether to print tuning information
    
    Returns:
        tuple: (best_model, best_params, best_score)
    """
    if verbose:
        logger.info("Performing hyperparameter tuning for SVM...")
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    
    # Initialize SVM
    svm = SVC(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        svm, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1 if verbose else 0
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    if verbose:
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def save_model(model, model_name, save_dir='models'):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        model_name (str): Name for the saved model
        save_dir (str): Directory to save the model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, f"{model_name}.pkl")
    joblib.dump(model, file_path)
    logger.info(f"Model saved to: {file_path}")

def load_model(model_name, load_dir='models'):
    """
    Load a trained model from disk.
    
    Args:
        model_name (str): Name of the model to load
        load_dir (str): Directory to load the model from
    
    Returns:
        Trained model object
    """
    file_path = os.path.join(load_dir, f"{model_name}.pkl")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    model = joblib.load(file_path)
    logger.info(f"Model loaded from: {file_path}")
    
    return model

def evaluate_model(model, X_test, y_test, model_name, label_encoder=None):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test (array): Test features
        y_test (array): Test labels
        model_name (str): Name of the model
        label_encoder: Label encoder for converting predictions back to strings
    
    Returns:
        dict: Evaluation metrics
    """
    logger.info(f"Evaluating {model_name} on test data...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    # Get classification report
    if label_encoder:
        target_names = label_encoder.classes_
    else:
        target_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
    
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': y_pred
    }
    
    logger.info(f"{model_name} Test Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1-Score (Weighted): {f1_weighted:.4f}")
    logger.info(f"  F1-Score (Macro): {f1_macro:.4f}")
    
    return results

def train_models_on_features(data_file='Data/features_30_sec.csv', use_hyperparameter_tuning=False, save_models=True):
    """
    Main function to train models on extracted features.
    
    Args:
        data_file (str): Path to the features CSV file
        use_hyperparameter_tuning (bool): Whether to use hyperparameter tuning
        save_models (bool): Whether to save trained models
    
    Returns:
        dict: Results from training both models
    """
    logger.info("=" * 60)
    logger.info("Training Models on Extracted Features")
    logger.info("=" * 60)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df, features, labels = load_features(data_file)
    
    # Encode labels
    encoded_labels, label_encoder, label_mapping = encode_labels(labels)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        features, encoded_labels, test_size=0.2, val_size=0.2
    )
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )
    
    # Save preprocessing artifacts
    if save_models:
        save_preprocessed_data({
            'scaler': scaler,
            'label_encoder': label_encoder,
            'X_test': X_test_scaled,
            'y_test': y_test
        })
    
    results = {}
    
    # Train Random Forest
    logger.info("\n" + "=" * 40)
    logger.info("Training Random Forest Classifier")
    logger.info("=" * 40)
    
    if use_hyperparameter_tuning:
        rf_model, rf_params, rf_cv_score = hyperparameter_tuning_rf(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        logger.info(f"Random Forest best parameters: {rf_params}")
    else:
        rf_model, rf_train_acc, rf_val_acc, rf_train_f1, rf_val_f1 = train_random_forest(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
    
    # Evaluate Random Forest
    rf_results = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest", label_encoder)
    results['random_forest'] = rf_results
    
    # Train SVM
    logger.info("\n" + "=" * 40)
    logger.info("Training Support Vector Machine")
    logger.info("=" * 40)
    
    if use_hyperparameter_tuning:
        svm_model, svm_params, svm_cv_score = hyperparameter_tuning_svm(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        logger.info(f"SVM best parameters: {svm_params}")
    else:
        svm_model, svm_train_acc, svm_val_acc, svm_train_f1, svm_val_f1 = train_svm(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
    
    # Evaluate SVM
    svm_results = evaluate_model(svm_model, X_test_scaled, y_test, "SVM", label_encoder)
    results['svm'] = svm_results
    
    # Save models
    if save_models:
        save_model(rf_model, "random_forest_model")
        save_model(svm_model, "svm_model")
        logger.info("Models saved successfully!")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"Random Forest - Test Accuracy: {rf_results['accuracy']:.4f}, F1-Score: {rf_results['f1_weighted']:.4f}")
    logger.info(f"SVM - Test Accuracy: {svm_results['accuracy']:.4f}, F1-Score: {svm_results['f1_weighted']:.4f}")
    
    return results

def load_spectrogram_images(data_dir='Data/images_original', target_size=(128, 128), verbose=True):
    """
    Load spectrogram images and labels from directory structure.
    
    Args:
        data_dir (str): Path to the images directory
        target_size (tuple): Target size for image resizing
        verbose (bool): Whether to print loading information
    
    Returns:
        tuple: (images, labels, label_encoder)
    """
    if verbose:
        logger.info(f"Loading spectrogram images from: {data_dir}")
        logger.info(f"Target size: {target_size}")
    
    images = []
    labels = []
    genre_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    genre_dirs.sort()  # Ensure consistent ordering
    
    if verbose:
        logger.info(f"Found genres: {genre_dirs}")
    
    for genre in genre_dirs:
        genre_path = os.path.join(data_dir, genre)
        image_files = glob.glob(os.path.join(genre_path, "*.png"))
        
        if verbose:
            logger.info(f"Loading {len(image_files)} images from {genre}")
        
        for image_file in image_files:
            try:
                # Load and preprocess image
                img = Image.open(image_file)
                
                # Convert RGBA to RGB if necessary
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img = img.resize(target_size)
                
                # Convert to numpy array and normalize
                img_array = np.array(img) / 255.0
                
                images.append(img_array)
                labels.append(genre)
                
            except Exception as e:
                if verbose:
                    logger.warning(f"Error loading image {image_file}: {e}")
                continue
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    if verbose:
        logger.info(f"Loaded {len(images)} images with shape {images.shape}")
        logger.info(f"Label distribution: {np.bincount(encoded_labels)}")
        logger.info(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    return images, encoded_labels, label_encoder

def create_cnn_model(input_shape=(128, 128, 3), num_classes=10, verbose=True):
    """
    Create a CNN model for spectrogram classification.
    
    Args:
        input_shape (tuple): Input shape of images
        num_classes (int): Number of output classes
        verbose (bool): Whether to print model information
    
    Returns:
        keras.Model: Compiled CNN model
    """
    if verbose:
        logger.info(f"Creating CNN model with input shape: {input_shape}")
        logger.info(f"Number of classes: {num_classes}")
    
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    if verbose:
        logger.info("CNN model created and compiled successfully")
        model.summary()
    
    return model

def train_cnn(images, labels, test_size=0.2, val_size=0.2, epochs=50, batch_size=32, 
              target_size=(128, 128), verbose=True):
    """
    Train a CNN model on spectrogram images.
    
    Args:
        images (array): Image data
        labels (array): Label data
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        target_size (tuple): Target image size
        verbose (bool): Whether to print training information
    
    Returns:
        tuple: (trained_model, history, test_results)
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("Training CNN on Spectrogram Images")
        logger.info("=" * 60)
        logger.info(f"Input shape: {images.shape}")
        logger.info(f"Number of classes: {len(np.unique(labels))}")
        logger.info(f"Training parameters: epochs={epochs}, batch_size={batch_size}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
    )
    
    if verbose:
        logger.info(f"Data split:")
        logger.info(f"  Train: {X_train.shape[0]} samples")
        logger.info(f"  Validation: {X_val.shape[0]} samples")
        logger.info(f"  Test: {X_test.shape[0]} samples")
    
    # Create model
    model = create_cnn_model(input_shape=(*target_size, 3), num_classes=len(np.unique(labels)), verbose=verbose)
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1 if verbose else 0
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1 if verbose else 0
        )
    ]
    
    # Train the model
    if verbose:
        logger.info("Starting CNN training...")
    
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1 if verbose else 0
    )
    
    # Evaluate on test set
    if verbose:
        logger.info("Evaluating CNN on test set...")
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    test_predictions = model.predict(X_test, verbose=0)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    
    test_f1 = f1_score(y_test, test_pred_classes, average='weighted')
    
    if verbose:
        logger.info(f"CNN Test Results:")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Test F1-Score: {test_f1:.4f}")
    
    test_results = {
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_predictions': test_pred_classes,
        'test_true_labels': y_test
    }
    
    return model, history, test_results

def save_cnn_model(model, model_name='cnn_model', save_dir='models'):
    """
    Save a trained CNN model.
    
    Args:
        model: Trained Keras model
        model_name (str): Name for the saved model
        save_dir (str): Directory to save the model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as H5 format
    h5_path = os.path.join(save_dir, f"{model_name}.h5")
    model.save(h5_path)
    logger.info(f"CNN model saved to: {h5_path}")
    
    # Also save as Keras format for better compatibility
    keras_path = os.path.join(save_dir, f"{model_name}.keras")
    model.save(keras_path)
    logger.info(f"CNN model (Keras format) saved to: {keras_path}")

def load_cnn_model(model_name='cnn_model', load_dir='models'):
    """
    Load a trained CNN model.
    
    Args:
        model_name (str): Name of the model to load
        load_dir (str): Directory to load the model from
    
    Returns:
        keras.Model: Loaded CNN model
    """
    h5_path = os.path.join(load_dir, f"{model_name}.h5")
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"CNN model file not found: {h5_path}")
    
    model = keras.models.load_model(h5_path)
    logger.info(f"CNN model loaded from: {h5_path}")
    
    return model

def train_cnn_on_spectrograms(data_dir='Data/images_original', target_size=(128, 128), 
                             epochs=50, batch_size=32, save_model=True):
    """
    Main function to train CNN on spectrogram images.
    
    Args:
        data_dir (str): Path to spectrogram images directory
        target_size (tuple): Target size for image resizing
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        save_model (bool): Whether to save the trained model
    
    Returns:
        dict: Training results
    """
    logger.info("=" * 60)
    logger.info("Training CNN on Spectrogram Images")
    logger.info("=" * 60)
    
    # Load spectrogram images
    images, labels, label_encoder = load_spectrogram_images(data_dir, target_size)
    
    # Train CNN
    model, history, test_results = train_cnn(
        images, labels, 
        epochs=epochs, 
        batch_size=batch_size, 
        target_size=target_size
    )
    
    # Save model and preprocessing artifacts
    if save_model:
        save_cnn_model(model)
        save_preprocessed_data({
            'cnn_label_encoder': label_encoder
        })
        logger.info("CNN model and preprocessing artifacts saved!")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CNN Training Summary")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {test_results['test_accuracy']:.4f}")
    logger.info(f"Test F1-Score: {test_results['test_f1']:.4f}")
    
    results = {
        'model': model,
        'history': history,
        'test_results': test_results,
        'label_encoder': label_encoder
    }
    
    return results

if __name__ == "__main__":
    # Train traditional ML models
    print("Training traditional ML models...")
    ml_results = train_models_on_features()
    
    # Train CNN on spectrograms
    print("\nTraining CNN on spectrograms...")
    cnn_results = train_cnn_on_spectrograms()
