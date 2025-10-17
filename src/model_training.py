# Music Genre Classification - Optimized Training Module
# GTZAN Dataset - Random Forest, SVM, CNN

"""
Model Training Module

Implements training for:
- Random Forest Classifier
- SVM Classifier  
- CNN for spectrogram images

Main Functions:
- train_models_on_features()
- train_cnn_on_spectrograms()
- save_model()
- load_model()
"""

import os
import sys
import glob
import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
from PIL import Image

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add src directory for utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_features, encode_labels, split_data, scale_features,
    save_preprocessed_data, load_preprocessed_data, GENRES
)

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# GPU configuration
# -------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        logger.warning(f"Could not set GPU memory growth: {e}")

# -------------------------------
# Random Forest / SVM Functions
# -------------------------------

def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_estimators: int = 500, max_depth: int = 20,
    min_samples_split: int = 5, min_samples_leaf: int = 2,
    random_state: int = 42, verbose: bool = True
) -> Tuple[RandomForestClassifier, float, float, float, float]:
    """Train Random Forest Classifier."""
    if verbose:
        logger.info(f"Random Forest: Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples")
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        verbose=1 if verbose else 0
    )
    rf.fit(X_train, y_train)
    
    y_train_pred = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')

    if verbose:
        logger.info(f"RF Training complete | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
    
    return rf, train_acc, val_acc, train_f1, val_f1


def train_svm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    kernel: str = 'rbf', C: float = 10.0, gamma: Any = 'auto',
    random_state: int = 42, verbose: bool = True
) -> Tuple[SVC, float, float, float, float]:
    """Train Support Vector Machine."""
    if verbose:
        logger.info(f"SVM: Training with kernel={kernel}, C={C}, gamma={gamma}")
    
    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        random_state=random_state,
        probability=True,
        class_weight='balanced',
        verbose=1 if verbose else 0
    )
    svm.fit(X_train, y_train)
    
    y_train_pred = svm.predict(X_train)
    y_val_pred = svm.predict(X_val)
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')

    if verbose:
        logger.info(f"SVM Training complete | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
    
    return svm, train_acc, val_acc, train_f1, val_f1


def hyperparameter_tuning_rf(X_train, y_train, cv=3, verbose=True):
    """Random Forest hyperparameter tuning."""
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1 if verbose else 0)
    grid.fit(X_train, y_train)
    if verbose:
        logger.info(f"Best RF params: {grid.best_params_}, CV F1: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def hyperparameter_tuning_svm(X_train, y_train, cv=3, verbose=True):
    """SVM hyperparameter tuning."""
    param_grid = {'C': [0.1,1,10], 'gamma': ['scale','auto'], 'kernel':['rbf','linear']}
    svm = SVC(random_state=42)
    grid = GridSearchCV(svm, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1 if verbose else 0)
    grid.fit(X_train, y_train)
    if verbose:
        logger.info(f"Best SVM params: {grid.best_params_}, CV F1: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def save_model(model, model_name: str, save_dir='models'):
    """Save model to disk."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}.pkl")
    joblib.dump(model, path)
    logger.info(f"Model saved: {path}")


def load_model(model_name: str, load_dir='models'):
    """Load model from disk."""
    path = os.path.join(load_dir, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded: {path}")
    return model


def evaluate_model(model, X_test, y_test, model_name: str, label_encoder=None) -> Dict[str, Any]:
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    if label_encoder:
        target_names = label_encoder.classes_
    else:
        target_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    
    logger.info(f"{model_name} | Acc: {accuracy:.4f}, F1_weighted: {f1_weighted:.4f}, F1_macro: {f1_macro:.4f}")
    
    return {'accuracy': accuracy, 'f1_weighted': f1_weighted, 'f1_macro': f1_macro, 
            'confusion_matrix': cm, 'classification_report': report, 'predictions': y_pred}


# -------------------------------
# CNN Functions
# -------------------------------

def load_spectrogram_images(data_dir='Data/images_original', target_size=(128,128)) -> Tuple[np.ndarray,np.ndarray,LabelEncoder]:
    """Load images and labels from directories."""
    images, labels = [], []
    genres = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))])
    if not genres:
        raise ValueError(f"No genre directories found in {data_dir}")

    for genre in genres:
        for f in glob.glob(os.path.join(data_dir, genre, '*.png')):
            try:
                img = Image.open(f).convert('RGB').resize(target_size)
                images.append(np.array(img)/255.0)
                labels.append(genre)
            except Exception as e:
                logger.warning(f"Error loading {f}: {e}")
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    logger.info(f"Loaded {len(images)} images with {len(np.unique(encoded_labels))} classes.")
    return images, encoded_labels, le


def create_cnn_model(input_shape=(128,128,3), num_classes=10) -> models.Model:
    """Build CNN for spectrogram classification."""
    model = models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape), layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)), layers.Dropout(0.25),
        layers.Conv2D(64,(3,3),activation='relu'), layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)), layers.Dropout(0.25),
        layers.Conv2D(128,(3,3),activation='relu'), layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)), layers.Dropout(0.25),
        layers.Conv2D(256,(3,3),activation='relu'), layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)), layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512,activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(256,activation='relu'), layers.Dropout(0.5),
        layers.Dense(num_classes,activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    logger.info("CNN model created.")
    return model


def train_cnn(
    images, labels, test_size=0.2, val_size=0.2,
    epochs=50, batch_size=32, target_size=(128,128)
):
    """Train CNN on spectrograms with data augmentation and class weights."""
    X_temp, X_test, y_temp, y_test = train_test_split(images, labels, test_size=test_size, stratify=labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size/(1-test_size), stratify=y_temp, random_state=42)
    
    model = create_cnn_model(input_shape=(*target_size,3), num_classes=len(np.unique(labels)))
    
    train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1,
                                       height_shift_range=0.1, horizontal_flip=True, zoom_range=0.1)
    
    cb = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ]
    
    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    
    history = model.fit(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train)//batch_size, epochs=epochs,
                        validation_data=(X_val,y_val), callbacks=cb, class_weight=class_weights, verbose=1)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    test_pred_classes = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_f1 = f1_score(y_test, test_pred_classes, average='weighted')
    
    test_results = {'test_accuracy': test_acc, 'test_f1': test_f1,
                    'test_predictions': test_pred_classes, 'test_true_labels': y_test}
    
    logger.info(f"CNN Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    return model, history.history, test_results


def save_cnn_model(model, model_name='cnn_model', save_dir='models'):
    """Save CNN model in H5 and SavedModel formats."""
    os.makedirs(save_dir, exist_ok=True)
    h5_path = os.path.join(save_dir, f"{model_name}.h5")
    model.save(h5_path)
    tf_path = os.path.join(save_dir, f"{model_name}_savedmodel")
    model.save(tf_path, save_format='tf')
    logger.info(f"CNN saved: {h5_path} & {tf_path}")


def train_models_on_features(data_file='Data/features_30_sec.csv', use_hyperparameter_tuning=False, save_models=True):
    """Train RF and SVM models on extracted features."""
    df, features, labels = load_features(data_file)
    encoded_labels, label_encoder, _ = encode_labels(labels)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, encoded_labels)
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)
    
    if save_models:
        save_preprocessed_data({'scaler':scaler,'label_encoder':label_encoder,'X_test':X_test,'y_test':y_test})
    
    if use_hyperparameter_tuning:
        rf_model, _, _ = hyperparameter_tuning_rf(X_train, y_train)
        svm_model, _, _ = hyperparameter_tuning_svm(X_train, y_train)
    else:
        rf_model, *_ = train_random_forest(X_train, y_train, X_val, y_val)
        svm_model, *_ = train_svm(X_train, y_train, X_val, y_val)
    
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest", label_encoder)
    svm_results = evaluate_model(svm_model, X_test, y_test, "SVM", label_encoder)
    
    if save_models:
        save_model(rf_model, "random_forest_model")
        save_model(svm_model, "svm_model")
    
    return {'random_forest': rf_results, 'svm': svm_results}


def train_cnn_on_spectrograms(data_dir='Data/images_original', target_size=(128,128), epochs=50, batch_size=32, save_model_flag=True):
    """Train CNN on spectrogram images."""
    images, labels, le = load_spectrogram_images(data_dir, target_size)
    model, history, test_results = train_cnn(images, labels, epochs=epochs, batch_size=batch_size, target_size=target_size)
    
    if save_model_flag:
        save_cnn_model(model)
        save_preprocessed_data({'cnn_label_encoder': le})
    
    return {'model': model, 'history': history, 'test_results': test_results, 'label_encoder': le}


# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    try:
        logger.info("Training traditional ML models...")
        ml_results = train_models_on_features()
        
        logger.info("Training CNN on spectrogram images...")
        cnn_results = train_cnn_on_spectrograms()
    except Exception as e:
        logger.error(f"Error during training: {e}")
