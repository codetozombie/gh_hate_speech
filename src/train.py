import argparse
import os
import sys
import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import warnings
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Handle imports for both relative and direct execution
try:
    from .feature_engineering import FeatureExtractor, SequenceFeatureExtractor, tokenize_sequences
    from .models.classical_ml import get_model, get_param_grid
    from .models.deep_learning import get_dl_model, train_dl_model
    from .models.ensembles import get_ensemble, fit_ensemble
    from .utils import set_seed, mlflow_logger, calculate_metrics, save_classification_report
except ImportError:
    # Add current directory to path for direct execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    from feature_engineering import FeatureExtractor, SequenceFeatureExtractor, tokenize_sequences
    from models.classical_ml import get_model, get_param_grid
    from models.deep_learning import get_dl_model, train_dl_model
    from models.ensembles import get_ensemble, fit_ensemble
    from utils import set_seed, mlflow_logger, calculate_metrics, save_classification_report

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available for experiment tracking")


def load_config() -> Dict[str, Any]:
    """
    Load configuration for training.
    Hardcoded paths and settings as per specification.

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    return {
        'data_paths': {
            'train': 'data/processed/train.csv',
            'val': 'data/processed/val.csv',
            'test': 'data/processed/test.csv',
            'models': 'data/models/',
            'predictions': 'data/processed/'
        },
        'feature_params': {
            'max_features': 5000,
            'max_len': 128,
            'ngram_range': (1, 2),
            'vocab_size': 5000
        },
        'cv_params': {
            'cv_folds': 5,
            'scoring': 'f1_weighted'
        },
        'dl_params': {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'device': 'cpu'
        }
    }


def load_processed_data(config: Dict[str, Any]) -> tuple:
    """
    Load preprocessed data from CSV files.

    Args:
        config: Configuration dictionary

    Returns:
        tuple: (train_data, val_data, test_data)
    """

    print("Loading preprocessed datasets...")

    # Load train data
    train_path = config['data_paths']['train']
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at: {train_path}")

    train_data = pd.read_csv(train_path)
    print(f"Train data loaded: {train_data.shape}")

    # Load validation data
    val_path = config['data_paths']['val']
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found at: {val_path}")

    val_data = pd.read_csv(val_path)
    print(f"Validation data loaded: {val_data.shape}")

    # Load test data
    test_path = config['data_paths']['test']
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at: {test_path}")

    test_data = pd.read_csv(test_path)
    print(f"Test data loaded: {test_data.shape}")

    print(f"Available columns: {list(train_data.columns)}")

    return train_data, val_data, test_data


def extract_features_and_labels(data: pd.DataFrame) -> tuple:
    """
    Extract features (text) and labels from DataFrame.

    Args:
        data: Input DataFrame

    Returns:
        tuple: (X, y) where X is text and y is labels
    """

    # Try to identify text and label columns
    text_col = None
    label_col = None

    # Common column names for text
    text_candidates = ['text', 'content', 'message',
                       'tweet', 'post', 'comment', 'cleaned_text']
    for col in text_candidates:
        if col in data.columns:
            text_col = col
            break

    # Common column names for labels
    label_candidates = ['label', 'class',
                        'target', 'hate', 'is_hate', 'category']
    for col in label_candidates:
        if col in data.columns:
            label_col = col
            break

    # If not found, use first text-like column and last column
    if text_col is None:
        for col in data.columns:
            if data[col].dtype == 'object':
                text_col = col
                break

        if text_col is None:
            text_col = data.columns[0]

    if label_col is None:
        # Look for numeric columns
        for col in data.columns:
            if col != text_col and data[col].dtype in ['int64', 'float64']:
                label_col = col
                break

        if label_col is None:
            label_col = data.columns[-1]

    print(f"Using columns: text='{text_col}', label='{label_col}'")

    # Extract data
    X = data[text_col].fillna('').astype(str).values
    y = data[label_col].values

    # Ensure labels are binary (0, 1)
    unique_labels = np.unique(y)
    print(f"Found labels: {unique_labels}")

    if len(unique_labels) == 2:
        # Map to 0, 1
        if set(unique_labels) == {0, 1}:
            # Already binary
            pass
        else:
            # Map to binary
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
            y = np.array([label_map[label] for label in y])
    else:
        # Multi-class to binary (first class as negative, rest as positive)
        y = (y != unique_labels[0]).astype(int)

    print(f"Final class distribution: {np.bincount(y)}")

    return X, y


def prepare_features(
    X: np.ndarray,
    feature_type: str,
    config: Dict[str, Any],
    is_training: bool = True,
    feature_extractor: Optional[FeatureExtractor] = None
) -> tuple:
    """
    Prepare features based on the specified feature type.

    Args:
        X: Input data (text or sequences)
        feature_type: Type of features ('tfidf', 'count', 'sequence', 'fasttext')
        config: Configuration dictionary
        is_training: Whether this is training phase
        feature_extractor: Pre-fitted feature extractor (for test data)

    Returns:
        tuple: (features, feature_extractor)
    """

    if feature_type == 'sequence':
        # For deep learning - convert text to sequences
        if is_training:
            # Convert text to tokenized sequences
            sequences = tokenize_sequences(
                X.tolist(),
                max_len=config['feature_params']['max_len'],
                vocab_size=config['feature_params']['vocab_size']
            )

            feature_extractor = SequenceFeatureExtractor(
                max_len=config['feature_params']['max_len'])
            features = feature_extractor.fit_transform(sequences)
        else:
            # Convert text to sequences using same tokenization
            sequences = tokenize_sequences(
                X.tolist(),
                max_len=config['feature_params']['max_len'],
                vocab_size=config['feature_params']['vocab_size']
            )
            features = feature_extractor.transform(sequences)
    else:
        # For classical ML - use traditional features
        if feature_extractor is None:
            feature_extractor = FeatureExtractor(
                method=feature_type,
                max_features=config['feature_params']['max_features'],
                ngram_range=config['feature_params']['ngram_range'],
                max_len=config['feature_params']['max_len']
            )

        if is_training:
            features = feature_extractor.fit_transform(X)
        else:
            features = feature_extractor.transform(X)

    return features, feature_extractor


def train_classical_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    variant: int,
    config: Dict[str, Any],
    use_grid_search: bool = True
) -> Dict[str, Any]:
    """
    Train classical machine learning model.
    GridSearchCV 5-fold as in NaijaHate/AfriHate.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_name: Name of the model
        variant: Model variant
        config: Configuration dictionary
        use_grid_search: Whether to use grid search

    Returns:
        Dict[str, Any]: Training results
    """

    print(
        f"Training classical {model_name.upper()} model (variant {variant})...")

    # Get base model
    model = get_model(model_name, variant=variant)

    if use_grid_search:
        # GridSearchCV 5-fold on TF-IDF/XGB (seed=42) as in NaijaHate/AfriHate
        param_grid = get_param_grid(model_name)

        if param_grid:
            print(
                f"Performing GridSearchCV with {len(param_grid)} parameter combinations...")

            # Remove random_state parameter that doesn't exist in GridSearchCV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=config['cv_params']['cv_folds'],
                scoring=config['cv_params']['scoring'],
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            print(f"Best parameters: {best_params}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            print("No parameter grid available. Using default model...")
            best_model = model
            best_model.fit(X_train, y_train)
            best_params = None
    else:
        # Direct training without grid search
        best_model = model
        best_model.fit(X_train, y_train)
        best_params = None

    # Evaluate on validation set
    val_predictions = best_model.predict(X_val)
    val_probas = best_model.predict_proba(X_val) if hasattr(
        best_model, 'predict_proba') else None

    # Calculate metrics
    metrics = calculate_metrics(y_val, val_predictions)

    print(f"Validation Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")

    return {
        'model': best_model,
        'metrics': metrics,
        'predictions': val_predictions,
        'probabilities': val_probas,
        'best_params': best_params
    }


def train_deep_learning_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    variant: int,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train deep learning model.

    Args:
        X_train: Training features (tokenized sequences)
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_name: Name of the model
        variant: Model variant
        config: Configuration dictionary

    Returns:
        Dict[str, Any]: Training results
    """

    print(
        f"Training deep learning {model_name.upper()} model (variant {variant})...")

    # Get model
    model = get_dl_model(
        name=model_name,
        variant=variant,
        vocab_size=config['feature_params']['vocab_size'],
        max_len=config['feature_params']['max_len']
    )

    # Train model
    history = train_dl_model(
        model=model,
        X_train=X_train.astype(np.int64),
        y_train=y_train.astype(np.int64),
        X_val=X_val.astype(np.int64),
        y_val=y_val.astype(np.int64),
        epochs=config['dl_params']['epochs'],
        batch_size=config['dl_params']['batch_size'],
        learning_rate=config['dl_params']['learning_rate'],
        device=config['dl_params']['device']
    )

    # Get final predictions
    import torch
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.LongTensor(X_val.astype(
            np.int64)).to(config['dl_params']['device'])
        outputs = model(X_val_tensor)
        val_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        val_probas = torch.softmax(outputs, dim=1).cpu().numpy()

    # Calculate metrics
    metrics = calculate_metrics(y_val, val_predictions)

    print(f"Validation Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")

    return {
        'model': model,
        'metrics': metrics,
        'predictions': val_predictions,
        'probabilities': val_probas,
        'history': history
    }


def train_ensemble_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    ensemble_type: str,
    base_models: list,
    variant: int,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train ensemble model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        ensemble_type: Type of ensemble ('voting', 'stacking')
        base_models: List of base model names
        variant: Ensemble variant
        config: Configuration dictionary

    Returns:
        Dict[str, Any]: Training results
    """

    print(f"Training {ensemble_type.upper()} ensemble (variant {variant})...")
    print(f"Base models: {base_models}")

    # Get ensemble
    ensemble = get_ensemble(
        ensemble_type=ensemble_type,
        base_models=base_models,
        variant=variant
    )

    # Train ensemble
    result = fit_ensemble(
        ensemble=ensemble,
        X_train=X_train,
        y_train=y_train
    )

    # Evaluate on validation set
    val_predictions = ensemble.predict(X_val)
    val_probas = ensemble.predict_proba(X_val) if hasattr(
        ensemble, 'predict_proba') else None

    # Calculate metrics
    metrics = calculate_metrics(y_val, val_predictions)

    print(f"Validation Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")

    return {
        'model': ensemble,
        'metrics': metrics,
        'predictions': val_predictions,
        'probabilities': val_probas,
        'train_info': result
    }


def save_model(model: Any, model_type: str, model_name: str, variant: int, config: Dict[str, Any]) -> str:
    """
    Save trained model.

    Args:
        model: Trained model
        model_type: Type of model ('classical', 'dl', 'ensemble')
        model_name: Name of the model
        variant: Model variant
        config: Configuration dictionary

    Returns:
        str: Path where model was saved
    """

    models_dir = config['data_paths']['models']
    os.makedirs(models_dir, exist_ok=True)

    if model_type == 'dl':
        # Save PyTorch model
        import torch
        filename = f"{model_type}_{model_name}_{variant}.pth"
        filepath = os.path.join(models_dir, filename)
        torch.save(model.state_dict(), filepath)
    else:
        # Save sklearn model
        filename = f"{model_type}_{model_name}_{variant}.pkl"
        filepath = os.path.join(models_dir, filename)
        joblib.dump(model, filepath)

    print(f"Model saved to: {filepath}")
    return filepath


def save_predictions(
    predictions: np.ndarray,
    split: str,
    config: Dict[str, Any],
    probabilities: Optional[np.ndarray] = None
) -> str:
    """
    Save predictions to CSV file.

    Args:
        predictions: Model predictions
        split: Data split ('train', 'val', 'test')
        config: Configuration dictionary
        probabilities: Prediction probabilities (optional)

    Returns:
        str: Path where predictions were saved
    """

    pred_dir = config['data_paths']['predictions']
    os.makedirs(pred_dir, exist_ok=True)

    # Create predictions DataFrame
    pred_df = pd.DataFrame({
        'prediction': predictions
    })

    if probabilities is not None:
        if probabilities.ndim == 2:
            for i in range(probabilities.shape[1]):
                pred_df[f'prob_class_{i}'] = probabilities[:, i]
        else:
            pred_df['probability'] = probabilities

    # Save to CSV
    filename = f"preds_{split}.csv"
    filepath = os.path.join(pred_dir, filename)
    pred_df.to_csv(filepath, index=False)

    print(f"Predictions saved to: {filepath}")
    return filepath


def main():
    """
    Main training function with CLI interface.
    """

    parser = argparse.ArgumentParser(
        description='Train Pidgin Hate Speech Detection Models')

    # Feature arguments
    parser.add_argument('--feat_type', type=str, default='tfidf',
                        choices=['tfidf', 'count', 'sequence', 'fasttext'],
                        help='Feature extraction type')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='classical',
                        choices=['classical', 'dl', 'ensemble'],
                        help='Type of model to train')

    parser.add_argument('--model', type=str, default='xgb',
                        help='Specific model name (e.g., xgb, bilstm, voting)')

    parser.add_argument('--variant', type=int, default=1,
                        choices=[1, 2, 3],
                        help='Model variant (1=base, 2=imbalance, 3=advanced)')

    # Advanced training options
    parser.add_argument('--advanced_type', type=str, default=None,
                        choices=['dl', 'transformer'],
                        help='Advanced model type for sequential training')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for DL training')

    parser.add_argument('--ensemble_type', type=str, default='voting',
                        choices=['voting', 'stacking'],
                        help='Type of ensemble')

    parser.add_argument('--base_models', type=str, nargs='+',
                        default=['lr', 'rf', 'xgb'],
                        help='Base models for ensemble')

    # Output options
    parser.add_argument('--no_grid_search', action='store_true',
                        help='Skip grid search for classical models')

    parser.add_argument('--experiment_name', type=str, default='pidgin_hate_training',
                        help='MLflow experiment name')

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(42)

    # Load configuration
    config = load_config()

    # Update config with CLI arguments
    config['dl_params']['epochs'] = args.epochs

    print("=" * 60)
    print("PIDGIN HATE SPEECH DETECTION - MODEL TRAINING")
    print("=" * 60)
    print(f"Model Type: {args.model_type}")
    print(f"Model: {args.model}")
    print(f"Variant: {args.variant}")
    print(f"Feature Type: {args.feat_type}")
    print("=" * 60)

    try:
        # Load preprocessed data
        print("\n1. Loading preprocessed data...")
        train_data, val_data, test_data = load_processed_data(config)

        # Extract features and labels
        X_train, y_train = extract_features_and_labels(train_data)
        X_val, y_val = extract_features_and_labels(val_data)
        X_test, y_test = extract_features_and_labels(test_data)

        print(
            f"Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

        # Prepare features
        print("\n2. Preparing features...")

        X_train_features, feature_extractor = prepare_features(
            X_train, args.feat_type, config, is_training=True
        )
        X_val_features, _ = prepare_features(
            X_val, args.feat_type, config, is_training=False,
            feature_extractor=feature_extractor
        )
        X_test_features, _ = prepare_features(
            X_test, args.feat_type, config, is_training=False,
            feature_extractor=feature_extractor
        )

        print(
            f"Feature shapes: Train={X_train_features.shape}, Val={X_val_features.shape}, Test={X_test_features.shape}")

        # Train model
        print("\n3. Training model...")

        if args.model_type == 'classical':
            # Train classical model
            result = train_classical_model(
                X_train_features, y_train, X_val_features, y_val,
                args.model, args.variant, config,
                use_grid_search=not args.no_grid_search
            )

        elif args.model_type == 'dl':
            # Train deep learning model
            result = train_deep_learning_model(
                X_train_features, y_train, X_val_features, y_val,
                args.model, args.variant, config
            )

        elif args.model_type == 'ensemble':
            # Train ensemble model
            result = train_ensemble_model(
                X_train_features, y_train, X_val_features, y_val,
                args.ensemble_type, args.base_models, args.variant, config
            )

        # Save model
        print("\n4. Saving model...")
        model_path = save_model(
            result['model'], args.model_type, args.model, args.variant, config
        )

        # Save predictions
        print("\n5. Saving predictions...")
        save_predictions(result['predictions'], 'val',
                         config, result.get('probabilities'))

        # Test on test set
        print("\n6. Evaluating on test set...")
        if args.model_type == 'dl':
            # Handle PyTorch model
            import torch
            result['model'].eval()
            with torch.no_grad():
                X_test_tensor = torch.LongTensor(X_test_features.astype(
                    np.int64)).to(config['dl_params']['device'])
                outputs = result['model'](X_test_tensor)
                test_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        else:
            test_predictions = result['model'].predict(X_test_features)

        test_metrics = calculate_metrics(y_test, test_predictions)

        print(f"Test Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {test_metrics['f1_score']:.4f}")

        save_predictions(test_predictions, 'test', config)

        # Save classification report
        save_classification_report(y_test, test_predictions)

        # Log to MLflow
        if MLFLOW_AVAILABLE:
            print("\n7. Logging to MLflow...")

            mlflow_params = {
                'model_type': args.model_type,
                'model_name': args.model,
                'variant': args.variant,
                'feature_type': args.feat_type,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            }

            if args.model_type == 'classical' and result.get('best_params'):
                mlflow_params.update(result['best_params'])

            mlflow_logger(
                experiment=args.experiment_name,
                tags={'model_type': args.model_type,
                      'feature_type': args.feat_type},
                params=mlflow_params,
                metrics={
                    'val_accuracy': result['metrics']['accuracy'],
                    'val_f1_score': result['metrics']['f1_score'],
                    'test_accuracy': test_metrics['accuracy'],
                    'test_f1_score': test_metrics['f1_score']
                },
                model=result['model']
            )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Model saved to: {model_path}")
        print(f"Final Test F1-Score: {test_metrics['f1_score']:.4f}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
