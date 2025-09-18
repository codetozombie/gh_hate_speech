import os
import sys
import numpy as np
import joblib
import warnings
from typing import List, Dict, Any, Union, Optional
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn

# Handle imports for both relative and direct execution
try:
    from .classical_ml import get_model
    from .deep_learning import get_dl_model, train_dl_model
    from ..utils import set_seed, mlflow_logger
except ImportError:
    # Add parent directory to path for direct execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from models.classical_ml import get_model
    from models.deep_learning import get_dl_model, train_dl_model
    from utils import set_seed, mlflow_logger

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available for ensemble logging")


class PyTorchSklearnWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper to make PyTorch models compatible with sklearn ensembles.
    Handles the conversion between sklearn and PyTorch data formats.
    """

    def __init__(self, pytorch_model, vocab_size=5000, max_len=128, device='cpu'):
        self.pytorch_model = pytorch_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.device = device
        self.is_trained = False

    def fit(self, X, y):
        """
        Train the PyTorch model using the sklearn interface.
        """
        # Convert to proper format for PyTorch training
        if isinstance(X, np.ndarray) and X.ndim == 2:
            # Assume X is already tokenized sequences
            X_torch = X.astype(np.int64)
        else:
            # If X needs tokenization, this would be handled by feature extraction
            X_torch = X

        y_torch = np.array(y, dtype=np.int64)

        # Train the model
        history = train_dl_model(
            model=self.pytorch_model,
            X_train=X_torch,
            y_train=y_torch,
            epochs=5,  # Reduced for ensemble training
            batch_size=32,
            device=self.device,
            experiment_name='ensemble_pytorch_component'
        )

        self.is_trained = True
        return self

    def predict(self, X):
        """
        Make predictions using the trained PyTorch model.
        """
        if not self.is_trained:
            raise ValueError("Model must be fitted before making predictions")

        self.pytorch_model.eval()

        # Convert to tensor
        if isinstance(X, np.ndarray):
            X_tensor = torch.LongTensor(X).to(self.device)
        else:
            X_tensor = X

        with torch.no_grad():
            outputs = self.pytorch_model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.cpu().numpy()

    def predict_proba(self, X):
        """
        Get prediction probabilities for ensemble soft voting.
        """
        if not self.is_trained:
            raise ValueError("Model must be fitted before making predictions")

        self.pytorch_model.eval()

        # Convert to tensor
        if isinstance(X, np.ndarray):
            X_tensor = torch.LongTensor(X).to(self.device)
        else:
            X_tensor = X

        with torch.no_grad():
            outputs = self.pytorch_model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()


def get_voting_ensemble(
    base_models: List[str] = ['lr', 'rf', 'xgb'],
    variant: int = 1,
    voting: str = 'soft',
    weights: Optional[List[float]] = None
) -> VotingClassifier:
    """
    Create voting ensemble as in Tesfay et al. (2025): Voting BiLSTM + mBERT (F1=0.86)
    and NaijaHate (Osei et al., 2024): Soft-voting RF/XGB + XLM-R (+13% precision).

    Args:
        base_models: List of base model names
        variant: Ensemble variant (1=classical, 2=hybrid, 3=advanced)
        voting: 'soft' or 'hard' voting
        weights: Optional weights for base models

    Returns:
        VotingClassifier: Configured voting ensemble
    """

    estimators = []

    # Classical models
    classical_models = [model for model in base_models if model in [
        'lr', 'nb', 'svm', 'dt', 'rf', 'xgb']]

    for model_name in classical_models:
        if variant == 1:
            # Baseline classical models
            model = get_model(model_name, variant=1)
        elif variant == 2:
            # Balanced models for imbalance handling
            model = get_model(model_name, variant=2)
        else:
            # Tuned models
            model = get_model(model_name, variant=3)

        estimators.append((f'{model_name}_classifier', model))

    # Deep learning models for hybrid variants
    if variant >= 2:
        dl_models = [model for model in base_models if model in [
            'bilstm', 'cnn', 'lstm', 'hybrid']]

        for model_name in dl_models:
            # Get PyTorch model
            pytorch_model = get_dl_model(
                model_name, variant=1 if variant == 2 else 2)

            # Wrap for sklearn compatibility
            wrapped_model = PyTorchSklearnWrapper(pytorch_model)
            estimators.append((f'{model_name}_classifier', wrapped_model))

    # Advanced variant: Add calibrated models
    if variant == 3:
        # Add calibrated versions of key models for better probability estimates
        if 'xgb' in classical_models:
            xgb_model = get_model('xgb', variant=3)
            calibrated_xgb = CalibratedClassifierCV(
                xgb_model, method='isotonic', cv=3)
            estimators.append(('calibrated_xgb', calibrated_xgb))

    # XGB in voting for imbalance as in Yusuf et al. (2025)
    ensemble = VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights
    )

    return ensemble


def get_stacking_ensemble(
    base_models: List[str] = ['lr', 'rf', 'xgb'],
    variant: int = 1,
    meta_learner: str = 'lr',
    cv: int = 5
) -> StackingClassifier:
    """
    Create stacking ensemble as in AfriHate (Muhammad et al., 2025): Stacking SVM/XGB + AfroXLMR (F1 +4.66)
    and Abdullahi et al. (2024): Stacking CNN-LSTM + AfriBERTa (F1=0.85).

    Args:
        base_models: List of base model names
        variant: Ensemble variant (1=classical, 2=hybrid, 3=advanced)
        meta_learner: Meta-learner model name
        cv: Cross-validation folds for stacking

    Returns:
        StackingClassifier: Configured stacking ensemble
    """

    estimators = []

    # Classical models
    classical_models = [model for model in base_models if model in [
        'lr', 'nb', 'svm', 'dt', 'rf', 'xgb']]

    for model_name in classical_models:
        if variant == 1:
            model = get_model(model_name, variant=1)
        elif variant == 2:
            # Balanced for imbalance
            model = get_model(model_name, variant=2)
        else:
            # Tuned models
            model = get_model(model_name, variant=3)

        estimators.append((f'{model_name}_base', model))

    # Deep learning models for hybrid variants
    if variant >= 2:
        dl_models = [model for model in base_models if model in [
            'bilstm', 'cnn', 'lstm', 'hybrid']]

        for model_name in dl_models:
            pytorch_model = get_dl_model(
                model_name, variant=1 if variant == 2 else 2)
            wrapped_model = PyTorchSklearnWrapper(pytorch_model)
            estimators.append((f'{model_name}_base', wrapped_model))

    # Configure meta-learner
    if meta_learner == 'lr':
        if variant >= 2:
            # Balanced meta-learner for imbalanced data
            final_estimator = LogisticRegression(
                class_weight='balanced', random_state=42)
        else:
            final_estimator = LogisticRegression(random_state=42)
    else:
        # Support other meta-learners
        final_estimator = get_model(
            meta_learner, variant=2 if variant >= 2 else 1)

    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        stack_method='predict_proba',  # Use probabilities for better stacking
        n_jobs=-1
    )

    return ensemble


def get_ensemble(
    ensemble_type: str = 'voting',
    base_models: List[str] = ['lr', 'rf', 'xgb'],
    variant: int = 1,
    **kwargs
) -> Union[VotingClassifier, StackingClassifier]:
    """
    Factory function to create ensemble models for Pidgin hate speech detection.

    Args:
        ensemble_type: Type of ensemble ('voting' or 'stacking')
        base_models: List of base model names
        variant: Ensemble variant (1=classical, 2=hybrid, 3=advanced)
        **kwargs: Additional parameters for specific ensemble types

    Returns:
        Union[VotingClassifier, StackingClassifier]: Configured ensemble

    Raises:
        ValueError: If ensemble_type is invalid
    """

    if ensemble_type not in ['voting', 'stacking']:
        raise ValueError(
            f"Invalid ensemble_type: {ensemble_type}. Must be 'voting' or 'stacking'")

    if variant not in [1, 2, 3]:
        raise ValueError(
            f"Invalid variant: {variant}. Must be 1 (classical), 2 (hybrid), or 3 (advanced)")

    # Set seed for reproducibility
    set_seed(42)

    if ensemble_type == 'voting':
        return get_voting_ensemble(base_models, variant, **kwargs)
    elif ensemble_type == 'stacking':
        return get_stacking_ensemble(base_models, variant, **kwargs)


def fit_ensemble(
    ensemble: Union[VotingClassifier, StackingClassifier],
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_path: Optional[str] = None,
    experiment_name: str = 'pidgin_hate_ensemble'
) -> Dict[str, Any]:
    """
    Fit ensemble model and optionally save it.

    Args:
        ensemble: Ensemble model to train
        X_train: Training features
        y_train: Training labels
        save_path: Optional path to save trained ensemble
        experiment_name: MLflow experiment name

    Returns:
        Dict[str, Any]: Training information and metrics
    """

    set_seed(42)

    print(f"Training ensemble with {len(ensemble.estimators)} base models...")

    # Fit the ensemble
    ensemble.fit(X_train, y_train)

    # Calculate training accuracy
    train_score = ensemble.score(X_train, y_train)

    print(f"Ensemble training completed. Training accuracy: {train_score:.4f}")

    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(ensemble, save_path)
        print(f"Ensemble saved to: {save_path}")

    # Log to MLflow if available
    if MLFLOW_AVAILABLE:
        try:
            # Get ensemble info
            ensemble_info = {
                'ensemble_type': type(ensemble).__name__,
                'num_estimators': len(ensemble.estimators),
                'base_models': [name for name, _ in ensemble.estimators]
            }

            mlflow_logger(
                experiment=experiment_name,
                tags={
                    'model_type': 'ensemble',
                    'ensemble_type': type(ensemble).__name__,
                    'framework': 'sklearn_pytorch_hybrid'
                },
                params=ensemble_info,
                metrics={'train_accuracy': train_score}
            )

        except Exception as e:
            print(f"MLflow logging failed: {e}")

    return {
        'train_accuracy': train_score,
        'ensemble_type': type(ensemble).__name__,
        'num_estimators': len(ensemble.estimators)
    }


def get_ensemble_info(ensemble: Union[VotingClassifier, StackingClassifier]) -> Dict[str, Any]:
    """
    Get information about an ensemble model.

    Args:
        ensemble: Trained ensemble model

    Returns:
        Dict[str, Any]: Ensemble information
    """

    info = {
        'type': type(ensemble).__name__,
        'num_estimators': len(ensemble.estimators),
        'base_models': []
    }

    for name, estimator in ensemble.estimators:
        model_info = {
            'name': name,
            'type': type(estimator).__name__
        }

        # Check if it's a pipeline
        if hasattr(estimator, 'steps'):
            model_info['pipeline_steps'] = [step[0]
                                            for step in estimator.steps]

        info['base_models'].append(model_info)

    # Add meta-learner info for stacking
    if isinstance(ensemble, StackingClassifier):
        info['meta_learner'] = type(ensemble.final_estimator_).__name__

    return info


def load_ensemble(model_path: str) -> Union[VotingClassifier, StackingClassifier]:
    """
    Load a saved ensemble model.

    Args:
        model_path: Path to saved ensemble model

    Returns:
        Union[VotingClassifier, StackingClassifier]: Loaded ensemble
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Ensemble model not found at: {model_path}")

    ensemble = joblib.load(model_path)
    print(f"Ensemble loaded from: {model_path}")

    return ensemble


if __name__ == '__main__':
    # Test ensemble creation and training
    print("Testing Ensemble Models for Pidgin Hate Speech\n")

    # Create dummy data
    np.random.seed(42)
    dummy_X = np.random.randint(0, 1000, (100, 128))  # Tokenized sequences
    dummy_y = np.random.randint(0, 2, (100,))  # Binary labels

    print(f"Dummy data shapes: X={dummy_X.shape}, y={dummy_y.shape}")

    # Test different ensemble types and variants
    ensemble_configs = [
        ('voting', ['lr', 'rf', 'xgb'], 1),
        ('voting', ['lr', 'rf', 'xgb'], 2),
        ('stacking', ['lr', 'rf', 'xgb'], 1),
        ('stacking', ['lr', 'rf', 'xgb'], 2),
    ]

    for ens_type, base_models, variant in ensemble_configs:
        try:
            print(
                f"\n{variant}. Testing {ens_type.upper()} ensemble variant {variant}:")
            print(f"   Base models: {base_models}")

            # Create ensemble
            ensemble = get_ensemble(
                ensemble_type=ens_type,
                base_models=base_models,
                variant=variant
            )

            print(f"   ✓ Ensemble created: {type(ensemble).__name__}")
            print(f"   ✓ Number of estimators: {len(ensemble.estimators)}")

            # Test fitting (reduced epochs for quick testing)
            if variant <= 1:  # Only test classical variants to avoid long DL training
                result = fit_ensemble(
                    ensemble=ensemble,
                    X_train=dummy_X,
                    y_train=dummy_y,
                    save_path=f'data/models/test_ensemble_{ens_type}_{variant}.pkl'
                )

                print(
                    f"   ✓ Training completed: {result['train_accuracy']:.4f} accuracy")

                # Test predictions
                predictions = ensemble.predict(dummy_X[:10])
                print(f"   ✓ Predictions shape: {predictions.shape}")

                if hasattr(ensemble, 'predict_proba'):
                    probas = ensemble.predict_proba(dummy_X[:10])
                    print(f"   ✓ Probabilities shape: {probas.shape}")

        except Exception as e:
            print(f"   ✗ Error with {ens_type} variant {variant}: {e}")

    print("\n5. Testing ensemble info:")
    try:
        test_ensemble = get_ensemble('voting', ['lr', 'rf'], variant=1)
        info = get_ensemble_info(test_ensemble)
        print(f"Ensemble info: {info}")
    except Exception as e:
        print(f"Error getting ensemble info: {e}")

    print("\nEnsemble model tests completed!")
