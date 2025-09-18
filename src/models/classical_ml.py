from typing import Dict, Any, Union
import warnings
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. XGB models will not work.")

# Handle imports for both relative and direct execution
try:
    from ..feature_engineering import FeatureExtractor
except ImportError:
    # Add parent directory to path for direct execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from feature_engineering import FeatureExtractor


def get_model(name: str, variant: int = 1) -> Pipeline:
    """
    Factory function to get classical ML models for hate speech detection.

    Args:
        name (str): Model name ('lr', 'nb', 'svm', 'dt', 'rf', 'xgb')
        variant (int): Model variant (1=baseline, 2=balanced, 3=tuned)

    Returns:
        Pipeline: Scikit-learn pipeline with feature extraction and model

    Raises:
        ValueError: If model name or variant is invalid
        ImportError: If XGBoost is not available for 'xgb' model
    """

    if name not in ['lr', 'nb', 'svm', 'dt', 'rf', 'xgb']:
        raise ValueError(
            f"Invalid model name: {name}. Must be one of ['lr', 'nb', 'svm', 'dt', 'rf', 'xgb']")

    if variant not in [1, 2, 3]:
        raise ValueError(
            f"Invalid variant: {variant}. Must be 1 (baseline), 2 (balanced), or 3 (tuned)")

    if name == 'xgb' and not XGBOOST_AVAILABLE:
        raise ImportError(
            "XGBoost not available. Please install with: pip install xgboost")

    # Get the appropriate model based on name and variant
    model = _get_base_model(name, variant)

    # Create pipeline with TF-IDF feature extraction
    pipeline = Pipeline([
        ('features', FeatureExtractor('tfidf')),
        ('model', model)
    ])

    return pipeline


def _get_base_model(name: str, variant: int) -> BaseEstimator:
    """
    Get the base model without pipeline wrapper.

    Args:
        name (str): Model name
        variant (int): Model variant

    Returns:
        BaseEstimator: Configured model
    """

    if name == 'lr':
        # Logistic Regression baseline as in NaijaHate (Osei et al., 2024)
        if variant == 1:
            return LogisticRegression(random_state=42, max_iter=1000)
        elif variant == 2:
            return LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        elif variant == 3:
            return LogisticRegression(random_state=42, max_iter=1000, C=1.0, class_weight='balanced')

    elif name == 'nb':
        # Multinomial Naive Bayes for text classification
        if variant == 1:
            return MultinomialNB()
        elif variant == 2:
            # NB doesn't have direct class_weight, use fit_prior=False for balance
            return MultinomialNB(fit_prior=False)
        elif variant == 3:
            return MultinomialNB(alpha=0.1, fit_prior=False)

    elif name == 'svm':
        # Linear SVM as in NaijaHate baselines
        if variant == 1:
            return SVC(kernel='linear', random_state=42, probability=True)
        elif variant == 2:
            return SVC(kernel='linear', random_state=42, probability=True, class_weight='balanced')
        elif variant == 3:
            return SVC(kernel='linear', random_state=42, probability=True, C=1.0, class_weight='balanced')

    elif name == 'dt':
        # Decision Tree baseline
        if variant == 1:
            return DecisionTreeClassifier(random_state=42)
        elif variant == 2:
            return DecisionTreeClassifier(random_state=42, class_weight='balanced')
        elif variant == 3:
            return DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=20, class_weight='balanced')

    elif name == 'rf':
        # Random Forest as in NaijaHate baselines
        if variant == 1:
            return RandomForestClassifier(random_state=42)
        elif variant == 2:
            return RandomForestClassifier(random_state=42, class_weight='balanced')
        elif variant == 3:
            # Tuned parameters for better performance
            return RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced'
            )

    elif name == 'xgb':
        # XGB for low-resource trees as in EkoHate (Oladipo et al., 2024)
        if variant == 1:
            return xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        elif variant == 2:
            # Calculate scale_pos_weight for imbalanced data
            # This will be approximated, ideally calculated from actual data
            return xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=2.0  # Approximate for hate/non-hate imbalance
            )
        elif variant == 3:
            # XGB tuned parameters as in EkoHate (F1~0.71 > RF) and VocalTweets (F1=0.86)
            return xgb.XGBClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=2.0,
                eval_metric='logloss'
            )


def get_param_grid(name: str) -> Dict[str, Any]:
    """
    Get parameter grid for GridSearchCV hyperparameter tuning.

    Args:
        name (str): Model name

    Returns:
        Dict[str, Any]: Parameter grid for GridSearchCV

    Raises:
        ValueError: If model name is invalid
    """

    if name == 'lr':
        return {
            'model__C': [0.1, 1.0, 10.0],
            'model__class_weight': [None, 'balanced'],
            'model__solver': ['liblinear', 'lbfgs']
        }

    elif name == 'nb':
        return {
            'model__alpha': [0.1, 0.5, 1.0, 2.0],
            'model__fit_prior': [True, False]
        }

    elif name == 'svm':
        return {
            'model__C': [0.1, 1.0, 10.0],
            'model__class_weight': [None, 'balanced'],
            'model__gamma': ['scale', 'auto']
        }

    elif name == 'dt':
        return {
            'model__max_depth': [5, 10, 15, None],
            'model__min_samples_split': [2, 5, 10, 20],
            'model__class_weight': [None, 'balanced']
        }

    elif name == 'rf':
        return {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [5, 10, 15, None],
            'model__min_samples_split': [2, 5, 10],
            'model__class_weight': [None, 'balanced']
        }

    elif name == 'xgb':
        # XGB grid from VocalTweets (Yusuf et al., 2024) and EkoHate patterns
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available for parameter grid")

        return {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 6, 9],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__subsample': [0.8, 0.9, 1.0],
            'model__colsample_bytree': [0.8, 0.9, 1.0],
            'model__scale_pos_weight': [1, 2, 3]
        }

    else:
        raise ValueError(
            f"Invalid model name: {name}. Must be one of ['lr', 'nb', 'svm', 'dt', 'rf', 'xgb']")


def get_model_info(name: str) -> Dict[str, str]:
    """
    Get information about a specific model.

    Args:
        name (str): Model name

    Returns:
        Dict[str, str]: Model information including description and references
    """

    model_info = {
        'lr': {
            'name': 'Logistic Regression',
            'description': 'Linear classifier for hate speech detection',
            'reference': 'NaijaHate (Osei et al., 2024) baseline'
        },
        'nb': {
            'name': 'Multinomial Naive Bayes',
            'description': 'Probabilistic classifier assuming feature independence',
            'reference': 'Classical text classification baseline'
        },
        'svm': {
            'name': 'Support Vector Machine',
            'description': 'Linear SVM for text classification',
            'reference': 'NaijaHate (Osei et al., 2024) baseline'
        },
        'dt': {
            'name': 'Decision Tree',
            'description': 'Tree-based classifier for interpretable decisions',
            'reference': 'NaijaHate (Osei et al., 2024) baseline'
        },
        'rf': {
            'name': 'Random Forest',
            'description': 'Ensemble of decision trees',
            'reference': 'NaijaHate (Osei et al., 2024) baseline, compared with XGB in EkoHate'
        },
        'xgb': {
            'name': 'XGBoost',
            'description': 'Gradient boosting for Pidgin imbalance handling',
            'reference': 'EkoHate (Oladipo et al., 2024): F1~0.71 > RF; VocalTweets (Yusuf et al., 2024): F1=0.86'
        }
    }

    return model_info.get(name, {'name': 'Unknown', 'description': 'Unknown model', 'reference': 'None'})


def list_available_models() -> Dict[str, Dict[str, str]]:
    """
    List all available models with their information.

    Returns:
        Dict[str, Dict[str, str]]: Dictionary of model names and their info
    """

    models = ['lr', 'nb', 'svm', 'dt', 'rf']
    if XGBOOST_AVAILABLE:
        models.append('xgb')

    return {name: get_model_info(name) for name in models}


if __name__ == '__main__':
    # Test the model factory
    print("Testing Classical ML Model Factory\n")

    # Test XGBoost model with balanced variant
    try:
        print("Testing XGBoost balanced model:")
        model = get_model('xgb', 2)
        print(f"XGB Model: {model}")
        print(f"XGB Steps: {model.steps}")
        print()

        # Test parameter grid
        print("Testing XGBoost parameter grid:")
        param_grid = get_param_grid('xgb')
        print(f"XGB Param Grid: {param_grid}")
        print()

    except ImportError as e:
        print(f"XGBoost test failed: {e}")
        print("Testing Logistic Regression instead:")
        model = get_model('lr', 2)
        print(f"LR Model: {model}")
        print()

    # Test all available models
    print("Available models:")
    for name, info in list_available_models().items():
        print(f"- {name}: {info['name']} ({info['reference']})")

    print("\nTesting all model variants:")
    test_models = ['lr', 'nb', 'svm', 'dt', 'rf']
    if XGBOOST_AVAILABLE:
        test_models.append('xgb')

    for model_name in test_models:
        for variant in [1, 2, 3]:
            try:
                model = get_model(model_name, variant)
                print(f"{model_name.upper()} variant {variant}: ✓")
            except Exception as e:
                print(f"{model_name.upper()} variant {variant}: ✗ ({e})")

    print("\nClassical ML factory tests completed!")
