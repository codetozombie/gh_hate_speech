import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc
import warnings

# Optional imports with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Torch seeding will be skipped.")

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available. Experiment tracking will be limited.")

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    warnings.warn("langdetect not available. Using regex-based detection.")


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    PyTorch-focused implementation as referenced in AfriHate (Muhammad et al., 2025).

    Args:
        seed (int): Random seed value
    """
    # Set Python hash seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set NumPy seed
    np.random.seed(seed)

    # Set PyTorch seed (primary focus)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set generator seed for newer PyTorch versions
        torch.Generator().manual_seed(seed)

    # Set random seed for Python's random module
    import random
    random.seed(seed)

    print(
        f"All random seeds set to {seed} for reproducibility (PyTorch-focused)")


def mlflow_logger(
    experiment: str = 'pidgin_hate',
    tags: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    model: Optional[Any] = None,
    model_name: str = 'model'
) -> str:
    """
    MLflow logger for experiment tracking as in EkoHate (Oladipo et al., 2024).
    Supports both PyTorch and sklearn models.

    Args:
        experiment (str): MLflow experiment name
        tags (dict): Tags to log
        params (dict): Parameters to log
        metrics (dict): Metrics to log
        model: Model to log (PyTorch or sklearn compatible)
        model_name (str): Name for the logged model

    Returns:
        str: Run ID

    Raises:
        ImportError: If MLflow is not available
    """
    if not MLFLOW_AVAILABLE:
        raise ImportError(
            "MLflow not available. Please install with: pip install mlflow")

    # Set experiment
    mlflow.set_experiment(experiment)

    with mlflow.start_run() as run:
        # Log tags
        if tags:
            mlflow.set_tags(tags)

        # Log parameters
        if params:
            mlflow.log_params(params)

        # Log metrics
        if metrics:
            mlflow.log_metrics(metrics)

        # Log model if provided
        if model is not None:
            try:
                # Check if it's a PyTorch model
                if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                    mlflow.pytorch.log_model(model, model_name)
                    print(f"PyTorch model logged as '{model_name}'")
                else:
                    # Assume sklearn model
                    mlflow.sklearn.log_model(model, model_name)
                    print(f"Sklearn model logged as '{model_name}'")
            except Exception as e:
                print(f"Warning: Could not log model: {e}")

        run_id = run.info.run_id
        print(f"MLflow run completed. Run ID: {run_id}")

    return run_id


def plot_metrics(
    y_true: List[int],
    y_pred: Union[List[int], List[List[float]], torch.Tensor],
    plot_type: str = 'cm',
    save_path: str = 'reports/figures/',
    title: Optional[str] = None,
    labels: Optional[List[str]] = None
) -> None:
    """
    Plot various evaluation metrics as in NaijaHate (Osei et al., 2024).
    Supports PyTorch tensors as input.

    Args:
        y_true: True labels
        y_pred: Predicted labels, probabilities, or PyTorch tensors
        plot_type: Type of plot ('cm', 'roc', 'pr', 'dist')
        save_path: Directory to save plots
        title: Plot title
        labels: Class labels for display

    Raises:
        ValueError: If plot_type is invalid
    """
    os.makedirs(save_path, exist_ok=True)

    # Convert PyTorch tensors to numpy if needed
    if TORCH_AVAILABLE and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if TORCH_AVAILABLE and isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    plt.figure(figsize=(8, 6))

    if labels is None:
        labels = ['Non-Hate', 'Hate']

    if plot_type == 'cm':
        # Confusion Matrix heatmap
        if isinstance(y_pred[0], (list, np.ndarray)) and len(y_pred[0]) > 1:
            # If probabilities, convert to predictions
            y_pred_binary = np.argmax(y_pred, axis=1)
        elif hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
            y_pred_binary = np.argmax(y_pred, axis=1)
        else:
            y_pred_binary = y_pred

        cm = confusion_matrix(y_true, y_pred_binary)

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title(title or 'Confusion Matrix - Pidgin Hate Speech Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        filename = 'confusion_matrix.png'

    elif plot_type == 'roc':
        # ROC Curve
        if isinstance(y_pred[0], (list, np.ndarray)) and len(y_pred[0]) > 1:
            # Use probabilities for positive class
            y_scores = np.array(y_pred)[:, 1]
        elif hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
            y_scores = y_pred[:, 1]
        else:
            y_scores = y_pred

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title or 'ROC Curve - Pidgin Hate Speech Detection')
        plt.legend(loc="lower right")
        filename = 'roc_curve.png'

    elif plot_type == 'pr':
        # Precision-Recall Curve
        if isinstance(y_pred[0], (list, np.ndarray)) and len(y_pred[0]) > 1:
            y_scores = np.array(y_pred)[:, 1]
        elif hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
            y_scores = y_pred[:, 1]
        else:
            y_scores = y_pred

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title or 'Precision-Recall Curve - Pidgin Hate Speech Detection')
        plt.legend(loc="lower left")
        filename = 'pr_curve.png'

    elif plot_type == 'dist':
        # Label Distribution
        unique, counts = np.unique(y_true, return_counts=True)

        plt.bar(range(len(unique)), counts, color=['lightblue', 'orange'])
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title(title or 'Label Distribution - Pidgin Hate Speech Dataset')
        plt.xticks(range(len(unique)), labels)

        # Add count annotations
        for i, count in enumerate(counts):
            plt.text(i, count + max(counts) * 0.01, str(count),
                     ha='center', va='bottom')

        filename = 'label_distribution.png'

    else:
        raise ValueError(
            f"Invalid plot_type: {plot_type}. Must be 'cm', 'roc', 'pr', or 'dist'")

    # Save plot
    filepath = os.path.join(save_path, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {filepath}")


def load_pidgin_dict() -> Dict[str, str]:
    """
    Load Pidgin-English translation dictionary.
    Hardcoded dictionary for common Pidgin expressions.

    Returns:
        Dict[str, str]: Pidgin to English translation mapping
    """
    return {
        # Greetings and common expressions
        'abeg': 'please',
        'chale': 'friend',
        'wale': 'friend',
        'charley': 'friend',
        'oya': 'come on',
        'jare': 'indeed',
        'ehn': 'yes',

        # Verbs and actions
        'chop': 'eat',
        'sabi': 'know',
        'gimme': 'give me',
        'dey': 'is/are',
        'go': 'will',
        'fit': 'can',
        'wan': 'want',
        'come': 'come',
        'carry': 'take',

        # Nouns
        'pikin': 'child',
        'wahala': 'trouble',
        'money': 'money',
        'chop money': 'bribe',
        'kpalongo': 'completely',
        'trotro': 'minibus',
        'obruni': 'white person',

        # Pronouns and determiners
        'im': 'he/she/it',
        'dem': 'they/them',
        'una': 'you all',
        'dis': 'this',
        'dat': 'that',
        'wetin': 'what',
        'wey': 'that/which',

        # Adjectives and adverbs
        'fine': 'good/beautiful',
        'well': 'good',
        'small': 'little',
        'big': 'large',
        'sharp': 'smart',
        'dull': 'slow',

        # Particles and connectors
        'na': 'is/it is',
        'no': 'not/don\'t',
        'nor': 'don\'t',
        'abi': 'right?',
        'se': 'that',
        'make': 'let',
        'for': 'to/for',
        'with': 'with'
    }


def code_mix_ratio(text: str) -> float:
    """
    Calculate code-mixing ratio between Pidgin and English.
    Implementation stub following VocalTweets (Yusuf et al., 2024) approach.

    Args:
        text (str): Input text to analyze

    Returns:
        float: Ratio of English words to total words (0.0 to 1.0)
    """
    if not text or not text.strip():
        return 0.0

    # Extract all alphabetic words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    if not words:
        return 0.0

    # Common English words for detection
    english_indicators = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
        'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be',
        'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'can', 'may', 'must', 'shall', 'might', 'ought',
        'about', 'above', 'across', 'after', 'against', 'along', 'among',
        'around', 'because', 'before', 'behind', 'below', 'beneath', 'beside',
        'between', 'beyond', 'during', 'except', 'from', 'inside', 'into',
        'like', 'near', 'off', 'outside', 'over', 'since', 'through',
        'throughout', 'till', 'toward', 'under', 'until', 'up', 'upon',
        'within', 'without', 'people', 'time', 'way', 'day', 'man', 'thing',
        'woman', 'life', 'child', 'world', 'school', 'state', 'family',
        'student', 'group', 'country', 'problem', 'hand', 'part', 'place',
        'case', 'week', 'company', 'system', 'program', 'question', 'work',
        'government', 'number', 'night', 'point', 'home', 'water', 'room',
        'mother', 'area', 'money', 'story', 'fact', 'month', 'lot', 'right',
        'study', 'book', 'eye', 'job', 'word', 'business', 'issue', 'side',
        'kind', 'head', 'house', 'service', 'friend', 'father', 'power',
        'hour', 'game', 'line', 'end', 'member', 'law', 'car', 'city', 'name'
    }

    # Load Pidgin dictionary for Pidgin word detection
    pidgin_dict = load_pidgin_dict()
    pidgin_words = set(pidgin_dict.keys())

    english_count = 0
    pidgin_count = 0

    for word in words:
        if word in english_indicators:
            english_count += 1
        elif word in pidgin_words:
            pidgin_count += 1
        else:
            # For unknown words, try language detection if available
            if LANGDETECT_AVAILABLE:
                try:
                    if detect(word) == 'en':
                        english_count += 1
                except:
                    # If detection fails, assume mixed/unknown
                    pass

    total_classified = english_count + pidgin_count

    if total_classified == 0:
        return 0.5  # Unknown/mixed content

    return english_count / total_classified


def calculate_metrics(y_true: Union[List[int], torch.Tensor], y_pred: Union[List[int], torch.Tensor]) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    Supports PyTorch tensors.

    Args:
        y_true: True labels (list or PyTorch tensor)
        y_pred: Predicted labels (list or PyTorch tensor)

    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Convert PyTorch tensors to numpy if needed
    if TORCH_AVAILABLE and isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if TORCH_AVAILABLE and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro')
    }

    return metrics


def save_classification_report(
    y_true: Union[List[int], torch.Tensor],
    y_pred: Union[List[int], torch.Tensor],
    labels: Optional[List[str]] = None,
    save_path: str = 'reports/'
) -> str:
    """
    Save detailed classification report to file.
    Supports PyTorch tensors.

    Args:
        y_true: True labels (list or PyTorch tensor)
        y_pred: Predicted labels (list or PyTorch tensor)
        labels: Class labels
        save_path: Directory to save report

    Returns:
        str: Path to saved report
    """
    # Convert PyTorch tensors to numpy if needed
    if TORCH_AVAILABLE and isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if TORCH_AVAILABLE and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    os.makedirs(save_path, exist_ok=True)

    if labels is None:
        labels = ['Non-Hate', 'Hate']

    report = classification_report(
        y_true, y_pred,
        target_names=labels,
        digits=4
    )

    filepath = os.path.join(save_path, 'classification_report.txt')

    with open(filepath, 'w') as f:
        f.write("Pidgin Hate Speech Detection - Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)

    print(f"Classification report saved to: {filepath}")
    return filepath


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array safely.

    Args:
        tensor: PyTorch tensor

    Returns:
        np.ndarray: NumPy array
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")

    return tensor.detach().cpu().numpy()


def numpy_to_tensor(array: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    Convert NumPy array to PyTorch tensor.

    Args:
        array: NumPy array
        device: Device to place tensor on ('cpu' or 'cuda')

    Returns:
        torch.Tensor: PyTorch tensor
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")

    return torch.from_numpy(array).to(device)


if __name__ == '__main__':
    # Test the utility functions
    print("Testing Pidgin NLP Utils (PyTorch-focused)\n")

    # Test seed setting
    print("1. Testing set_seed():")
    set_seed(42)
    print()

    # Test dummy metrics plotting
    print("2. Testing plot_metrics():")
    dummy_y_true = [0, 1, 0, 1, 1, 0, 1, 0]
    dummy_y_pred = [0, 1, 0, 0, 1, 1, 1, 0]

    # Test with PyTorch tensors if available
    if TORCH_AVAILABLE:
        print("Testing with PyTorch tensors:")
        y_true_tensor = torch.tensor(dummy_y_true)
        y_pred_tensor = torch.tensor(dummy_y_pred)
        try:
            plot_metrics(y_true_tensor, y_pred_tensor, 'cm')
            plot_metrics(dummy_y_true, dummy_y_pred, 'dist')
            print("Dummy plots saved successfully (PyTorch compatible)")
        except Exception as e:
            print(f"Plot error: {e}")
    else:
        try:
            plot_metrics(dummy_y_true, dummy_y_pred, 'cm')
            plot_metrics(dummy_y_true, dummy_y_pred, 'dist')
            print("Dummy plots saved successfully")
        except Exception as e:
            print(f"Plot error: {e}")
    print()

    # Test Pidgin dictionary
    print("3. Testing load_pidgin_dict():")
    pidgin_dict = load_pidgin_dict()
    print(f"Loaded {len(pidgin_dict)} Pidgin translations")
    print(f"Sample: {dict(list(pidgin_dict.items())[:5])}")
    print()

    # Test code-mixing ratio
    print("4. Testing code_mix_ratio():")
    test_texts = [
        "I dey go school today",  # Mixed
        "Abeg make you chop food",  # Mostly Pidgin
        "The weather is very nice today",  # Mostly English
        "Wetin you dey do for house?"  # Mostly Pidgin
    ]

    for text in test_texts:
        ratio = code_mix_ratio(text)
        print(f"'{text}' -> English ratio: {ratio:.2f}")
    print()

    # Test metrics calculation with PyTorch support
    print("5. Testing calculate_metrics():")
    if TORCH_AVAILABLE:
        print("Testing with PyTorch tensors:")
        y_true_tensor = torch.tensor(dummy_y_true)
        y_pred_tensor = torch.tensor(dummy_y_pred)
        metrics = calculate_metrics(y_true_tensor, y_pred_tensor)
    else:
        metrics = calculate_metrics(dummy_y_true, dummy_y_pred)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    print()

    # Test MLflow (if available)
    print("6. Testing MLflow logging:")
    if MLFLOW_AVAILABLE:
        try:
            run_id = mlflow_logger(
                experiment='test_pidgin_pytorch',
                tags={'model': 'test', 'language': 'pidgin',
                      'framework': 'pytorch'},
                params={'test_param': 'test_value'},
                metrics={'test_accuracy': 0.85}
            )
            print(f"MLflow test successful. Run ID: {run_id}")
        except Exception as e:
            print(f"MLflow error: {e}")
    else:
        print("MLflow not available - skipping test")

    print("\nAll utility tests completed (PyTorch-focused)!")
