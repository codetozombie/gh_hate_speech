import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Union, Optional, Dict, Any
import os
import sys
import warnings
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    warnings.warn(
        "SMOTE not available. Install with: pip install imbalanced-learn")

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available for model logging")

# Handle imports for both relative and direct execution
try:
    from ..utils import set_seed, mlflow_logger
except ImportError:
    # Add parent directory to path for direct execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from utils import set_seed, mlflow_logger


class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM for Pidgin hate speech detection.
    Based on AfriHate (Muhammad et al., 2025): BiLSTM on Pidgin (F1~0.78)
    and NaijaHate (Osei et al., 2024): BiLSTM for tweets (~0.16% hate).
    """

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 100,
                 hidden_dim: int = 128, num_classes: int = 2,
                 dropout: float = 0.2, num_layers: int = 1):
        super(BiLSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # *2 for bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = self.dropout(embedded)

        # BiLSTM
        lstm_out, (hidden, cell) = self.bilstm(embedded)
        # Take the last output (considering both directions)
        output = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)

        output = self.dropout(output)
        output = self.fc(output)  # (batch_size, num_classes)

        return output


class CNNModel(nn.Module):
    """
    1D CNN for Pidgin hate speech detection.
    Based on Yusuf et al. (2025, arXiv:2311.10541v2): CNN (F1=0.84 Engausa).
    """

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 100,
                 num_filters: int = 128, filter_sizes: list = [3, 4, 5],
                 num_classes: int = 2, dropout: float = 0.2):
        super(CNNModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # Multiple filter sizes for different n-gram features
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = self.dropout(embedded)

        # Transpose for Conv1d: (batch_size, embed_dim, seq_len)
        embedded = embedded.transpose(1, 2)

        # Apply multiple convolutions
        conv_outputs = []
        for conv in self.convs:
            # (batch_size, num_filters, new_seq_len)
            conv_out = F.relu(conv(embedded))
            # Global max pooling
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)

        # Concatenate all conv outputs
        # (batch_size, len(filter_sizes) * num_filters)
        output = torch.cat(conv_outputs, dim=1)
        output = self.dropout(output)
        output = self.fc(output)  # (batch_size, num_classes)

        return output


class LSTMModel(nn.Module):
    """
    Basic LSTM baseline for Pidgin hate speech detection.
    """

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 100,
                 hidden_dim: int = 128, num_classes: int = 2,
                 dropout: float = 0.2, num_layers: int = 1):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = self.dropout(embedded)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Take the last output
        output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        output = self.dropout(output)
        output = self.fc(output)  # (batch_size, num_classes)

        return output


class HybridCNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM with attention for Pidgin hate speech detection.
    Based on Abdullahi et al. (2024, ACL WOAH): CNN-LSTM hybrid (F1=0.80 Hausa)
    and Tesfay et al. (2025, arXiv:2505.12116): BiLSTM/CNN multi-task (F1=0.86).
    """

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 100,
                 num_filters: int = 64, filter_sizes: list = [3, 4, 5],
                 hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.2):
        super(HybridCNNLSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # CNN component - Use same padding to maintain sequence length
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs, padding=fs//2)
            for fs in filter_sizes
        ])

        # BiLSTM component
        cnn_output_dim = len(filter_sizes) * num_filters
        self.bilstm = nn.LSTM(
            cnn_output_dim, hidden_dim,
            batch_first=True, bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Final classification layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = self.dropout(embedded)

        # CNN component
        # Transpose for Conv1d: (batch_size, embed_dim, seq_len)
        conv_input = embedded.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            # (batch_size, num_filters, seq_len_out)
            conv_out = F.relu(conv(conv_input))
            # Ensure all conv outputs have the same sequence length
            if conv_out.size(2) != seq_len:
                # Trim or pad to match original sequence length
                if conv_out.size(2) > seq_len:
                    conv_out = conv_out[:, :, :seq_len]
                else:
                    padding_needed = seq_len - conv_out.size(2)
                    conv_out = F.pad(conv_out, (0, padding_needed))
            conv_outputs.append(conv_out)

        # Concatenate conv outputs and transpose back
        # (batch_size, total_filters, seq_len)
        cnn_out = torch.cat(conv_outputs, dim=1)
        # (batch_size, seq_len, total_filters)
        cnn_out = cnn_out.transpose(1, 2)

        # BiLSTM component
        # (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.bilstm(cnn_out)

        # Attention mechanism
        attention_weights = F.softmax(self.attention(
            lstm_out), dim=1)  # (batch_size, seq_len, 1)
        # (batch_size, hidden_dim * 2)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)

        # Final classification
        output = self.dropout(attended_output)
        output = self.fc(output)  # (batch_size, num_classes)

        return output


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Implementation stub for variant 2 models.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_dl_model(
    name: str,
    variant: int = 1,
    vocab_size: int = 5000,
    max_len: int = 128,
    embed_dim: int = 100,
    num_classes: int = 2
) -> nn.Module:
    """
    Factory function to get deep learning models for Pidgin hate speech detection.

    Args:
        name (str): Model name ('bilstm', 'cnn', 'lstm', 'hybrid')
        variant (int): Model variant (1=base, 2=imbalance, 3=hybrid/advanced)
        vocab_size (int): Vocabulary size
        max_len (int): Maximum sequence length
        embed_dim (int): Embedding dimension
        num_classes (int): Number of output classes

    Returns:
        nn.Module: PyTorch model

    Raises:
        ValueError: If model name or variant is invalid
    """

    if name not in ['bilstm', 'cnn', 'lstm', 'hybrid']:
        raise ValueError(
            f"Invalid model name: {name}. Must be one of ['bilstm', 'cnn', 'lstm', 'hybrid']")

    if variant not in [1, 2, 3]:
        raise ValueError(
            f"Invalid variant: {variant}. Must be 1 (base), 2 (imbalance), or 3 (advanced)")

    # Base parameters
    base_params = {
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'num_classes': num_classes,
        'dropout': 0.2
    }

    # Variant-specific adjustments
    if variant == 2:
        # Imbalance handling variant
        base_params['dropout'] = 0.3  # Higher dropout for regularization
    elif variant == 3:
        # Advanced/hybrid variant
        base_params['dropout'] = 0.1  # Lower dropout for complex models

    if name == 'bilstm':
        # Bidirectional LSTM as in AfriHate (Muhammad et al., 2025)
        model_params = {**base_params, 'hidden_dim': 128, 'num_layers': 1}
        if variant == 3:
            model_params['hidden_dim'] = 256
            model_params['num_layers'] = 2
        model = BiLSTMModel(**model_params)

    elif name == 'cnn':
        # CNN as in Yusuf et al. (2025, arXiv:2311.10541v2)
        model_params = {**base_params,
                        'num_filters': 128, 'filter_sizes': [3, 4, 5]}
        if variant == 3:
            model_params['num_filters'] = 256
            model_params['filter_sizes'] = [2, 3, 4, 5]
        model = CNNModel(**model_params)

    elif name == 'lstm':
        # Basic LSTM baseline
        model_params = {**base_params, 'hidden_dim': 128, 'num_layers': 1}
        if variant == 3:
            model_params['hidden_dim'] = 256
            model_params['num_layers'] = 2
        model = LSTMModel(**model_params)

    elif name == 'hybrid':
        # Hybrid CNN-LSTM as in Abdullahi et al. (2024) for African code-mix
        model_params = {
            **base_params,
            'num_filters': 64,
            'filter_sizes': [3, 4, 5],
            'hidden_dim': 128
        }
        if variant == 3:
            model_params['num_filters'] = 128
            model_params['filter_sizes'] = [2, 3, 4, 5, 6]
            model_params['hidden_dim'] = 256
        model = HybridCNNLSTMModel(**model_params)

    return model


def train_dl_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    val_split: float = 0.1,
    class_weights: Optional[Dict[int, float]] = None,
    use_focal_loss: bool = False,
    patience: int = 3,
    device: str = 'cpu',
    save_path: Optional[str] = None,
    experiment_name: str = 'pidgin_hate_dl'
) -> Dict[str, Any]:
    """
    Train deep learning model for Pidgin hate speech detection.

    Args:
        model: PyTorch model to train
        X_train: Training features (tokenized sequences)
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        val_split: Validation split ratio if X_val not provided
        class_weights: Class weights for imbalanced data
        use_focal_loss: Whether to use focal loss
        patience: Early stopping patience
        device: Device to train on ('cpu' or 'cuda')
        save_path: Path to save trained model
        experiment_name: MLflow experiment name

    Returns:
        Dict[str, Any]: Training history and metrics
    """

    # Set seed for reproducibility
    set_seed(42)

    # Move model to device
    model = model.to(device)

    # Prepare data
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_split, random_state=42, stratify=y_train
        )

    # Apply SMOTE if needed (for imbalanced data)
    if SMOTE_AVAILABLE and len(np.unique(y_train)) == 2:
        # Reshape for SMOTE (flatten sequence data)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        smote = SMOTE(random_state=42)
        X_train_flat, y_train = smote.fit_resample(X_train_flat, y_train)
        X_train = X_train_flat.reshape(-1, X_train.shape[1])
        print(f"Applied SMOTE. New training size: {len(X_train)}")

    # Convert to tensors
    X_train_tensor = torch.LongTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.LongTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    else:
        if class_weights:
            weights = torch.FloatTensor(
                [class_weights[i] for i in range(len(class_weights))]).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Log to MLflow if available
    if MLFLOW_AVAILABLE:
        try:
            final_metrics = {
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'final_train_acc': history['train_acc'][-1],
                'final_val_acc': history['val_acc'][-1],
                'best_val_loss': best_val_loss
            }

            mlflow_logger(
                experiment=experiment_name,
                tags={'model_type': 'deep_learning', 'framework': 'pytorch'},
                params={'epochs': epochs, 'batch_size': batch_size,
                        'learning_rate': learning_rate},
                metrics=final_metrics,
                model=model
            )
        except Exception as e:
            print(f"MLflow logging failed: {e}")

    return history


if __name__ == '__main__':
    # Test the deep learning models
    print("Testing Deep Learning Models for Pidgin Hate Speech\n")

    # Test model creation
    print("1. Testing model creation:")

    models_to_test = ['bilstm', 'cnn', 'lstm', 'hybrid']

    for model_name in models_to_test:
        for variant in [1, 2, 3]:
            try:
                model = get_dl_model(model_name, variant,
                                     vocab_size=1000, max_len=64)
                print(
                    f"✓ {model_name.upper()} variant {variant}: {model.__class__.__name__}")

                # Test forward pass
                # batch_size=2, seq_len=64
                dummy_input = torch.randint(0, 1000, (2, 64))
                with torch.no_grad():
                    output = model(dummy_input)
                print(f"  Output shape: {output.shape}")

            except Exception as e:
                print(f"✗ {model_name.upper()} variant {variant}: {e}")

    print("\n2. Testing model summary:")
    # Test a specific model
    model = get_dl_model('bilstm', 1, vocab_size=5000, max_len=128)
    print(f"BiLSTM Model Architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n3. Testing training setup:")
    # Create dummy data
    X_dummy = np.random.randint(0, 5000, (100, 128))
    y_dummy = np.random.randint(0, 2, (100,))

    print(f"Dummy data shapes: X={X_dummy.shape}, y={y_dummy.shape}")

    # Test training (short run)
    try:
        history = train_dl_model(
            model, X_dummy, y_dummy,
            epochs=2, batch_size=16,
            save_path='data/models/test_bilstm.pth'
        )
        print(
            f"Training completed. Final validation accuracy: {history['val_acc'][-1]:.4f}")
    except Exception as e:
        print(f"Training test failed: {e}")

    print("\nDeep learning model tests completed!")
