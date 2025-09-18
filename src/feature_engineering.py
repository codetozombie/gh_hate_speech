import numpy as np
import pandas as pd
import joblib
import re
import os
from typing import Tuple, Union, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from collections import Counter
import warnings

# Optional imports for advanced features
try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pad_sequence
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Embedding features will be limited.")

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. Tokenizer features will be limited.")

try:
    from gensim.models import FastText
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    warnings.warn("Gensim not available. FastText features will be limited.")

from data_preprocessing import load_splits, clean_text

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Feature extractor for Pidgin hate speech detection.
    Supports TF-IDF, BoW, embeddings, and transformer tokenization.
    """

    def __init__(self, feature_type: str = 'tfidf', **kwargs):
        """
        Initialize feature extractor.

        Args:
            feature_type: Type of features ('tfidf', 'bow', 'embed', 'tokenize')
            **kwargs: Additional parameters for specific extractors
        """
        self.feature_type = feature_type
        self.kwargs = kwargs
        self.extractor = None
        self.vocab = None
        self.word_to_idx = None
        self.fitted = False

    def fit(self, X_train: list):
        """
        Fit the feature extractor on training data.

        Args:
            X_train: List of training texts
        """
        if self.feature_type == 'tfidf':
            # TF-IDF bigrams for Pidgin slang as in VocalTweets (Yusuf et al., 2024)
            self.extractor = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=self.kwargs.get('max_features', 5000),
                min_df=self.kwargs.get('min_df', 2),
                stop_words=None,  # Keep Pidgin stopwords
                lowercase=True,
                token_pattern=r'\b\w+\b'
            )
            self.extractor.fit(X_train)

        elif self.feature_type == 'bow':
            # BoW alternative for code-mix as in AfriHate (Muhammad et al., 2025)
            self.extractor = CountVectorizer(
                ngram_range=(1, 2),
                max_features=self.kwargs.get('max_features', 5000),
                min_df=self.kwargs.get('min_df', 2),
                stop_words=None,
                lowercase=True,
                token_pattern=r'\b\w+\b'
            )
            self.extractor.fit(X_train)

        elif self.feature_type == 'embed':
            # PyTorch Embedding preparation for DL as in EkoHate (Oladipo et al., 2024)
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch required for embedding features")

            # Build vocabulary from training data
            tokenized_texts = [nltk.word_tokenize(text.lower()) for text in X_train]
            all_words = [word for tokens in tokenized_texts for word in tokens]
            word_counts = Counter(all_words)

            vocab_size = self.kwargs.get('vocab_size', 5000)
            most_common = word_counts.most_common(vocab_size - 2)  # Reserve 2 for special tokens

            self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
            for word, _ in most_common:
                self.word_to_idx[word] = len(self.word_to_idx)
            
            self.vocab = self.word_to_idx
            self.vocab_size = len(self.word_to_idx)

        elif self.feature_type == 'tokenize':
            # HuggingFace tokenizer for transformers as in AfriHate
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers required for tokenizer features")

            model_name = self.kwargs.get('model_name', 'masakhane.io/afriberta-base')
            try:
                self.extractor = AutoTokenizer.from_pretrained(model_name)
            except:
                # Fallback to a more common model if masakhane is not available
                self.extractor = AutoTokenizer.from_pretrained('bert-base-uncased')
                warnings.warn(f"Could not load {model_name}, using bert-base-uncased as fallback")

        self.fitted = True
        return self

    def transform(self, X: list):
        """
        Transform texts to features.

        Args:
            X: List of texts to transform

        Returns:
            Transformed features
        """
        if not self.fitted:
            raise ValueError("Extractor must be fitted before transform")

        if self.feature_type in ['tfidf', 'bow']:
            return self.extractor.transform(X)

        elif self.feature_type == 'embed':
            max_len = self.kwargs.get('max_len', 128)
            sequences = []
            
            for text in X:
                tokens = nltk.word_tokenize(text.lower())
                # Convert tokens to indices
                indices = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
                # Truncate or pad sequences
                if len(indices) > max_len:
                    indices = indices[:max_len]
                else:
                    indices.extend([self.word_to_idx['<PAD>']] * (max_len - len(indices)))
                sequences.append(indices)
            
            return np.array(sequences)

        elif self.feature_type == 'tokenize':
            max_len = self.kwargs.get('max_len', 128)
            return self.extractor(
                X,
                return_tensors='pt',
                padding='max_length',
                max_length=max_len,
                truncation=True
            )

    def fit_transform(self, X: list):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


def get_pipeline(feature_type: str, **kwargs) -> FeatureExtractor:
    """
    Get a feature extraction pipeline.

    Args:
        feature_type: Type of features to extract
        **kwargs: Additional parameters

    Returns:
        FeatureExtractor instance
    """
    return FeatureExtractor(feature_type=feature_type, **kwargs)


def calculate_code_mix_ratio(text: str) -> float:
    """
    Calculate code-mixing ratio (English words / total words).
    Stub implementation for Pidgin-English code-switching analysis.

    Args:
        text: Input text

    Returns:
        Ratio of English words to total words
    """
    # Simple heuristic: common English words pattern
    english_pattern = r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b'
    english_matches = len(re.findall(english_pattern, text.lower()))
    total_words = len(text.split())

    return english_matches / max(total_words, 1)


def get_features(
    feat_type: str = 'tfidf',
    split: str = 'train',
    base_path: str = 'data/processed/',
    max_len: int = 128,
    vocab_size: int = 5000
) -> Tuple[Union[np.ndarray, torch.Tensor, Any], np.ndarray, Any]:
    """
    Extract features for hate speech detection.

    Args:
        feat_type: Feature type ('tfidf', 'bow', 'embed', 'tokenize')
        split: Data split to load ('train', 'val', 'test')
        base_path: Base path to processed data
        max_len: Maximum sequence length for embeddings/tokenization
        vocab_size: Vocabulary size for embeddings

    Returns:
        Tuple of (features, labels, fitted_extractor)
    """
    # Load data splits
    train_df, val_df, test_df = load_splits(base_path)

    # Select appropriate split
    if split == 'train':
        df = train_df
    elif split == 'val':
        df = val_df
    elif split == 'test':
        df = test_df
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

    # Clean texts
    texts = [clean_text(text) for text in df['text'].tolist()]
    labels = df['label'].values

    # Check if fitted extractor exists
    fitted_path = os.path.join(base_path, f'{feat_type}_fitted.pkl')

    if split == 'train':
        # Fit new extractor on training data
        extractor = get_pipeline(
            feat_type,
            max_features=5000 if feat_type in ['tfidf', 'bow'] else None,
            max_len=max_len,
            vocab_size=vocab_size
        )

        features = extractor.fit_transform(texts)

        # Save fitted extractor
        os.makedirs(base_path, exist_ok=True)
        joblib.dump(extractor, fitted_path)

    else:
        # Load fitted extractor and transform
        if not os.path.exists(fitted_path):
            raise FileNotFoundError(
                f"Fitted extractor not found at {fitted_path}. "
                "Please run with split='train' first to fit the extractor."
            )

        extractor = joblib.load(fitted_path)
        features = extractor.transform(texts)

    return features, labels, extractor


class PidginEmbedding(nn.Module):
    """
    PyTorch embedding layer for Pidgin hate speech detection.
    Implementation for deep learning models as referenced in EkoHate (Oladipo et al., 2024).
    """
    
    def __init__(self, vocab_size: int = 5000, embedding_dim: int = 100, padding_idx: int = 0):
        """
        Initialize the embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            padding_idx: Index for padding token
        """
        super(PidginEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """Forward pass through embedding layer."""
        return self.dropout(self.embedding(x))


def create_embedding_layer(vocab_size: int = 5000, embedding_dim: int = 100) -> PidginEmbedding:
    """
    Create a trainable PyTorch embedding layer for deep learning models.
    Implementation as referenced in EkoHate (Oladipo et al., 2024).

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings

    Returns:
        PyTorch Embedding layer
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch required for embedding layer")

    return PidginEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim)


def extract_fasttext_features(texts: list, model_path: str = None) -> np.ndarray:
    """
    Extract FastText embeddings for code-mixed texts.
    Implementation stub as referenced in AfriHate (Muhammad et al., 2025).

    Args:
        texts: List of texts
        model_path: Path to pre-trained FastText model

    Returns:
        FastText feature matrix
    """
    if not GENSIM_AVAILABLE:
        raise ImportError("Gensim required for FastText features")

    if model_path and os.path.exists(model_path):
        model = FastText.load(model_path)
    else:
        # Train a simple FastText model on the texts
        tokenized_texts = [text.lower().split() for text in texts]
        model = FastText(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

    # Get document vectors (average of word vectors)
    features = []
    for text in texts:
        words = text.lower().split()
        if words:
            word_vectors = [model.wv[word] for word in words if word in model.wv]
            if word_vectors:
                features.append(np.mean(word_vectors, axis=0))
            else:
                features.append(np.zeros(model.vector_size))
        else:
            features.append(np.zeros(model.vector_size))

    return np.array(features)


def texts_to_tensor(texts: list, word_to_idx: dict, max_len: int = 128) -> torch.Tensor:
    """
    Convert texts to PyTorch tensor for deep learning models.
    
    Args:
        texts: List of texts
        word_to_idx: Word to index mapping
        max_len: Maximum sequence length
        
    Returns:
        PyTorch tensor of shape (batch_size, max_len)
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch required for tensor conversion")
    
    sequences = []
    for text in texts:
        tokens = nltk.word_tokenize(text.lower())
        indices = [word_to_idx.get(token, word_to_idx.get('<UNK>', 1)) for token in tokens]
        
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices.extend([word_to_idx.get('<PAD>', 0)] * (max_len - len(indices)))
        
        sequences.append(torch.tensor(indices))
    
    return torch.stack(sequences)


if __name__ == '__main__':
    # Test the feature extraction pipeline
    print("Testing TF-IDF features...")
    try:
        X, y, vec = get_features('tfidf', 'train')
        print(f"TF-IDF features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Feature type: {type(X)}")

        # Test other splits
        X_val, y_val, _ = get_features('tfidf', 'val')
        print(f"Validation TF-IDF features shape: {X_val.shape}")

        # Test other feature types
        print("\nTesting BoW features...")
        X_bow, y_bow, vec_bow = get_features('bow', 'train')
        print(f"BoW features shape: {X_bow.shape}")

        if PYTORCH_AVAILABLE:
            print("\nTesting embedding features...")
            X_embed, y_embed, vec_embed = get_features('embed', 'train')
            print(f"Embedding features shape: {X_embed.shape}")
            
            # Test PyTorch embedding layer
            embedding_layer = create_embedding_layer(vocab_size=5000, embedding_dim=100)
            print(f"PyTorch embedding layer created: {embedding_layer}")
            
            # Convert to tensor and test
            if hasattr(vec_embed, 'word_to_idx'):
                sample_texts = ["This is a test", "Another sample text"]
                tensor_data = texts_to_tensor(sample_texts, vec_embed.word_to_idx)
                print(f"Tensor shape: {tensor_data.shape}")
                
                # Test forward pass
                embedded = embedding_layer(tensor_data)
                print(f"Embedded shape: {embedded.shape}")

        if TRANSFORMERS_AVAILABLE:
            print("\nTesting tokenizer features...")
            X_token, y_token, vec_token = get_features('tokenize', 'train')
            print(f"Tokenizer features shape: {X_token['input_ids'].shape}")

        print("\nFeature extraction tests completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()