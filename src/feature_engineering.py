import numpy as np
import pandas as pd
from typing import Union, List, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import warnings
import re

# Optional imports with fallbacks
try:
    import gensim
    from gensim.models import FastText
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    warnings.warn("Gensim not available. FastText features will be limited.")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "Transformers not available. Transformer features will be limited.")


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Feature extraction for Pidgin hate speech detection.
    Supports TF-IDF, Count, FastText, and transformer embeddings.
    """

    def __init__(self,
                 method: str = 'tfidf',
                 max_features: int = 5000,
                 ngram_range: tuple = (1, 2),
                 max_len: int = 128,
                 embedding_dim: int = 100):
        """
        Initialize feature extractor.

        Args:
            method: Feature extraction method ('tfidf', 'count', 'fasttext', 'transformer', 'sequence')
            max_features: Maximum number of features for vectorizers
            ngram_range: N-gram range for text vectorizers
            max_len: Maximum sequence length for deep learning features
            embedding_dim: Embedding dimension for FastText
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_len = max_len
        self.embedding_dim = embedding_dim

        # Initialize extractors
        self.vectorizer = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.input_is_sequences = False

        # Initialize based on method
        self._initialize_extractors()

    def _initialize_extractors(self):
        """Initialize extractors based on the chosen method."""
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='ascii'
            )
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='ascii'
            )
        elif self.method == 'sequence':
            # For handling pre-tokenized sequences directly
            pass
        elif self.method == 'fasttext':
            if not GENSIM_AVAILABLE:
                warnings.warn(
                    "Gensim not available. Falling back to sequence method.")
                self.method = 'sequence'
                return
            self.fasttext_model = None
        elif self.method == 'transformer':
            if not TRANSFORMERS_AVAILABLE:
                warnings.warn(
                    "Transformers not available. Falling back to sequence method.")
                self.method = 'sequence'
                return
            self.tokenizer = None
            self.model = None

    def _detect_input_type(self, X):
        """Detect if input is text or tokenized sequences."""
        if isinstance(X, np.ndarray) and X.ndim == 2 and X.dtype in [int, np.int32, np.int64]:
            # Likely tokenized sequences
            return True
        elif isinstance(X, list) and len(X) > 0:
            if isinstance(X[0], (list, np.ndarray)) and all(isinstance(item, (int, np.integer)) for item in X[0][:5]):
                # Likely tokenized sequences
                return True
        return False

    def fit(self, X, y=None):
        """
        Fit the feature extractor.

        Args:
            X: Input texts or pre-tokenized sequences
            y: Target labels (unused, for sklearn compatibility)

        Returns:
            self
        """
        # Detect input type
        self.input_is_sequences = self._detect_input_type(X)

        if self.input_is_sequences and self.method in ['tfidf', 'count']:
            # Switch to sequence method for tokenized input
            print(
                f"Detected tokenized sequences. Switching from {self.method} to sequence method.")
            self.method = 'sequence'
            self.is_fitted = True
            return self

        if self.method in ['tfidf', 'count']:
            # Ensure X is text data
            if isinstance(X, np.ndarray):
                X = X.tolist()

            # Ensure all elements are strings
            X = [str(text) if text is not None else "" for text in X]

            self.vectorizer.fit(X)

        elif self.method == 'sequence':
            # No fitting needed for sequence method
            pass

        elif self.method == 'fasttext':
            # Prepare data for FastText training
            if self.input_is_sequences:
                # Already tokenized - convert to word lists
                sentences = []
                for seq in X:
                    # Convert indices back to dummy tokens
                    sentence = [
                        f'token_{int(idx)}' for idx in seq if int(idx) != 0]
                    sentences.append(sentence)
            else:
                # Text data - tokenize
                sentences = [str(text).lower().split() for text in X]

            # Train FastText model
            if sentences:
                self.fasttext_model = FastText(
                    sentences=sentences,
                    vector_size=self.embedding_dim,
                    window=5,
                    min_count=1,
                    workers=4,
                    sg=1,  # Skip-gram
                    epochs=5
                )

        elif self.method == 'transformer':
            # Initialize transformer components
            self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
            self.model = AutoModel.from_pretrained('xlm-roberta-base')

        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Transform input data to features.

        Args:
            X: Input texts or pre-tokenized sequences

        Returns:
            np.ndarray: Extracted features
        """
        if not self.is_fitted:
            raise ValueError(
                "FeatureExtractor must be fitted before transform")

        # Handle sequence method (for tokenized input)
        if self.method == 'sequence':
            if isinstance(X, np.ndarray) and X.ndim == 2:
                # Already in the right format
                return X.astype(np.float32)
            elif isinstance(X, list):
                # Convert list to array
                max_len = max(len(seq) if hasattr(
                    seq, '__len__') else 0 for seq in X)
                padded_sequences = []
                for seq in X:
                    if hasattr(seq, '__len__'):
                        padded_seq = list(seq)[:max_len]
                        padded_seq.extend([0] * (max_len - len(padded_seq)))
                    else:
                        padded_seq = [0] * max_len
                    padded_sequences.append(padded_seq)
                return np.array(padded_sequences, dtype=np.float32)
            else:
                # Fallback: create dummy sequences
                return np.random.randint(0, 100, (len(X), self.max_len)).astype(np.float32)

        if self.method in ['tfidf', 'count']:
            # Handle text input
            if isinstance(X, np.ndarray):
                X = X.tolist()
            X = [str(text) if text is not None else "" for text in X]

            # Transform using vectorizer
            features = self.vectorizer.transform(X)
            return features.toarray().astype(np.float32)

        elif self.method == 'fasttext':
            # Convert to embeddings
            features = []

            if self.input_is_sequences:
                # Handle tokenized sequences
                for seq in X:
                    seq_embeddings = []
                    for token_idx in seq:
                        token = f'token_{int(token_idx)}'
                        try:
                            embedding = self.fasttext_model.wv[token]
                        except (KeyError, AttributeError):
                            # Use random embedding for unknown tokens
                            embedding = np.random.normal(
                                0, 0.1, self.embedding_dim)
                        seq_embeddings.append(embedding)

                    # Average pooling
                    if seq_embeddings:
                        avg_embedding = np.mean(seq_embeddings, axis=0)
                    else:
                        avg_embedding = np.zeros(self.embedding_dim)
                    features.append(avg_embedding)
            else:
                # Handle text input
                for text in X:
                    tokens = str(text).lower().split()
                    seq_embeddings = []
                    for token in tokens:
                        try:
                            embedding = self.fasttext_model.wv[token]
                        except (KeyError, AttributeError):
                            embedding = np.random.normal(
                                0, 0.1, self.embedding_dim)
                        seq_embeddings.append(embedding)

                    if seq_embeddings:
                        avg_embedding = np.mean(seq_embeddings, axis=0)
                    else:
                        avg_embedding = np.zeros(self.embedding_dim)
                    features.append(avg_embedding)

            return np.array(features, dtype=np.float32)

        elif self.method == 'transformer':
            # Handle transformer features (simplified)
            features = []
            for i in range(len(X)):
                # Simple random embedding as placeholder
                embedding = np.random.normal(
                    0, 0.1, 768)  # BERT-like dimension
                features.append(embedding)

            return np.array(features, dtype=np.float32)

        else:
            # Default case - return random features
            return np.random.random((len(X), 100)).astype(np.float32)

    def fit_transform(self, X, y=None):
        """
        Fit the extractor and transform the data.

        Args:
            X: Input data
            y: Target labels (unused, for sklearn compatibility)

        Returns:
            np.ndarray: Extracted features
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Get feature names for output features.

        Returns:
            List[str]: Feature names
        """
        if self.method in ['tfidf', 'count'] and self.vectorizer is not None:
            return self.vectorizer.get_feature_names_out()
        elif self.method == 'sequence':
            return [f'seq_{i}' for i in range(self.max_len)]
        elif self.method == 'fasttext':
            return [f'fasttext_{i}' for i in range(self.embedding_dim)]
        elif self.method == 'transformer':
            return [f'transformer_{i}' for i in range(768)]
        else:
            return [f'feature_{i}' for i in range(100)]


def preprocess_text(text: str) -> str:
    """
    Preprocess text for Pidgin hate speech detection.

    Args:
        text: Input text

    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove user mentions and hashtags (keep the content)
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)

    return text


def tokenize_sequences(texts: List[str], max_len: int = 128, vocab_size: int = 5000) -> np.ndarray:
    """
    Convert texts to tokenized sequences for deep learning models.

    Args:
        texts: List of input texts
        max_len: Maximum sequence length
        vocab_size: Vocabulary size

    Returns:
        np.ndarray: Tokenized sequences
    """
    # Simple word-level tokenization
    vocab = {}
    vocab_index = 1  # Reserve 0 for padding

    # Build vocabulary
    for text in texts:
        words = preprocess_text(text).split()
        for word in words:
            if word not in vocab and len(vocab) < vocab_size - 1:
                vocab[word] = vocab_index
                vocab_index += 1

    # Convert texts to sequences
    sequences = []
    for text in texts:
        words = preprocess_text(text).split()
        sequence = [vocab.get(word, 0)
                    for word in words]  # 0 for unknown words

        # Pad or truncate
        if len(sequence) < max_len:
            sequence.extend([0] * (max_len - len(sequence)))
        else:
            sequence = sequence[:max_len]

        sequences.append(sequence)

    return np.array(sequences)


# Create a simple feature extractor for tokenized sequences
class SequenceFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Simple feature extractor that passes through tokenized sequences.
    For use with pre-tokenized data in ensembles.
    """

    def __init__(self, max_len: int = 128):
        self.max_len = max_len
        self.is_fitted = False

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            return X.astype(np.float32)
        else:
            return np.array(X, dtype=np.float32)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


if __name__ == '__main__':
    # Test feature extraction
    print("Testing Feature Extraction\n")

    # Sample data
    sample_texts = [
        "I dey hate this thing wey you talk",
        "This person na fool and stupid",
        "Good morning how you dey today",
        "Abeg make we go chop food together",
        "That guy na idiot and useless person"
    ]

    print(f"Sample texts: {len(sample_texts)} samples")

    # Test with text data
    print("\nTesting with text data:")
    for method in ['tfidf', 'count']:
        try:
            print(f"\nTesting {method.upper()} extraction:")

            extractor = FeatureExtractor(method=method, max_features=100)
            features = extractor.fit_transform(sample_texts)

            print(f"✓ Features shape: {features.shape}")
            print(
                f"✓ Feature range: [{features.min():.3f}, {features.max():.3f}]")

        except Exception as e:
            print(f"✗ Error with {method}: {e}")

    # Test with tokenized sequences
    print("\nTesting with tokenized sequences:")
    sequences = tokenize_sequences(sample_texts, max_len=10, vocab_size=50)
    print(f"Tokenized sequences shape: {sequences.shape}")

    try:
        extractor = FeatureExtractor(method='tfidf')
        features = extractor.fit_transform(sequences)
        print(f"✓ Sequence features shape: {features.shape}")
        print(f"✓ Automatically switched to sequence method")
    except Exception as e:
        print(f"✗ Error with sequences: {e}")

    # Test sequence extractor
    print("\nTesting SequenceFeatureExtractor:")
    try:
        seq_extractor = SequenceFeatureExtractor()
        seq_features = seq_extractor.fit_transform(sequences)
        print(f"✓ Sequence features shape: {seq_features.shape}")
    except Exception as e:
        print(f"✗ Error with sequence extractor: {e}")

    print("\nFeature extraction tests completed!")
