import pandas as pd
import numpy as np
import re
import nltk
import random
from typing import Dict, List, Set, Tuple, Union, Optional
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Hardcoded dictionaries and sets for Pidgin preprocessing
SPELLING_DICT = {
    'de': 'dey', 'di': 'dey', 'se': 'say', 'dis': 'dis', 'ds': 'dis',
    'dat': 'dat', 'da': 'dat', 'dem': 'dem', 'dm': 'dem', 'im': 'im',
    'm': 'im', 'una': 'una', 'unu': 'una', 'wetin': 'wetin', 'wetn': 'wetin',
    'howfa': 'howfa', 'hfa': 'howfa', 'pikin': 'pikin', 'pkn': 'pikin',
    'wahala': 'wahala', 'whla': 'wahala', 'chop': 'chop', 'chp': 'chop',
    'sabi': 'sabi', 'sb': 'sabi', 'gimme': 'gimme', 'gm': 'gimme',
    'nor': 'no', 'no': 'no', 'abi': 'abi', 'ab': 'abi', 'oya': 'oya',
    'ya': 'oya', 'jare': 'jare', 'jr': 'jare', 'ehn': 'ehn', 'hn': 'ehn',
    'kpalongo': 'kpalongo', 'kpa': 'kpalongo', 'charley': 'charley',
    'chali': 'charley', 'wale': 'wale', 'wl': 'wale', 'barb': 'barb',
    'brb': 'barb', 'nibbies': 'nibbies', 'nb': 'nibbies', 'figa': 'figa',
    'fg': 'figa', 'rydee': 'rydee', 'ryd': 'rydee', 'hung': 'hung',
    'hg': 'hung', 'tiya': 'tiya', 'ty': 'tiya', 'obruni': 'obruni',
    'ob': 'obruni', 'tro-tro': 'trotro', 'tt': 'trotro', 'dey touch': 'dey touch',
    'dt': 'dey touch'
}

STOPWORDS_SET = {
    'a', 'de', 'dey', 'for', 'from', 'go', 'i', 'im', 'na', 'no', 'of',
    'on', 'one', 'or', 'she', 'so', 'to', 'we', 'wetin', 'abi', 'ehn',
    'nor', 'o'
}

PRESERVE_SET = {
    'chale', 'wate', 'abeg', 'wale', 'barb', 'chop money', 'kpalongo',
    'obruni', 'trotro', 'wahala', 'sabi', 'pikin', 'figa', 'hung', 'tiya',
    'nibbies', 'dey touch', 'ehuoo'
}

IMPORTANT_WORDS = {
    'chale_wale': 'friend/bro variants',
    'abeg_oya': 'pleading/urging particles',
    'wahala_sabi': 'trouble/knowledge core slang',
    'pikin_tiya': 'family/fatigue expressions',
    'obruni_trotro': 'cultural references (foreigner/transport)'
}

# English words for code-switching detection
ENGLISH_WORDS = {
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
    'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be',
    'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'can', 'may', 'must', 'shall', 'might', 'ought',
    'need', 'dare', 'used', 'able', 'going', 'want', 'like', 'know',
    'think', 'see', 'get', 'make', 'take', 'come', 'give', 'look', 'use',
    'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call'
}


def load_data(path: str) -> pd.DataFrame:
    """
    Load hate speech dataset from CSV file.

    Args:
        path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset with 'text' and 'label' columns

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required columns are missing
    """
    try:
        df = pd.read_csv(path)

        # Validate required columns
        required_cols = ['text', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Basic data validation
        df = df.dropna(subset=['text', 'label'])
        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].astype(int)

        # Validate labels are binary
        unique_labels = df['label'].unique()
        if not all(label in [0, 1] for label in unique_labels):
            raise ValueError(
                "Labels must be binary (0 for non-hate, 1 for hate)")

        print(f"Loaded dataset: {len(df)} samples")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def _initial_cleaning(text: str) -> str:
    """Remove HTML, newlines, and basic cleanup."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove newlines and normalize whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\r+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def _replace_special_tokens(text: str) -> str:
    """Replace URLs, mentions, hashtags, and emojis with special tokens."""
    # URLs
    text = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
    text = re.sub(
        r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)

    # Mentions (@ followed by username)
    text = re.sub(r'@\w+', '[USER]', text)

    # Hashtags
    text = re.sub(r'#\w+', '[HASHTAG]', text)

    # Emojis (Unicode ranges for emojis)
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+"
    )
    text = emoji_pattern.sub('[EMOJI]', text)

    return text


def _normalize_whitespace_and_case(text: str) -> str:
    """Lowercase and normalize whitespace."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _handle_punctuation(text: str) -> str:
    """Limit punctuation repeats and detach from words."""
    # Limit repeated punctuation to max 3
    text = re.sub(r'([.!?]){4,}', r'\1\1\1', text)
    text = re.sub(r'([,;:]){3,}', r'\1\1', text)

    # Detach punctuation from words (add spaces)
    text = re.sub(r'([a-zA-Z])([.!?,:;])', r'\1 \2', text)
    text = re.sub(r'([.!?,:;])([a-zA-Z])', r'\1 \2', text)

    return text


def _basic_tokenization(text: str) -> List[str]:
    """Basic tokenization using NLTK word_tokenize."""
    try:
        tokens = nltk.word_tokenize(text)
        return [token for token in tokens if token.strip()]
    except Exception:
        # Fallback to simple split
        return text.split()


def normalize_spelling(tokens: List[str], spelling_dict: Dict[str, str]) -> List[str]:
    """
    Normalize spelling variations using provided dictionary.

    Args:
        tokens (List[str]): List of tokens
        spelling_dict (Dict[str, str]): Mapping of variations to standard forms

    Returns:
        List[str]: Normalized tokens
    """
    normalized = []
    for token in tokens:
        # Check if token should be preserved
        if token.lower() in PRESERVE_SET:
            normalized.append(token)
        else:
            # Apply spelling normalization
            normalized_token = spelling_dict.get(token.lower(), token)
            normalized.append(normalized_token)

    return normalized


def _handle_code_switching(tokens: List[str]) -> List[str]:
    """Flag English words in code-switched text."""
    processed = []
    for token in tokens:
        # Skip special tokens and preserve set
        if token.startswith('[') and token.endswith(']'):
            processed.append(token)
        elif token.lower() in PRESERVE_SET:
            processed.append(token)
        elif token.lower() in ENGLISH_WORDS and len(token) > 2:
            # Flag as English code-switch
            processed.append(f'[EN:{token}]')
        else:
            processed.append(token)

    return processed


def generate_variants(token: str, spelling_dict: Dict[str, str], prob: float = 0.2) -> str:
    """
    Generate spelling variants for data augmentation.

    Args:
        token (str): Original token
        spelling_dict (Dict[str, str]): Spelling dictionary
        prob (float): Probability of generating variant

    Returns:
        str: Original token or variant
    """
    if random.random() > prob:
        return token

    # Find variants that map to this token
    variants = [k for k, v in spelling_dict.items() if v == token.lower()]

    if variants and random.random() < 0.5:
        return random.choice(variants)

    # Simple character-level variations for Pidgin
    if len(token) > 3:
        variations = [
            token.replace('e', 'a'),  # Common vowel swap
            token.replace('o', 'u'),  # Another common swap
            token[:-1] + token[-1] *
            2 if token[-1] in 'aeiou' else token,  # Double last vowel
        ]
        return random.choice(variations + [token])

    return token


def _remove_stopwords(tokens: List[str], stopwords: Set[str]) -> List[str]:
    """Remove stopwords while preserving important Pidgin expressions."""
    return [token for token in tokens if token.lower() not in stopwords or token.lower() in PRESERVE_SET]


def _light_lemmatization(tokens: List[str]) -> List[str]:
    """Apply light lemmatization rules for Pidgin."""
    lemmatized = []
    for token in tokens:
        # Skip special tokens and preserve set
        if token.startswith('[') or token.lower() in PRESERVE_SET:
            lemmatized.append(token)
            continue

        # Simple suffix removal rules
        lemma = token
        if token.endswith('ing') and len(token) > 5:
            lemma = token[:-3]
        elif token.endswith('ed') and len(token) > 4:
            lemma = token[:-2]
        elif token.endswith('s') and len(token) > 3 and not token.endswith('ss'):
            lemma = token[:-1]

        lemmatized.append(lemma)

    return lemmatized


def _simulate_subword_tokenization(tokens: List[str]) -> List[str]:
    """Simulate BPE-like subword tokenization for out-of-vocabulary handling."""
    subword_tokens = []

    for token in tokens:
        # Skip special tokens
        if token.startswith('[') and token.endswith(']'):
            subword_tokens.append(token)
            continue

        # Skip short tokens and preserve set
        if len(token) <= 3 or token.lower() in PRESERVE_SET:
            subword_tokens.append(token)
            continue

        # Simple subword splitting for long unknown words
        if len(token) > 8:
            # Split into roughly equal parts
            mid = len(token) // 2
            subword_tokens.extend([token[:mid] + '@@', token[mid:]])
        else:
            subword_tokens.append(token)

    return subword_tokens


def clean_text(text: str,
               augment: bool = False,
               remove_stops: bool = False,
               handle_code_switch: bool = True,
               apply_lemmatization: bool = False) -> Union[str, List[str]]:
    """
    Main text preprocessing pipeline for Pidgin hate speech detection.

    Args:
        text (str): Raw input text
        augment (bool): Whether to generate augmented variants
        remove_stops (bool): Whether to remove stopwords
        handle_code_switch (bool): Whether to flag code-switching
        apply_lemmatization (bool): Whether to apply light lemmatization

    Returns:
        Union[str, List[str]]: Processed text or list of variants if augmenting
    """
    if pd.isna(text) or not text.strip():
        return "" if not augment else [""]

    try:
        # Step 1: Initial cleaning
        text = _initial_cleaning(text)

        # Step 2: Special token replacement
        text = _replace_special_tokens(text)

        # Step 3: Lowercase & whitespace normalization
        text = _normalize_whitespace_and_case(text)

        # Step 4: Punctuation handling
        text = _handle_punctuation(text)

        # Step 5: Tokenization
        tokens = _basic_tokenization(text)

        # Step 6: Spelling normalization
        tokens = normalize_spelling(tokens, SPELLING_DICT)

        # Step 7: Code-switching handling
        if handle_code_switch:
            tokens = _handle_code_switching(tokens)

        # Step 8: Augmentation (if requested)
        if augment:
            variants = []
            for _ in range(3):  # Generate 3 variants
                aug_tokens = [generate_variants(
                    token, SPELLING_DICT) for token in tokens]
                variants.append(' '.join(aug_tokens))
            return variants

        # Step 9: Preserve key expressions (already handled in normalize_spelling)

        # Step 10: Stopword removal
        if remove_stops:
            tokens = _remove_stopwords(tokens, STOPWORDS_SET)

        # Step 11: Light lemmatization
        if apply_lemmatization:
            tokens = _light_lemmatization(tokens)

        # Step 12: Subword tokenization
        tokens = _simulate_subword_tokenization(tokens)

        # Step 13: Reassembly
        processed_text = ' '.join(tokens)

        return processed_text

    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return text if not augment else [text]


def balance_dataset(df: pd.DataFrame,
                    method: str = 'smote',
                    val_size: float = 0.2,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Balance dataset using SMOTE and split into train/val/test.

    Args:
        df (pd.DataFrame): Input dataset
        method (str): Balancing method ('smote', 'undersample', 'oversample')
        val_size (float): Proportion of validation set
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Balanced train, validation, and test DataFrames
    """
    try:
        # First split into train+val and test to avoid data leakage
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']
        )

        # Then split train+val into train and validation
        # Adjust val_size to account for the remaining data after test split
        adjusted_val_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            random_state=random_state,
            stratify=train_val_df['label']
        )

        print(
            f"Original train distribution: {train_df['label'].value_counts().to_dict()}")
        print(
            f"Validation distribution: {val_df['label'].value_counts().to_dict()}")
        print(
            f"Test distribution: {test_df['label'].value_counts().to_dict()}")

        if method.lower() == 'smote':
            # Apply SMOTE to training data only
            # Create TF-IDF features for SMOTE
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=2,
                lowercase=True
            )

            X_train_text = train_df['text'].fillna('').astype(str)
            X_train_tfidf = vectorizer.fit_transform(X_train_text)
            y_train = train_df['label']

            # Apply SMOTE
            smote = SMOTE(random_state=random_state)
            X_train_balanced, y_train_balanced = smote.fit_resample(
                X_train_tfidf, y_train)

            # Since SMOTE generates synthetic samples, we need to handle this carefully
            # We'll oversample the minority class by duplicating existing samples
            minority_class = y_train.value_counts().idxmin()
            majority_class = y_train.value_counts().idxmax()

            minority_samples = train_df[train_df['label'] == minority_class]
            majority_samples = train_df[train_df['label'] == majority_class]

            # Calculate how many samples to generate
            target_size = len(majority_samples)
            samples_needed = target_size - len(minority_samples)

            if samples_needed > 0:
                # Oversample minority class by duplicating with slight text variations
                augmented_samples = []
                for _ in range(samples_needed):
                    sample = minority_samples.sample(
                        1, random_state=random_state+_).iloc[0]
                    augmented_text = clean_text(sample['text'], augment=True)
                    if isinstance(augmented_text, list):
                        augmented_text = augmented_text[0]

                    augmented_samples.append({
                        'text': augmented_text,
                        'label': sample['label']
                    })

                augmented_df = pd.DataFrame(augmented_samples)
                train_balanced = pd.concat(
                    [train_df, augmented_df], ignore_index=True)
            else:
                train_balanced = train_df.copy()

        elif method.lower() == 'oversample':
            # Simple oversampling by duplication
            minority_class = train_df['label'].value_counts().idxmin()
            majority_class = train_df['label'].value_counts().idxmax()

            minority_samples = train_df[train_df['label'] == minority_class]
            majority_samples = train_df[train_df['label'] == majority_class]

            target_size = len(majority_samples)
            oversampled_minority = minority_samples.sample(
                n=target_size,
                replace=True,
                random_state=random_state
            )

            train_balanced = pd.concat(
                [majority_samples, oversampled_minority], ignore_index=True)

        elif method.lower() == 'undersample':
            # Simple undersampling
            minority_class = train_df['label'].value_counts().idxmin()
            majority_class = train_df['label'].value_counts().idxmax()

            minority_samples = train_df[train_df['label'] == minority_class]
            majority_samples = train_df[train_df['label'] == majority_class]

            target_size = len(minority_samples)
            undersampled_majority = majority_samples.sample(
                n=target_size,
                random_state=random_state
            )

            train_balanced = pd.concat(
                [minority_samples, undersampled_majority], ignore_index=True)

        else:
            raise ValueError(f"Unknown balancing method: {method}")

        # Shuffle the balanced training set
        train_balanced = train_balanced.sample(
            frac=1, random_state=random_state).reset_index(drop=True)

        print(
            f"Balanced train distribution: {train_balanced['label'].value_counts().to_dict()}")
        print(
            f"Final sizes - Train: {len(train_balanced)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_balanced, val_df, test_df

    except Exception as e:
        raise Exception(f"Error balancing dataset: {str(e)}")


def save_processed(data_dir: str,
                   train_df: pd.DataFrame,
                   val_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   processed_subdir: str = 'processed') -> None:
    """
    Save processed train, validation, and test datasets to CSV files.

    Args:
        data_dir (str): Base data directory
        train_df (pd.DataFrame): Training dataset
        val_df (pd.DataFrame): Validation dataset
        test_df (pd.DataFrame): Test dataset
        processed_subdir (str): Subdirectory name for processed data
    """
    import os

    try:
        # Create processed data directory
        processed_dir = os.path.join(data_dir, processed_subdir)
        os.makedirs(processed_dir, exist_ok=True)

        # Save datasets
        train_path = os.path.join(processed_dir, 'train.csv')
        val_path = os.path.join(processed_dir, 'val.csv')
        test_path = os.path.join(processed_dir, 'test.csv')

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Saved training data: {train_path} ({len(train_df)} samples)")
        print(f"Saved validation data: {val_path} ({len(val_df)} samples)")
        print(f"Saved test data: {test_path} ({len(test_df)} samples)")

        # Save preprocessing statistics
        stats = {
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'train_hate_ratio': train_df['label'].mean(),
            'val_hate_ratio': val_df['label'].mean(),
            'test_hate_ratio': test_df['label'].mean(),
            'train_distribution': train_df['label'].value_counts().to_dict(),
            'val_distribution': val_df['label'].value_counts().to_dict(),
            'test_distribution': test_df['label'].value_counts().to_dict()
        }

        stats_path = os.path.join(processed_dir, 'preprocessing_stats.txt')
        with open(stats_path, 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

        print(f"Saved preprocessing statistics: {stats_path}")

    except Exception as e:
        raise Exception(f"Error saving processed data: {str(e)}")


def preprocess_pipeline(input_path: str,
                        output_dir: str,
                        clean_params: Optional[Dict] = None,
                        balance_method: str = 'smote',
                        val_size: float = 0.2,
                        test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline from raw data to balanced train/val/test sets.

    Args:
        input_path (str): Path to raw CSV file
        output_dir (str): Output directory for processed data
        clean_params (Dict, optional): Parameters for text cleaning
        balance_method (str): Method for dataset balancing
        val_size (float): Validation set proportion
        test_size (float): Test set proportion

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Processed train, validation, and test DataFrames
    """
    print("Starting preprocessing pipeline...")

    # Default cleaning parameters
    if clean_params is None:
        clean_params = {
            'remove_stops': False,
            'handle_code_switch': True,
            'apply_lemmatization': False
        }

    try:
        # Load data
        print("Loading data...")
        df = load_data(input_path)

        # Clean text
        print("Cleaning text...")
        df['text'] = df['text'].apply(lambda x: clean_text(x, **clean_params))

        # Remove empty texts after cleaning
        df = df[df['text'].str.strip() != ''].reset_index(drop=True)
        print(f"After cleaning: {len(df)} samples")

        # Balance and split dataset
        print("Balancing and splitting dataset...")
        train_df, val_df, test_df = balance_dataset(
            df,
            method=balance_method,
            val_size=val_size,
            test_size=test_size
        )

        # Save processed data
        print("Saving processed data...")
        save_processed(output_dir, train_df, val_df, test_df)

        print("Preprocessing pipeline completed successfully!")
        return train_df, val_df, test_df

    except Exception as e:
        raise Exception(f"Preprocessing pipeline failed: {str(e)}")


def load_splits(base_path: str = 'data/processed/') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load processed train, validation, and test splits.

    Args:
        base_path (str): Base path to processed data directory

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test DataFrames

    Raises:
        FileNotFoundError: If any of the split files are missing
    """
    import os

    train_path = os.path.join(base_path, 'train.csv')
    val_path = os.path.join(base_path, 'val.csv')
    test_path = os.path.join(base_path, 'test.csv')

    # Check if all files exist
    missing_files = []
    for path, split_name in [(train_path, 'train'), (val_path, 'val'), (test_path, 'test')]:
        if not os.path.exists(path):
            missing_files.append(f"{split_name}: {path}")

    if missing_files:
        raise FileNotFoundError(
            f"Missing processed data files:\n" + "\n".join(missing_files) +
            f"\nPlease run the preprocessing pipeline first to generate these files."
        )

    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        # Validate DataFrames have required columns
        required_cols = ['text', 'label']
        for df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
            missing_cols = [
                col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing columns in {name} data: {missing_cols}")

        print(
            f"Loaded splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    except Exception as e:
        raise Exception(f"Error loading splits: {str(e)}")


if __name__ == "__main__":
    # Example usage
    try:
        # Load and preprocess data
        input_file = "data/hate.csv"
        output_directory = "data"

        train_data, val_data, test_data = preprocess_pipeline(
            input_path=input_file,
            output_dir=output_directory,
            balance_method='smote',
            val_size=0.15,
            test_size=0.15
        )

        print("\nPreprocessing completed!")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")

    except Exception as e:
        print(f"Error: {str(e)}")
