"""
NLP Feature Caching Utilities

This module provides efficient caching for text feature extraction to speed up
model training, especially during hyperparameter optimization with Optuna.

IMPORTANT: Caching is only safe AFTER fitting on the full training dataset.
DO NOT use caching during cross-validation folds as it may cause data leakage.
"""

import joblib
import os
import hashlib
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def get_cache_key(train_data: pd.DataFrame, text_params: dict, text_columns: list) -> str:
    """
    Generate a unique cache key based on data and text parameters.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data (used to compute hash of indices)
    text_params : dict
        Text extraction parameters (max_features_word, max_features_char, n_components)
    text_columns : list
        List of text column names being used
        
    Returns:
    --------
    str : Cache key (filename-safe string)
    """
    # Create a unique identifier based on:
    # 1. Data shape and index hash
    # 2. Text parameters
    # 3. Text columns used
    
    data_id = f"{train_data.shape}_{hash(tuple(train_data.index))}"
    params_id = "_".join([f"{k}_{v}" for k, v in sorted(text_params.items())])
    cols_id = "_".join(sorted(text_columns))
    
    # Create hash for filename safety
    full_id = f"{data_id}_{params_id}_{cols_id}"
    cache_key = hashlib.md5(full_id.encode()).hexdigest()[:16]
    
    return cache_key


def save_text_features_cache(
    cache_key: str,
    text_extractor,
    X_train_text: np.ndarray,
    X_test_text: Optional[np.ndarray] = None,
    cache_dir: str = '.nlp_cache'
) -> None:
    """
    Save fitted text extractor and extracted features to cache.
    
    Parameters:
    -----------
    cache_key : str
        Unique cache identifier
    text_extractor : TextFeatureExtractor
        Fitted text feature extractor
    X_train_text : np.ndarray
        Extracted training text features
    X_test_text : np.ndarray, optional
        Extracted test text features (if available)
    cache_dir : str
        Directory to store cache files
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_path = os.path.join(cache_dir, f'cache_{cache_key}')
    
    cache_data = {
        'text_extractor': text_extractor,
        'X_train_text': X_train_text,
        'X_test_text': X_test_text,
        'params': {
            'max_features_word': text_extractor.max_features_word,
            'max_features_char': text_extractor.max_features_char,
            'n_components': text_extractor.n_components
        }
    }
    
    joblib.dump(cache_data, f'{cache_path}.joblib')
    print(f"✓ Text features cached to: {cache_path}.joblib")


def load_text_features_cache(
    cache_key: str,
    cache_dir: str = '.nlp_cache'
) -> Optional[dict]:
    """
    Load cached text extractor and features.
    
    Parameters:
    -----------
    cache_key : str
        Unique cache identifier
    cache_dir : str
        Directory where cache files are stored
        
    Returns:
    --------
    dict or None : Cache data if found, None otherwise
        Contains: 'text_extractor', 'X_train_text', 'X_test_text', 'params'
    """
    cache_path = os.path.join(cache_dir, f'cache_{cache_key}.joblib')
    
    if os.path.exists(cache_path):
        print(f"✓ Loading cached text features from: {cache_path}")
        cache_data = joblib.load(cache_path)
        print(f"  Train text shape: {cache_data['X_train_text'].shape}")
        if cache_data['X_test_text'] is not None:
            print(f"  Test text shape: {cache_data['X_test_text'].shape}")
        print(f"  Text params: {cache_data['params']}")
        return cache_data
    else:
        print(f"✗ No cache found for key: {cache_key}")
        return None


def check_cache_exists(cache_key: str, cache_dir: str = '.nlp_cache') -> bool:
    """
    Check if cache exists for given key.
    
    Parameters:
    -----------
    cache_key : str
        Unique cache identifier
    cache_dir : str
        Directory where cache files are stored
        
    Returns:
    --------
    bool : True if cache exists, False otherwise
    """
    cache_path = os.path.join(cache_dir, f'cache_{cache_key}.joblib')
    return os.path.exists(cache_path)


def clear_all_caches(cache_dir: str = '.nlp_cache') -> int:
    """
    Clear all cached text features.
    
    Parameters:
    -----------
    cache_dir : str
        Directory where cache files are stored
        
    Returns:
    --------
    int : Number of cache files deleted
    """
    if not os.path.exists(cache_dir):
        print(f"Cache directory '{cache_dir}' does not exist")
        return 0
    
    cache_files = [f for f in os.listdir(cache_dir) if f.startswith('cache_') and f.endswith('.joblib')]
    
    if not cache_files:
        print(f"No cache files found in '{cache_dir}'")
        return 0
    
    print(f"Deleting {len(cache_files)} cache file(s) from '{cache_dir}'...")
    
    for cache_file in cache_files:
        cache_path = os.path.join(cache_dir, cache_file)
        try:
            os.remove(cache_path)
            print(f"  ✓ Deleted: {cache_file}")
        except Exception as e:
            print(f"  ✗ Failed to delete {cache_file}: {e}")
    
    # Try to remove directory if empty
    try:
        if not os.listdir(cache_dir):
            os.rmdir(cache_dir)
            print(f"  ✓ Removed empty cache directory: {cache_dir}")
    except Exception as e:
        print(f"  Note: Cache directory not removed: {e}")
    
    return len(cache_files)


def get_or_create_text_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    text_columns: list,
    text_params: dict,
    use_cache: bool = True,
    cache_dir: str = '.nlp_cache'
) -> Tuple[object, np.ndarray, np.ndarray]:
    """
    Get text features from cache or create them if not cached.
    
    This is a high-level convenience function that handles the entire
    cache workflow.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame
        Test data
    text_columns : list
        List of text column names to combine
    text_params : dict
        Text extraction parameters
    use_cache : bool
        Whether to use caching
    cache_dir : str
        Cache directory
        
    Returns:
    --------
    tuple : (text_extractor, X_train_text, X_test_text)
    """
    from helpers.nlp import TextFeatureExtractor, combine_text_columns
    
    if use_cache:
        # Generate cache key
        cache_key = get_cache_key(train_data, text_params, text_columns)
        print(f"Cache key: {cache_key}")
        
        # Try to load from cache
        cache_data = load_text_features_cache(cache_key, cache_dir)
        
        if cache_data is not None:
            return (
                cache_data['text_extractor'],
                cache_data['X_train_text'],
                cache_data['X_test_text']
            )
    
    # No cache or caching disabled - extract features
    print("Extracting text features (no cache)...")
    
    train_text = combine_text_columns(train_data, columns=text_columns)
    test_text = combine_text_columns(test_data, columns=text_columns)
    
    text_extractor = TextFeatureExtractor(
        max_features_word=text_params['max_features_word'],
        max_features_char=text_params['max_features_char'],
        n_components=text_params['n_components'],
        verbose=True
    )
    
    X_train_text = text_extractor.fit_transform(train_text)
    X_test_text = text_extractor.transform(test_text)
    
    if use_cache:
        # Save to cache for future use
        cache_key = get_cache_key(train_data, text_params, text_columns)
        save_text_features_cache(
            cache_key, text_extractor, X_train_text, X_test_text, cache_dir
        )
    
    return text_extractor, X_train_text, X_test_text


# Example usage documentation
if __name__ == "__main__":
    print(__doc__)
    print("\nExample Usage:")
    print("=" * 60)
    print("""
# In your notebook, after loading data:

from helpers.nlp_cache import get_or_create_text_features, clear_all_caches

text_params = {
    'max_features_word': 30000,
    'max_features_char': 30000,
    'n_components': 300
}

text_columns = ['description', 'screen_name', 'location']

# Get features (uses cache if available)
text_extractor, X_train_text, X_test_text = get_or_create_text_features(
    train, test, text_columns, text_params, use_cache=True
)

# ... do your modeling ...

# At the end of notebook, clean up caches
clear_all_caches()
    """)
