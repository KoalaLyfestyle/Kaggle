"""
NLP Feature Extraction Pipeline for Twitter Bot Detection

This module provides text feature extraction using:
- Word-level TF-IDF (unigrams + bigrams)
- Character-level TF-IDF (3-5 char n-grams)
- TruncatedSVD dimensionality reduction
- Combined text from multiple columns

Usage:
------
from helpers.nlp import TextFeatureExtractor, combine_text_columns

# Combine text columns
train_text = combine_text_columns(train_df, columns=['description', 'screen_name', 'location'])
test_text = combine_text_columns(test_df, columns=['description', 'screen_name', 'location'])

# Create and fit extractor
extractor = TextFeatureExtractor(
    max_features_word=30000,
    max_features_char=30000,
    n_components=300
)

# Fit on training data only
X_text_train = extractor.fit_transform(train_text)

# Transform test data
X_text_test = extractor.transform(test_text)

# Combine with structured features
import numpy as np
X_train_combined = np.hstack([X_train_structured, X_text_train])
X_test_combined = np.hstack([X_test_structured, X_text_test])
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack
import warnings

warnings.filterwarnings('ignore')

# Try to import cuML for GPU acceleration, fallback to sklearn
try:
    import cupy as cp
    from cuml.feature_extraction.text import TfidfVectorizer as cumlTfidfVectorizer
    from cuml.decomposition import TruncatedSVD as cumlTruncatedSVD
    CUML_AVAILABLE = True
    print("✓ cuML detected - GPU acceleration enabled for TF-IDF and TruncatedSVD")
except ImportError:
    CUML_AVAILABLE = False
    print("✗ cuML not available - using CPU-based scikit-learn")

# Always import sklearn as fallback
from sklearn.feature_extraction.text import TfidfVectorizer as sklearnTfidfVectorizer
from sklearn.decomposition import TruncatedSVD as sklearnTruncatedSVD


def combine_text_columns(df, columns=['description', 'screen_name', 'location'], separator=' '):
    """
    Combine multiple text columns into a single text field for NLP processing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with text columns
    columns : list of str
        Names of text columns to combine (default: ['description', 'screen_name', 'location'])
    separator : str
        String to use when joining columns (default: ' ')
    
    Returns:
    --------
    pd.Series : Combined text column with blanks filled as empty strings
    
    Example:
    --------
    >>> train_text = combine_text_columns(train, ['description', 'screen_name'])
    >>> test_text = combine_text_columns(test, ['description', 'screen_name'])
    """
    combined_texts = []
    
    for col in columns:
        if col in df.columns:
            # Fill NaN with empty string and convert to string
            text_col = df[col].fillna('').astype(str)
            combined_texts.append(text_col)
        else:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping.")
    
    if not combined_texts:
        # Return empty strings if no valid columns found
        return pd.Series([''] * len(df), index=df.index)
    
    # Combine all text columns
    combined = combined_texts[0]
    for text_col in combined_texts[1:]:
        combined = combined + separator + text_col
    
    return combined


class TextFeatureExtractor:
    """
    Complete text feature extraction pipeline combining:
    - Word-level TF-IDF (1-2 grams)
    - Character-level TF-IDF (3-5 grams)
    - TruncatedSVD dimensionality reduction
    
    This class ensures proper train/test splitting by fitting only on training data
    and transforming both train and test with the same learned vectorizers.
    
    **GPU Acceleration**: 
    - TF-IDF vectorization uses cuML (RAPIDS) for GPU acceleration when available
    - SVD always uses CPU (sklearn) for better stability and sparse matrix support
    - This hybrid approach gives best performance and reliability
    
    Parameters:
    -----------
    max_features_word : int, default=30000
        Maximum number of features for word-level TF-IDF
    max_features_char : int, default=30000
        Maximum number of features for character-level TF-IDF
    n_components : int, default=300
        Number of components for TruncatedSVD reduction
    word_ngram_range : tuple, default=(1, 2)
        N-gram range for word-level TF-IDF (unigrams + bigrams)
    char_ngram_range : tuple, default=(3, 5)
        N-gram range for character-level TF-IDF (3-5 character n-grams)
    min_df : int or float, default=2
        Minimum document frequency for TF-IDF (removes very rare features)
    max_df : float, default=0.95
        Maximum document frequency for TF-IDF (removes very common features)
    use_idf : bool, default=True
        Enable inverse-document-frequency weighting
    sublinear_tf : bool, default=True
        Apply sublinear tf scaling (log normalization)
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Print progress information
    use_gpu : bool, default=True
        Use GPU acceleration via cuML if available. Set to False to force CPU usage.
    
    Attributes:
    -----------
    word_vectorizer_ : TfidfVectorizer
        Fitted word-level TF-IDF vectorizer (cuML or sklearn)
    char_vectorizer_ : TfidfVectorizer
        Fitted character-level TF-IDF vectorizer (cuML or sklearn)
    svd_ : TruncatedSVD
        Fitted TruncatedSVD transformer (cuML or sklearn)
    is_fitted_ : bool
        Whether the extractor has been fitted
    use_gpu : bool
        Whether GPU acceleration is being used
    
    Methods:
    --------
    fit(texts)
        Fit vectorizers and SVD on training texts
    transform(texts)
        Transform texts using fitted vectorizers and SVD
    fit_transform(texts)
        Fit and transform in one step (for training data)
    get_feature_names()
        Get names of output features (svd_0, svd_1, ..., svd_n)
    
    Examples:
    ---------
    >>> # CPU mode (default if cuML not installed)
    >>> extractor = TextFeatureExtractor(n_components=300)
    >>> X_train = extractor.fit_transform(train_texts)
    >>> X_test = extractor.transform(test_texts)
    
    >>> # Force CPU mode even if GPU available
    >>> extractor = TextFeatureExtractor(n_components=300, use_gpu=False)
    
    >>> # GPU mode (automatic if cuML installed)
    >>> extractor = TextFeatureExtractor(n_components=300, use_gpu=True)
    """
    
    def __init__(
        self,
        max_features_word=30000,
        max_features_char=30000,
        n_components=300,
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        use_idf=True,
        sublinear_tf=True,
        random_state=42,
        verbose=True,
        use_gpu=True
    ):
        self.max_features_word = max_features_word
        self.max_features_char = max_features_char
        self.n_components = n_components
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.random_state = random_state
        self.verbose = verbose
        self.use_gpu = use_gpu and CUML_AVAILABLE
        
        # Initialize vectorizers
        self.word_vectorizer_ = None
        self.char_vectorizer_ = None
        self.svd_ = None
        self.is_fitted_ = False
        
        # Determine which implementation to use
        if self.use_gpu:
            self.TfidfVectorizer = cumlTfidfVectorizer
            self.TruncatedSVD = sklearnTruncatedSVD  # Always use sklearn for SVD
            if self.verbose:
                print("🚀 Using GPU-accelerated cuML for TF-IDF (SVD uses CPU for stability)")
        else:
            self.TfidfVectorizer = sklearnTfidfVectorizer
            self.TruncatedSVD = sklearnTruncatedSVD
            if self.verbose and CUML_AVAILABLE:
                print("ℹ️  Using CPU-based scikit-learn (GPU disabled by user)")
            elif self.verbose:
                print("ℹ️  Using CPU-based scikit-learn (cuML not available)")
    
    def fit(self, texts):
        """
        Fit the text feature extractors on training data.
        
        Parameters:
        -----------
        texts : array-like of str
            Training texts (e.g., pd.Series or list of strings)
        
        Returns:
        --------
        self : TextFeatureExtractor
            Fitted extractor
        """
        if self.verbose:
            print("="*70)
            print("FITTING TEXT FEATURE EXTRACTOR")
            print("="*70)
            print(f"Number of texts: {len(texts)}")
        
        # Ensure texts are string type
        texts = pd.Series(texts).fillna('').astype(str)
        
        # 1. Fit word-level TF-IDF
        if self.verbose:
            print(f"\n1. Fitting word-level TF-IDF (ngrams={self.word_ngram_range})...")
        
        # Build vectorizer parameters (cuML doesn't support token_pattern)
        word_vectorizer_params = {
            'max_features': self.max_features_word,
            'ngram_range': self.word_ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'use_idf': self.use_idf,
            'sublinear_tf': self.sublinear_tf,
            'analyzer': 'word'
        }
        
        # Only add token_pattern for sklearn (cuML doesn't support it)
        if not self.use_gpu:
            word_vectorizer_params['token_pattern'] = r'\b\w+\b'
        
        self.word_vectorizer_ = self.TfidfVectorizer(**word_vectorizer_params)
        
        word_features = self.word_vectorizer_.fit_transform(texts)
        
        if self.verbose:
            print(f"   Word features shape: {word_features.shape}")
            if hasattr(self.word_vectorizer_, 'vocabulary_'):
                print(f"   Vocabulary size: {len(self.word_vectorizer_.vocabulary_)}")
        
        # 2. Fit character-level TF-IDF
        if self.verbose:
            print(f"\n2. Fitting character-level TF-IDF (ngrams={self.char_ngram_range})...")
        
        self.char_vectorizer_ = self.TfidfVectorizer(
            max_features=self.max_features_char,
            ngram_range=self.char_ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            use_idf=self.use_idf,
            sublinear_tf=self.sublinear_tf,
            analyzer='char'
        )
        
        char_features = self.char_vectorizer_.fit_transform(texts)
        
        if self.verbose:
            print(f"   Char features shape: {char_features.shape}")
            if hasattr(self.char_vectorizer_, 'vocabulary_'):
                print(f"   Vocabulary size: {len(self.char_vectorizer_.vocabulary_)}")
        
        # 3. Concatenate sparse features
        if self.verbose:
            print(f"\n3. Concatenating word + char features...")
        
        # Convert cuML sparse matrices to scipy sparse format if using GPU
        if self.use_gpu:
            try:
                import cupy as cp
                from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
                from scipy.sparse import csr_matrix as scipy_csr_matrix
                
                # Convert cuML/CuPy sparse to scipy sparse
                if hasattr(word_features, 'get'):  # CuPy array
                    word_features = scipy_csr_matrix(word_features.get())
                if hasattr(char_features, 'get'):  # CuPy array
                    char_features = scipy_csr_matrix(char_features.get())
            except ImportError:
                pass
        
        combined_features = sparse_hstack([word_features, char_features])
        
        if self.verbose:
            print(f"   Combined features shape: {combined_features.shape}")
        
        # 4. Fit TruncatedSVD for dimensionality reduction
        if self.verbose:
            print(f"\n4. Fitting TruncatedSVD (n_components={self.n_components})...")
        
        # Ensure n_components doesn't exceed feature count
        n_components_actual = min(self.n_components, combined_features.shape[1] - 1)
        
        if n_components_actual < self.n_components:
            print(f"   Warning: Reducing n_components from {self.n_components} to {n_components_actual}")
            print(f"   (Cannot exceed {combined_features.shape[1] - 1} for this data)")
        
        # IMPORTANT: Always use CPU for SVD (sparse matrices are more efficient)
        # GPU SVD requires dense arrays which can be too large and cause CUDA errors
        if self.verbose and self.use_gpu:
            print(f"   Note: Using CPU for TruncatedSVD (more efficient with sparse matrices)")
        
        # Use sklearn's TruncatedSVD for sparse matrix support
        self.svd_ = sklearnTruncatedSVD(
            n_components=n_components_actual,
            random_state=self.random_state
        )
        
        self.svd_.fit(combined_features)
        
        if self.verbose:
            if hasattr(self.svd_, 'explained_variance_ratio_'):
                explained_variance = self.svd_.explained_variance_ratio_.sum()
                # Handle cuML vs sklearn differences
                if self.use_gpu and hasattr(explained_variance, 'get'):
                    explained_variance = float(explained_variance.get())
                print(f"   SVD explained variance: {explained_variance:.2%}")
            print(f"   Output shape: ({len(texts)}, {n_components_actual})")
        
        self.is_fitted_ = True
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("TEXT FEATURE EXTRACTOR FITTED SUCCESSFULLY")
            print(f"{'='*70}\n")
        
        return self
    
    def transform(self, texts):
        """
        Transform texts using fitted vectorizers and SVD.
        
        Parameters:
        -----------
        texts : array-like of str
            Texts to transform (e.g., validation or test data)
        
        Returns:
        --------
        np.ndarray : Dense array of shape (n_samples, n_components)
            Reduced text features (always returned as NumPy array on CPU)
        """
        if not self.is_fitted_:
            raise ValueError("TextFeatureExtractor must be fitted before transform. Call fit() first.")
        
        if self.verbose:
            print(f"Transforming {len(texts)} texts...")
        
        # Ensure texts are string type
        texts = pd.Series(texts).fillna('').astype(str)
        
        # 1. Transform with word vectorizer
        word_features = self.word_vectorizer_.transform(texts)
        
        # 2. Transform with char vectorizer
        char_features = self.char_vectorizer_.transform(texts)
        
        # 3. Concatenate (convert cuML sparse to scipy sparse if needed)
        if self.use_gpu:
            try:
                from scipy.sparse import csr_matrix as scipy_csr_matrix
                
                # Convert cuML/CuPy sparse to scipy sparse
                if hasattr(word_features, 'get'):  # CuPy array
                    word_features = scipy_csr_matrix(word_features.get())
                if hasattr(char_features, 'get'):  # CuPy array
                    char_features = scipy_csr_matrix(char_features.get())
            except ImportError:
                pass
        
        combined_features = sparse_hstack([word_features, char_features])
        
        # 4. Apply SVD reduction (always CPU - handles sparse efficiently)
        reduced_features = self.svd_.transform(combined_features)
        
        # SVD output is already a NumPy array (we use sklearn's TruncatedSVD)
        if self.verbose:
            print(f"   Output shape: {reduced_features.shape}")
        
        return reduced_features
    
    def fit_transform(self, texts):
        """
        Fit and transform in one step (convenience method for training data).
        
        Parameters:
        -----------
        texts : array-like of str
            Training texts
        
        Returns:
        --------
        np.ndarray : Dense array of shape (n_samples, n_components)
            Reduced text features
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self):
        """
        Get names of output features.
        
        Returns:
        --------
        list of str : Feature names ['svd_0', 'svd_1', ..., 'svd_n']
        """
        if not self.is_fitted_:
            raise ValueError("TextFeatureExtractor must be fitted before getting feature names.")
        
        n_components_actual = self.svd_.n_components
        return [f'text_svd_{i}' for i in range(n_components_actual)]
    
    def __repr__(self):
        """String representation of the extractor."""
        if self.is_fitted_:
            return (
                f"TextFeatureExtractor(fitted=True, "
                f"word_features={self.max_features_word}, "
                f"char_features={self.max_features_char}, "
                f"n_components={self.svd_.n_components})"
            )
        else:
            return (
                f"TextFeatureExtractor(fitted=False, "
                f"word_features={self.max_features_word}, "
                f"char_features={self.max_features_char}, "
                f"n_components={self.n_components})"
            )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the text feature extraction pipeline.
    Run this script to test the NLP pipeline on your data.
    """
    
    print("="*70)
    print("TEXT FEATURE EXTRACTION PIPELINE - EXAMPLE USAGE")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # Example 1: Combine text columns
    print("\n" + "="*70)
    print("EXAMPLE 1: COMBINING TEXT COLUMNS")
    print("="*70)
    
    text_columns = ['description', 'screen_name', 'location']
    train_text = combine_text_columns(train, columns=text_columns)
    test_text = combine_text_columns(test, columns=text_columns)
    
    print(f"\nCombined text shape: {train_text.shape}")
    print("\nFirst 3 examples:")
    for i in range(3):
        print(f"  [{i}]: {train_text.iloc[i][:100]}...")
    
    # Example 2: Extract text features
    print("\n" + "="*70)
    print("EXAMPLE 2: TEXT FEATURE EXTRACTION")
    print("="*70)
    
    # Create extractor with default parameters
    extractor = TextFeatureExtractor(
        max_features_word=30000,
        max_features_char=30000,
        n_components=300,
        verbose=True
    )
    
    # Fit on training data and transform
    print("\nFitting on training data...")
    X_text_train = extractor.fit_transform(train_text)
    
    # Transform test data
    print("\nTransforming test data...")
    X_text_test = extractor.transform(test_text)
    
    print(f"\nFinal shapes:")
    print(f"  Training text features: {X_text_train.shape}")
    print(f"  Test text features: {X_text_test.shape}")
    
    # Example 3: Combine with structured features
    print("\n" + "="*70)
    print("EXAMPLE 3: COMBINING WITH STRUCTURED FEATURES")
    print("="*70)
    
    from helpers.process_data import process_data
    
    # Process structured features
    X_train_structured, feature_list = process_data(train, verbose=False)
    X_test_structured = process_data(test, verbose=False, expected_features=feature_list)
    
    print(f"\nStructured features shape:")
    print(f"  Training: {X_train_structured.shape}")
    print(f"  Test: {X_test_structured.shape}")
    
    # Combine with text features
    X_train_combined = np.hstack([X_train_structured.values, X_text_train])
    X_test_combined = np.hstack([X_test_structured.values, X_text_test])
    
    print(f"\nCombined features shape:")
    print(f"  Training: {X_train_combined.shape}")
    print(f"  Test: {X_test_combined.shape}")
    
    # Feature counts
    n_structured = X_train_structured.shape[1]
    n_text = X_text_train.shape[1]
    n_total = X_train_combined.shape[1]
    
    print(f"\nFeature breakdown:")
    print(f"  Structured features: {n_structured}")
    print(f"  Text features (SVD): {n_text}")
    print(f"  Total features: {n_total}")
    
    print("\n" + "="*70)
    print("TEXT FEATURE EXTRACTION PIPELINE READY!")
    print("="*70)
    print("\nTo use in your models:")
    print("  1. Import: from helpers.nlp import TextFeatureExtractor, combine_text_columns")
    print("  2. Combine text: train_text = combine_text_columns(train, ['description', 'screen_name'])")
    print("  3. Create extractor: extractor = TextFeatureExtractor(n_components=300)")
    print("  4. Fit: X_text_train = extractor.fit_transform(train_text)")
    print("  5. Transform: X_text_test = extractor.transform(test_text)")
    print("  6. Combine: X_combined = np.hstack([X_structured, X_text])")
