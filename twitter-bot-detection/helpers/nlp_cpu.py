"""
NLP Feature Extraction Pipeline for Twitter Bot Detection (CPU-Only Version)

This is a simplified CPU-only version using pure scikit-learn.
Use this version for Google Colab or when you don't have cuML installed.

Usage:
------
from helpers.nlp_cpu import TextFeatureExtractor, combine_text_columns

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
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings

warnings.filterwarnings('ignore')


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
    """
    combined_texts = []
    
    for col in columns:
        if col in df.columns:
            text_col = df[col].fillna('').astype(str)
            combined_texts.append(text_col)
        else:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping.")
    
    if not combined_texts:
        return pd.Series([''] * len(df), index=df.index)
    
    combined = combined_texts[0]
    for text_col in combined_texts[1:]:
        combined = combined + separator + text_col
    
    return combined


class TextFeatureExtractor:
    """
    CPU-only text feature extraction pipeline combining:
    - Word-level TF-IDF (1-2 grams)
    - Character-level TF-IDF (3-5 grams)
    - TruncatedSVD dimensionality reduction
    
    This version uses pure scikit-learn for maximum compatibility.
    
    Parameters:
    -----------
    max_features_word : int, default=30000
        Maximum number of features for word-level TF-IDF
    max_features_char : int, default=30000
        Maximum number of features for character-level TF-IDF
    n_components : int, default=300
        Number of components for TruncatedSVD reduction
    word_ngram_range : tuple, default=(1, 2)
        N-gram range for word-level TF-IDF
    char_ngram_range : tuple, default=(3, 5)
        N-gram range for character-level TF-IDF
    min_df : int or float, default=2
        Minimum document frequency for TF-IDF
    max_df : float, default=0.95
        Maximum document frequency for TF-IDF
    use_idf : bool, default=True
        Enable inverse-document-frequency weighting
    sublinear_tf : bool, default=True
        Apply sublinear tf scaling
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Print progress information
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
        verbose=True
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
        
        self.word_vectorizer_ = None
        self.char_vectorizer_ = None
        self.svd_ = None
        self.is_fitted_ = False
        
        if self.verbose:
            print("ℹ️  Using CPU-based scikit-learn (simple and reliable)")
    
    def fit(self, texts):
        """
        Fit the text feature extractors on training data.
        
        Parameters:
        -----------
        texts : array-like of str
            Training texts
        
        Returns:
        --------
        self : TextFeatureExtractor
            Fitted extractor
        """
        if self.verbose:
            print("="*70)
            print("FITTING TEXT FEATURE EXTRACTOR (CPU)")
            print("="*70)
            print(f"Number of texts: {len(texts)}")
        
        texts = pd.Series(texts).fillna('').astype(str)
        
        # 1. Fit word-level TF-IDF
        if self.verbose:
            print(f"\n1. Fitting word-level TF-IDF (ngrams={self.word_ngram_range})...")
        
        self.word_vectorizer_ = TfidfVectorizer(
            max_features=self.max_features_word,
            ngram_range=self.word_ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            use_idf=self.use_idf,
            sublinear_tf=self.sublinear_tf,
            analyzer='word',
            token_pattern=r'\b\w+\b'
        )
        
        word_features = self.word_vectorizer_.fit_transform(texts)
        
        if self.verbose:
            print(f"   Word features shape: {word_features.shape}")
        
        # 2. Fit character-level TF-IDF
        if self.verbose:
            print(f"\n2. Fitting character-level TF-IDF (ngrams={self.char_ngram_range})...")
        
        self.char_vectorizer_ = TfidfVectorizer(
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
        
        # 3. Concatenate sparse features
        if self.verbose:
            print(f"\n3. Concatenating word + char features...")
        
        combined_features = sparse_hstack([word_features, char_features])
        
        if self.verbose:
            print(f"   Combined features shape: {combined_features.shape}")
        
        # 4. Fit TruncatedSVD
        if self.verbose:
            print(f"\n4. Fitting TruncatedSVD (n_components={self.n_components})...")
        
        n_components_actual = min(self.n_components, combined_features.shape[1] - 1)
        
        if n_components_actual < self.n_components:
            print(f"   Warning: Reducing n_components from {self.n_components} to {n_components_actual}")
        
        self.svd_ = TruncatedSVD(
            n_components=n_components_actual,
            random_state=self.random_state
        )
        
        self.svd_.fit(combined_features)
        
        if self.verbose:
            explained_variance = self.svd_.explained_variance_ratio_.sum()
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
            Texts to transform
        
        Returns:
        --------
        np.ndarray : Dense array of shape (n_samples, n_components)
            Reduced text features
        """
        if not self.is_fitted_:
            raise ValueError("TextFeatureExtractor must be fitted before transform.")
        
        if self.verbose:
            print(f"Transforming {len(texts)} texts...")
        
        texts = pd.Series(texts).fillna('').astype(str)
        
        # Transform with vectorizers
        word_features = self.word_vectorizer_.transform(texts)
        char_features = self.char_vectorizer_.transform(texts)
        
        # Concatenate
        combined_features = sparse_hstack([word_features, char_features])
        
        # Apply SVD
        reduced_features = self.svd_.transform(combined_features)
        
        if self.verbose:
            print(f"   Output shape: {reduced_features.shape}")
        
        return reduced_features
    
    def fit_transform(self, texts):
        """
        Fit and transform in one step.
        
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
        """Get names of output features."""
        if not self.is_fitted_:
            raise ValueError("TextFeatureExtractor must be fitted first.")
        
        n_components_actual = self.svd_.n_components
        return [f'text_svd_{i}' for i in range(n_components_actual)]
    
    def __repr__(self):
        """String representation."""
        if self.is_fitted_:
            return (
                f"TextFeatureExtractor(CPU, fitted=True, "
                f"n_components={self.svd_.n_components})"
            )
        else:
            return (
                f"TextFeatureExtractor(CPU, fitted=False, "
                f"n_components={self.n_components})"
            )
