"""
Data Preprocessing Pipeline for Random Forest Models
Twitter Bot Detection Competition

Simple preprocessing functions for random forest models with configurable parameters.
No transformations (like log scaling) applied to numerical features since tree-based
models handle raw numerical data effectively.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def process_twitter_data_for_rf(df, verbose=True, expected_features=None):
    """
    Complete preprocessing function for Twitter bot detection data using Random Forest models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (can be train or test data)
    verbose : bool, default=True
        Whether to print progress information
    expected_features : list, optional
        List of expected feature column names. If provided, ensures output has exactly these columns.
        Missing columns are added with value 0. Used for test data consistency.
    
    Returns:
    --------
    pd.DataFrame: Processed features (without target if it exists)
    list: List of all feature column names (only returned for training data when expected_features=None)
    """
    
    # =============================================================================
    # PREPROCESSING PARAMETERS (EDIT THESE DIRECTLY)
    # =============================================================================
    
    # Numerical feature parameters
    numeric_imputation_strategy = 'median'  # 'mean', 'median', 'constant'
    
    # Binary feature parameters  
    binary_imputation_strategy = 'most_frequent'  # 'most_frequent', 'constant'
    
    # Categorical feature parameters
    categorical_encoding = 'onehot'  # 'onehot' or 'label'
    
    # Feature inclusion parameters
    include_statuses_count = True  # Whether to include statuses_count (may be derived from account_age_days * average_tweets_per_day)
    
    # =============================================================================
    # FEATURE SELECTION AND PREPARATION
    # =============================================================================
    
    if verbose:
        print("=== Random Forest Data Preprocessing ===")
        print(f"Original dataframe shape: {df.shape}")
    
    X = df.copy()
    
    # Define feature categories
    numeric_features = [
        'favourites_count', 'followers_count', 'friends_count',
        'average_tweets_per_day', 'account_age_days'
    ]
    
    if include_statuses_count:
        numeric_features.append('statuses_count')
    
    binary_features = [
        'default_profile', 'default_profile_image', 'geo_enabled', 'verified'
    ]
    
    categorical_features = ['lang']
    
    # Features to drop (TBC)
    drop_features = [
        'id',  # Redundant with account_age_days
        'created_at',  # Already captured in account_age_days
        'location',  # Textual data, inconsistent format from Twitter
        'description',  # Text feature, complex to process
        'screen_name',  # Text feature, complex to process
        'profile_background_image_url',  # URL, not useful for prediction
        'profile_image_url'  # URL, not useful for prediction
    ]

    if not include_statuses_count:
        drop_features.append('statuses_count')
    
    # Remove features that don't exist in the data
    numeric_features = [f for f in numeric_features if f in X.columns]
    binary_features = [f for f in binary_features if f in X.columns]
    categorical_features = [f for f in categorical_features if f in X.columns]
    drop_features = [f for f in drop_features if f in X.columns]
    
    if verbose:
        print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
        print(f"Binary features ({len(binary_features)}): {binary_features}")
        print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        print(f"Dropping {len(drop_features)} features: {drop_features}")
    
    # Drop irrelevant features
    X = X.drop(columns=drop_features)
    
    # Initialize list to collect processed feature columns
    processed_columns = []
    
    # =============================================================================
    # NUMERICAL FEATURES
    # =============================================================================
    
    if verbose:
        print("\nProcessing numerical features...")
    
    # favourites_count
    if 'favourites_count' in X.columns:
        favourites_count = X['favourites_count'].copy()
        
        # Handle missing values with imputation
        if favourites_count.isnull().any():
            median_val = favourites_count.median()
            favourites_count = favourites_count.fillna(median_val)
            if verbose:
                print(f"  - favourites_count: filled {X['favourites_count'].isnull().sum()} missing values with {median_val}")
        
        # Ensure non-negative values
        favourites_count = np.maximum(favourites_count, 0)
        processed_columns.append(favourites_count.rename('favourites_count'))
    
    # followers_count
    if 'followers_count' in X.columns:
        followers_count = X['followers_count'].copy()
        
        # Handle missing values with median imputation
        if followers_count.isnull().any():
            median_val = followers_count.median()
            followers_count = followers_count.fillna(median_val)
            if verbose:
                print(f"  - followers_count: filled {X['followers_count'].isnull().sum()} missing values with {median_val}")
        
        # Ensure non-negative values
        followers_count = np.maximum(followers_count, 0)
        processed_columns.append(followers_count.rename('followers_count'))
    
    # friends_count
    if 'friends_count' in X.columns:
        friends_count = X['friends_count'].copy()
        
        # Handle missing values with median imputation
        if friends_count.isnull().any():
            median_val = friends_count.median()
            friends_count = friends_count.fillna(median_val)
            if verbose:
                print(f"  - friends_count: filled {X['friends_count'].isnull().sum()} missing values with {median_val}")
        
        # Ensure non-negative values
        friends_count = np.maximum(friends_count, 0)
        processed_columns.append(friends_count.rename('friends_count'))
    
    # average_tweets_per_day
    if 'average_tweets_per_day' in X.columns:
        average_tweets_per_day = X['average_tweets_per_day'].copy()
        
        # Handle missing values with median imputation
        if average_tweets_per_day.isnull().any():
            median_val = average_tweets_per_day.median()
            average_tweets_per_day = average_tweets_per_day.fillna(median_val)
            if verbose:
                print(f"  - average_tweets_per_day: filled {X['average_tweets_per_day'].isnull().sum()} missing values with {median_val}")
        
        # Ensure non-negative values
        average_tweets_per_day = np.maximum(average_tweets_per_day, 0)
        processed_columns.append(average_tweets_per_day.rename('average_tweets_per_day'))
    
    # account_age_days
    if 'account_age_days' in X.columns:
        account_age_days = X['account_age_days'].copy()
        
        # Handle missing values with median imputation
        if account_age_days.isnull().any():
            median_val = account_age_days.median()
            account_age_days = account_age_days.fillna(median_val)
            if verbose:
                print(f"  - account_age_days: filled {X['account_age_days'].isnull().sum()} missing values with {median_val}")
        
        # Ensure non-negative values
        account_age_days = np.maximum(account_age_days, 0)
        processed_columns.append(account_age_days.rename('account_age_days'))
    
    # statuses_count (optional)
    if include_statuses_count and 'statuses_count' in X.columns:
        statuses_count = X['statuses_count'].copy()
        
        # Handle missing values with median imputation
        if statuses_count.isnull().any():
            median_val = statuses_count.median()
            statuses_count = statuses_count.fillna(median_val)
            if verbose:
                print(f"  - statuses_count: filled {X['statuses_count'].isnull().sum()} missing values with {median_val}")
        
        # Ensure non-negative values
        statuses_count = np.maximum(statuses_count, 0)
        processed_columns.append(statuses_count.rename('statuses_count'))
    
    # =============================================================================
    # BINARY FEATURES
    # =============================================================================
    
    if verbose:
        print("\nProcessing binary features")
    
    # default_profile
    if 'default_profile' in X.columns:
        default_profile = X['default_profile'].copy()
        
        # Convert to numeric and handle missing values
        default_profile = pd.to_numeric(default_profile, errors='coerce')
        if default_profile.isnull().any():
            mode_val = X['default_profile'].mode()
            mode_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
            default_profile = default_profile.fillna(mode_val)
            if verbose:
                print(f"  - default_profile: filled {X['default_profile'].isnull().sum()} missing values with {mode_val}")
        
        # Ensure binary 0/1 format
        default_profile = np.clip(default_profile, 0, 1).astype(int)
        processed_columns.append(default_profile.rename('default_profile'))
    
    # default_profile_image
    if 'default_profile_image' in X.columns:
        default_profile_image = X['default_profile_image'].copy()
        
        # Convert to numeric and handle missing values
        default_profile_image = pd.to_numeric(default_profile_image, errors='coerce')
        if default_profile_image.isnull().any():
            mode_val = X['default_profile_image'].mode()
            mode_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
            default_profile_image = default_profile_image.fillna(mode_val)
            if verbose:
                print(f"  - default_profile_image: filled {X['default_profile_image'].isnull().sum()} missing values with {mode_val}")
        
        # Ensure binary 0/1 format
        default_profile_image = np.clip(default_profile_image, 0, 1).astype(int)
        processed_columns.append(default_profile_image.rename('default_profile_image'))
    
    # geo_enabled
    if 'geo_enabled' in X.columns:
        geo_enabled = X['geo_enabled'].copy()
        
        # Convert to numeric and handle missing values
        geo_enabled = pd.to_numeric(geo_enabled, errors='coerce')
        if geo_enabled.isnull().any():
            mode_val = X['geo_enabled'].mode()
            mode_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
            geo_enabled = geo_enabled.fillna(mode_val)
            if verbose:
                print(f"  - geo_enabled: filled {X['geo_enabled'].isnull().sum()} missing values with {mode_val}")
        
        # Ensure binary 0/1 format
        geo_enabled = np.clip(geo_enabled, 0, 1).astype(int)
        processed_columns.append(geo_enabled.rename('geo_enabled'))
    
    # verified
    if 'verified' in X.columns:
        verified = X['verified'].copy()
        
        # Convert to numeric and handle missing values
        verified = pd.to_numeric(verified, errors='coerce')
        if verified.isnull().any():
            mode_val = X['verified'].mode()
            mode_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
            verified = verified.fillna(mode_val)
            if verbose:
                print(f"  - verified: filled {X['verified'].isnull().sum()} missing values with {mode_val}")
        
        # Ensure binary 0/1 format
        verified = np.clip(verified, 0, 1).astype(int)
        processed_columns.append(verified.rename('verified'))
    
    # =============================================================================
    # CATEGORICAL FEATURES
    # =============================================================================
    
    if verbose:
        print("\nProcessing categorical features...")
    
    # lang (language)
    if 'lang' in X.columns:
        lang = X['lang'].copy()
        
        # Handle missing values by creating 'unknown' category
        lang = lang.fillna('unknown')
        if verbose and X['lang'].isnull().any():
            print(f"  - lang: filled {X['lang'].isnull().sum()} missing values with 'unknown'")
        
        # Group low-frequency categories as 'other' (minimum frequency = 100)
        threshold = 0
        lang_counts = lang.value_counts()
        low_freq_langs = lang_counts[lang_counts < threshold].index.tolist()
        lang = lang.replace(low_freq_langs, 'other')
        
        if verbose:
            n_before = len(lang_counts)
            n_after = len(lang.unique())
            n_grouped = len(low_freq_langs)
            print(f"  - lang: {n_before} -> {n_after} categories (grouped {n_grouped} with freq < {threshold})")
        
        # One-hot encode language feature
        if categorical_encoding == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            lang_encoded = encoder.fit_transform(lang.values.reshape(-1, 1))
            
            # Create column names for each language category
            lang_categories = encoder.categories_[0]
            lang_columns = []
            for i, category in enumerate(lang_categories):
                col_name = f'lang_{category}'
                lang_col = pd.Series(lang_encoded[:, i], index=X.index, name=col_name)
                lang_columns.append(lang_col)
                processed_columns.append(lang_col)
            
            if verbose:
                print(f"  - lang: created {len(lang_columns)} one-hot encoded columns")
        
        elif categorical_encoding == 'label':
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            lang_encoded = pd.Series(encoder.fit_transform(lang), index=X.index, name='lang')
            processed_columns.append(lang_encoded)
            
            if verbose:
                print(f"  - lang: label encoded into {len(encoder.classes_)} labels")
    
    # =============================================================================
    # TEXTUAL FEATURES
    # =============================================================================
    
    if verbose:
        print("\nProcessing textual features...")
    
    # Extract textual features from original df before dropping columns
    
    # location_len - Length of location field
    # Bots often have empty or very short/generic location names
    if 'location' in df.columns:
        location = df['location'].copy()
        location = location.fillna('')  # Fill NaN with empty string
        location_len = location.astype(str).str.len()
        processed_columns.append(location_len.rename('location_len'))
        if verbose:
            print(f"  - location_len: mean={location_len.mean():.2f}, max={location_len.max()}")
    
    # description_len - Length of profile description
    # Bots typically have very short, generic, or missing descriptions
    if 'description' in df.columns:
        description = df['description'].copy()
        description = description.fillna('')  # Fill NaN with empty string
        description_len = description.astype(str).str.len()
        processed_columns.append(description_len.rename('description_len'))
        if verbose:
            print(f"  - description_len: mean={description_len.mean():.2f}, max={description_len.max()}")
    
    # screen_name_len - Length of username/handle
    # Bot usernames often follow patterns (very short, or very long with random characters)
    if 'screen_name' in df.columns:
        screen_name = df['screen_name'].copy()
        screen_name = screen_name.fillna('')  # Fill NaN with empty string
        screen_name_len = screen_name.astype(str).str.len()
        processed_columns.append(screen_name_len.rename('screen_name_len'))
        if verbose:
            print(f"  - screen_name_len: mean={screen_name_len.mean():.2f}, max={screen_name_len.max()}")
    
    # profile_background_image_url_len - Length of background image URL
    # Bots often use default backgrounds (shorter URLs) or no custom backgrounds
    if 'profile_background_image_url' in df.columns:
        bg_image_url = df['profile_background_image_url'].copy()
        bg_image_url = bg_image_url.fillna('')  # Fill NaN with empty string
        profile_background_image_url_len = bg_image_url.astype(str).str.len()
        processed_columns.append(profile_background_image_url_len.rename('profile_background_image_url_len'))
        if verbose:
            print(f"  - profile_background_image_url_len: mean={profile_background_image_url_len.mean():.2f}, max={profile_background_image_url_len.max()}")
    
    # profile_image_url_len - Length of profile image URL
    # Bots often use default profile images (shorter URLs) or follow URL patterns
    if 'profile_image_url' in df.columns:
        profile_image_url = df['profile_image_url'].copy()
        profile_image_url = profile_image_url.fillna('')  # Fill NaN with empty string
        profile_image_url_len = profile_image_url.astype(str).str.len()
        processed_columns.append(profile_image_url_len.rename('profile_image_url_len'))
        if verbose:
            print(f"  - profile_image_url_len: mean={profile_image_url_len.mean():.2f}, max={profile_image_url_len.max()}")
    
    # Additional textual features
    
    # description_has_url - Whether description contains URLs (common in spam bots)
    if 'description' in df.columns:
        description = df['description'].copy()
        description = description.fillna('')
        # Check for common URL patterns
        description_has_url = description.astype(str).str.contains(
            r'http[s]?://|www\.|\.com|\.net|\.org|t\.co', 
            case=False, 
            na=False
        ).astype(int)
        processed_columns.append(description_has_url.rename('description_has_url'))
        if verbose:
            print(f"  - description_has_url: {description_has_url.sum()} accounts have URLs in description ({description_has_url.mean()*100:.1f}%)")
    
    # screen_name_has_numbers - Whether username contains numbers (common bot pattern)
    if 'screen_name' in df.columns:
        screen_name = df['screen_name'].copy()
        screen_name = screen_name.fillna('')
        screen_name_has_numbers = screen_name.astype(str).str.contains(r'\d', na=False).astype(int)
        processed_columns.append(screen_name_has_numbers.rename('screen_name_has_numbers'))
        if verbose:
            print(f"  - screen_name_has_numbers: {screen_name_has_numbers.sum()} accounts have numbers in username ({screen_name_has_numbers.mean()*100:.1f}%)")
    
    # location_is_empty - Whether location field is empty (many bots don't set location)
    if 'location' in df.columns:
        location = df['location'].copy()
        location_is_empty = (location.isnull() | (location.astype(str).str.strip() == '') | (location.astype(str).str.lower() == 'unknown')).astype(int)
        processed_columns.append(location_is_empty.rename('location_is_empty'))
        if verbose:
            print(f"  - location_is_empty: {location_is_empty.sum()} accounts have empty/unknown location ({location_is_empty.mean()*100:.1f}%)")
    
    # description_is_empty - Whether description is empty (many bots have no description)
    if 'description' in df.columns:
        description = df['description'].copy()
        description_is_empty = (description.isnull() | (description.astype(str).str.strip() == '')).astype(int)
        processed_columns.append(description_is_empty.rename('description_is_empty'))
        if verbose:
            print(f"  - description_is_empty: {description_is_empty.sum()} accounts have empty description ({description_is_empty.mean()*100:.1f}%)")
    
    # =============================================================================
    # COMBINE ALL PROCESSED FEATURES
    # =============================================================================
    
    if verbose:
        print(f"\nCombining {len(processed_columns)} processed feature columns...")
    
    # Combine all processed features into final dataframe
    if processed_columns:
        X_processed = pd.concat(processed_columns, axis=1)
    else:
        X_processed = pd.DataFrame(index=X.index)
    
    # Final checks
    missing_values = X_processed.isnull().sum().sum()
    
    # Handle expected features for test data consistency
    if expected_features is not None:
        if verbose:
            print(f"\nEnsuring consistency with expected features...")
            print(f"Expected features: {len(expected_features)}")
            print(f"Current features: {len(X_processed.columns)}")
        
        # Add missing columns with zeros
        missing_cols = set(expected_features) - set(X_processed.columns)
        for col in missing_cols:
            X_processed[col] = 0
            if verbose:
                print(f"  - Added missing column '{col}' with value 0")
        
        # Remove extra columns (shouldn't happen but safety check)
        extra_cols = set(X_processed.columns) - set(expected_features)
        if extra_cols:
            X_processed = X_processed.drop(columns=list(extra_cols))
            if verbose:
                print(f"  - Removed extra columns: {list(extra_cols)}")
        
        # Reorder columns to match expected order
        X_processed = X_processed[expected_features]
        
        if verbose:
            print(f"  - Final shape after consistency check: {X_processed.shape}")
    
    if verbose:
        print(f"\nFinal results:")
        print(f"  - Processed dataframe shape: {X_processed.shape}")
        print(f"  - Missing values: {missing_values}")
        print("=== Preprocessing Complete ===")
    
    # Return processed features and feature list (for training data)
    if expected_features is None:
        # This is training data - return features and feature list
        return X_processed, list(X_processed.columns)
    else:
        # This is test data - return only features
        return X_processed


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the preprocessing pipeline.
    Run this script to test the preprocessing on your data.
    """
    
    # Load data
    print("Loading data...")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    # Example 1: Process training data (returns features and feature list)
    print("\n=== Processing Training Data ===")
    X_train_processed, feature_list = process_twitter_data_for_rf(train, verbose=True)
    print(f"Training data processed: {X_train_processed.shape}")
    print(f"Feature list length: {len(feature_list)}")
    
    # Example 2: Process test data with expected features (returns only features)
    print("\n=== Processing Test Data ===")
    X_test_processed = process_twitter_data_for_rf(test, verbose=True, expected_features=feature_list)
    print(f"Test data processed: {X_test_processed.shape}")
    
    # Verify consistency
    print(f"\n=== Consistency Check ===")
    print(f"Same shapes: {X_train_processed.shape[1] == X_test_processed.shape[1]}")
    print(f"Same columns: {list(X_train_processed.columns) == list(X_test_processed.columns)}")
    
    print(f"\nProcessed shapes:")
    print(f"  - Training features: {X_train_processed.shape}")
    print(f"  - Test features: {X_test_processed.shape}")
    print("\nPreprocessing pipeline ready for Random Forest modeling!")
    print("\nTo modify preprocessing parameters, edit the variables at the top of the function:")
