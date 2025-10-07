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


def process_data(df, verbose=True, expected_features=None):
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
    # ENGINEERED FEATURES FOR TREE-BASED MODELS
    # =============================================================================
    
    if verbose:
        print("\nEngineering additional features...")
    
    # First, combine processed columns so far to access them for feature engineering
    X_temp = pd.concat(processed_columns, axis=1) if processed_columns else pd.DataFrame(index=X.index)
    
    # --- Normalize activity by account lifetime ---
    if 'followers_count' in X_temp.columns and 'account_age_days' in X_temp.columns:
        followers_per_day = X_temp['followers_count'] / (X_temp['account_age_days'] + 1)
        processed_columns.append(followers_per_day.rename('followers_per_day'))
        if verbose:
            print(f"  - followers_per_day: mean={followers_per_day.mean():.4f}")
    
    if 'friends_count' in X_temp.columns and 'account_age_days' in X_temp.columns:
        friends_per_day = X_temp['friends_count'] / (X_temp['account_age_days'] + 1)
        processed_columns.append(friends_per_day.rename('friends_per_day'))
        if verbose:
            print(f"  - friends_per_day: mean={friends_per_day.mean():.4f}")
    
    if 'statuses_count' in X_temp.columns and 'account_age_days' in X_temp.columns:
        statuses_per_day = X_temp['statuses_count'] / (X_temp['account_age_days'] + 1)
        processed_columns.append(statuses_per_day.rename('statuses_per_day'))
        if verbose:
            print(f"  - statuses_per_day: mean={statuses_per_day.mean():.4f}")
    
    if 'favourites_count' in X_temp.columns and 'account_age_days' in X_temp.columns:
        favourites_per_day = X_temp['favourites_count'] / (X_temp['account_age_days'] + 1)
        processed_columns.append(favourites_per_day.rename('favourites_per_day'))
        if verbose:
            print(f"  - favourites_per_day: mean={favourites_per_day.mean():.4f}")
    
    # --- Ratios between social metrics ---
    if 'followers_count' in X_temp.columns and 'friends_count' in X_temp.columns:
        followers_to_friends = X_temp['followers_count'] / (X_temp['friends_count'] + 1)
        processed_columns.append(followers_to_friends.rename('followers_to_friends'))
        if verbose:
            print(f"  - followers_to_friends: mean={followers_to_friends.mean():.4f}")
    
    if 'friends_count' in X_temp.columns and 'followers_count' in X_temp.columns:
        friends_to_followers = X_temp['friends_count'] / (X_temp['followers_count'] + 1)
        processed_columns.append(friends_to_followers.rename('friends_to_followers'))
        if verbose:
            print(f"  - friends_to_followers: mean={friends_to_followers.mean():.4f}")
    
    if 'favourites_count' in X_temp.columns and 'statuses_count' in X_temp.columns:
        favourites_to_statuses = X_temp['favourites_count'] / (X_temp['statuses_count'] + 1)
        processed_columns.append(favourites_to_statuses.rename('favourites_to_statuses'))
        if verbose:
            print(f"  - favourites_to_statuses: mean={favourites_to_statuses.mean():.4f}")
    
    # --- Binary × Continuous hybrids ---
    if 'default_profile_image' in X_temp.columns and 'followers_count' in X_temp.columns:
        default_profile_image_followers = X_temp['default_profile_image'] * X_temp['followers_count']
        processed_columns.append(default_profile_image_followers.rename('default_profile_image_followers'))
        if verbose:
            print(f"  - default_profile_image_followers: mean={default_profile_image_followers.mean():.4f}")
    
    if 'verified' in X_temp.columns and 'followers_count' in X_temp.columns:
        verified_followers = X_temp['verified'] * X_temp['followers_count']
        processed_columns.append(verified_followers.rename('verified_followers'))
        if verbose:
            print(f"  - verified_followers: mean={verified_followers.mean():.4f}")
    
    if 'verified' in X_temp.columns and 'statuses_count' in X_temp.columns:
        verified_statuses = X_temp['verified'] * X_temp['statuses_count']
        processed_columns.append(verified_statuses.rename('verified_statuses'))
        if verbose:
            print(f"  - verified_statuses: mean={verified_statuses.mean():.4f}")
    
    # --- Length-based ratios ---
    if 'description_len' in X_temp.columns and 'screen_name_len' in X_temp.columns:
        desc_to_screenname_len = X_temp['description_len'] / (X_temp['screen_name_len'] + 1)
        processed_columns.append(desc_to_screenname_len.rename('desc_to_screenname_len'))
        if verbose:
            print(f"  - desc_to_screenname_len: mean={desc_to_screenname_len.mean():.4f}")
    
    if 'description_len' in X_temp.columns and 'location_len' in X_temp.columns:
        desc_to_location_len = X_temp['description_len'] / (X_temp['location_len'] + 1)
        processed_columns.append(desc_to_location_len.rename('desc_to_location_len'))
        if verbose:
            print(f"  - desc_to_location_len: mean={desc_to_location_len.mean():.4f}")
    
    # --- Profile completeness score ---
    profile_quality_score = pd.Series(0, index=X.index, name='profile_quality_score')
    
    if 'description_len' in X_temp.columns:
        profile_quality_score += (X_temp['description_len'] > 0).astype(int)
    
    if 'location_len' in X_temp.columns:
        profile_quality_score += (X_temp['location_len'] > 0).astype(int)
    
    if 'default_profile' in X_temp.columns:
        profile_quality_score += (1 - X_temp['default_profile'])
    
    if 'default_profile_image' in X_temp.columns:
        profile_quality_score += (1 - X_temp['default_profile_image'])
    
    if 'geo_enabled' in X_temp.columns:
        profile_quality_score += X_temp['geo_enabled']
    
    processed_columns.append(profile_quality_score)
    if verbose:
        print(f"  - profile_quality_score: mean={profile_quality_score.mean():.4f}, max={profile_quality_score.max()}")
    
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
    X_train_processed, feature_list = process_data(train, verbose=True)
    print(f"Training data processed: {X_train_processed.shape}")
    print(f"Feature list length: {len(feature_list)}")
    
    # Example 2: Process test data with expected features (returns only features)
    print("\n=== Processing Test Data ===")
    X_test_processed = process_data(test, verbose=True, expected_features=feature_list)
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


# =============================================================================
# SPLIT PREPROCESSING FOR CROSS-VALIDATION (NO DATA LEAKAGE)
# =============================================================================

def preprocess_data(df, verbose=True, include_statuses_count=True):
    """
    Preprocessing that can be safely done BEFORE cross-validation without data leakage.
    This includes all deterministic transformations that don't depend on statistics from the data.
    
    This function extracts features and performs transformations that are based purely on
    individual data points, not on aggregate statistics across the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (raw data)
    verbose : bool, default=True
        Whether to print progress information
    include_statuses_count : bool, default=True
        Whether to include statuses_count feature
    
    Returns:
    --------
    pd.DataFrame: Preprocessed features (deterministic transformations only, may have missing values)
    """
    
    if verbose:
        print("=== Pre-CV Data Preprocessing (No Leakage) ===")
        print(f"Original dataframe shape: {df.shape}")
    
    X = df.copy()
    processed_columns = []
    
    # =============================================================================
    # BASIC NUMERICAL FEATURES (with non-negative clipping, no imputation)
    # =============================================================================
    
    if verbose:
        print("\nExtracting numerical features...")
    
    numeric_features = ['favourites_count', 'followers_count', 'friends_count',
                       'average_tweets_per_day', 'account_age_days']
    
    if include_statuses_count:
        numeric_features.append('statuses_count')
    
    for feat in numeric_features:
        if feat in X.columns:
            col = X[feat].copy()
            # Only clip non-negative (deterministic), don't impute yet
            col = col.apply(lambda x: max(x, 0) if pd.notna(x) else np.nan)
            processed_columns.append(col.rename(feat))
            if verbose:
                print(f"  - {feat}: extracted (missing: {col.isnull().sum()})")
    
    # =============================================================================
    # BINARY FEATURES (convert to numeric, no imputation)
    # =============================================================================
    
    if verbose:
        print("\nExtracting binary features...")
    
    binary_features = ['default_profile', 'default_profile_image', 'geo_enabled', 'verified']
    
    for feat in binary_features:
        if feat in X.columns:
            col = pd.to_numeric(X[feat], errors='coerce')
            # Clip to 0/1 for valid values, keep NaN as is
            col = col.apply(lambda x: int(np.clip(x, 0, 1)) if pd.notna(x) else np.nan)
            processed_columns.append(col.rename(feat))
            if verbose:
                print(f"  - {feat}: extracted (missing: {col.isnull().sum()})")
    
    # =============================================================================
    # CATEGORICAL FEATURES (no encoding yet, just fill NaN)
    # =============================================================================
    
    if verbose:
        print("\nExtracting categorical features...")
    
    if 'lang' in X.columns:
        lang = X['lang'].copy()
        lang = lang.fillna('unknown')
        processed_columns.append(lang.rename('lang'))
        if verbose:
            print(f"  - lang: extracted ({len(lang.unique())} categories)")
    
    # =============================================================================
    # TEXTUAL FEATURES (all deterministic extractions)
    # =============================================================================
    
    if verbose:
        print("\nExtracting textual features...")
    
    # Length features
    if 'location' in df.columns:
        location_len = df['location'].fillna('').astype(str).str.len()
        processed_columns.append(location_len.rename('location_len'))
        if verbose:
            print(f"  - location_len: mean={location_len.mean():.2f}")
    
    if 'description' in df.columns:
        description_len = df['description'].fillna('').astype(str).str.len()
        processed_columns.append(description_len.rename('description_len'))
        if verbose:
            print(f"  - description_len: mean={description_len.mean():.2f}")
    
    if 'screen_name' in df.columns:
        screen_name_len = df['screen_name'].fillna('').astype(str).str.len()
        processed_columns.append(screen_name_len.rename('screen_name_len'))
        if verbose:
            print(f"  - screen_name_len: mean={screen_name_len.mean():.2f}")
    
    if 'profile_background_image_url' in df.columns:
        bg_url_len = df['profile_background_image_url'].fillna('').astype(str).str.len()
        processed_columns.append(bg_url_len.rename('profile_background_image_url_len'))
        if verbose:
            print(f"  - profile_background_image_url_len: mean={bg_url_len.mean():.2f}")
    
    if 'profile_image_url' in df.columns:
        profile_url_len = df['profile_image_url'].fillna('').astype(str).str.len()
        processed_columns.append(profile_url_len.rename('profile_image_url_len'))
        if verbose:
            print(f"  - profile_image_url_len: mean={profile_url_len.mean():.2f}")
    
    # Binary pattern features
    if 'description' in df.columns:
        description = df['description'].fillna('')
        description_has_url = description.astype(str).str.contains(
            r'http[s]?://|www\.|\.com|\.net|\.org|t\.co', case=False, na=False
        ).astype(int)
        processed_columns.append(description_has_url.rename('description_has_url'))
        if verbose:
            print(f"  - description_has_url: {description_has_url.sum()} accounts ({description_has_url.mean()*100:.1f}%)")
    
    if 'screen_name' in df.columns:
        screen_name_has_numbers = df['screen_name'].fillna('').astype(str).str.contains(r'\d', na=False).astype(int)
        processed_columns.append(screen_name_has_numbers.rename('screen_name_has_numbers'))
        if verbose:
            print(f"  - screen_name_has_numbers: {screen_name_has_numbers.sum()} accounts ({screen_name_has_numbers.mean()*100:.1f}%)")
    
    if 'location' in df.columns:
        location = df['location']
        location_is_empty = (location.isnull() | (location.astype(str).str.strip() == '') | 
                            (location.astype(str).str.lower() == 'unknown')).astype(int)
        processed_columns.append(location_is_empty.rename('location_is_empty'))
        if verbose:
            print(f"  - location_is_empty: {location_is_empty.sum()} accounts ({location_is_empty.mean()*100:.1f}%)")
    
    if 'description' in df.columns:
        description_is_empty = (df['description'].isnull() | 
                               (df['description'].astype(str).str.strip() == '')).astype(int)
        processed_columns.append(description_is_empty.rename('description_is_empty'))
        if verbose:
            print(f"  - description_is_empty: {description_is_empty.sum()} accounts ({description_is_empty.mean()*100:.1f}%)")
    
    # =============================================================================
    # ENGINEERED FEATURES (all deterministic calculations)
    # =============================================================================
    
    if verbose:
        print("\nEngineering features...")
    
    # Combine processed columns to access them for feature engineering
    X_temp = pd.concat(processed_columns, axis=1) if processed_columns else pd.DataFrame(index=X.index)
    
    # Per-day metrics
    if 'followers_count' in X_temp.columns and 'account_age_days' in X_temp.columns:
        followers_per_day = X_temp['followers_count'] / (X_temp['account_age_days'] + 1)
        processed_columns.append(followers_per_day.rename('followers_per_day'))
        if verbose:
            print(f"  - followers_per_day: mean={followers_per_day.mean():.4f}")
    
    if 'friends_count' in X_temp.columns and 'account_age_days' in X_temp.columns:
        friends_per_day = X_temp['friends_count'] / (X_temp['account_age_days'] + 1)
        processed_columns.append(friends_per_day.rename('friends_per_day'))
        if verbose:
            print(f"  - friends_per_day: mean={friends_per_day.mean():.4f}")
    
    if 'statuses_count' in X_temp.columns and 'account_age_days' in X_temp.columns:
        statuses_per_day = X_temp['statuses_count'] / (X_temp['account_age_days'] + 1)
        processed_columns.append(statuses_per_day.rename('statuses_per_day'))
        if verbose:
            print(f"  - statuses_per_day: mean={statuses_per_day.mean():.4f}")
    
    if 'favourites_count' in X_temp.columns and 'account_age_days' in X_temp.columns:
        favourites_per_day = X_temp['favourites_count'] / (X_temp['account_age_days'] + 1)
        processed_columns.append(favourites_per_day.rename('favourites_per_day'))
        if verbose:
            print(f"  - favourites_per_day: mean={favourites_per_day.mean():.4f}")
    
    # Ratio features
    if 'followers_count' in X_temp.columns and 'friends_count' in X_temp.columns:
        followers_to_friends = X_temp['followers_count'] / (X_temp['friends_count'] + 1)
        processed_columns.append(followers_to_friends.rename('followers_to_friends'))
        if verbose:
            print(f"  - followers_to_friends: mean={followers_to_friends.mean():.4f}")
    
    if 'friends_count' in X_temp.columns and 'followers_count' in X_temp.columns:
        friends_to_followers = X_temp['friends_count'] / (X_temp['followers_count'] + 1)
        processed_columns.append(friends_to_followers.rename('friends_to_followers'))
        if verbose:
            print(f"  - friends_to_followers: mean={friends_to_followers.mean():.4f}")
    
    if 'favourites_count' in X_temp.columns and 'statuses_count' in X_temp.columns:
        favourites_to_statuses = X_temp['favourites_count'] / (X_temp['statuses_count'] + 1)
        processed_columns.append(favourites_to_statuses.rename('favourites_to_statuses'))
        if verbose:
            print(f"  - favourites_to_statuses: mean={favourites_to_statuses.mean():.4f}")
    
    # Binary × Continuous hybrids
    if 'default_profile_image' in X_temp.columns and 'followers_count' in X_temp.columns:
        default_profile_image_followers = X_temp['default_profile_image'] * X_temp['followers_count']
        processed_columns.append(default_profile_image_followers.rename('default_profile_image_followers'))
        if verbose:
            print(f"  - default_profile_image_followers: mean={default_profile_image_followers.mean():.4f}")
    
    if 'verified' in X_temp.columns and 'followers_count' in X_temp.columns:
        verified_followers = X_temp['verified'] * X_temp['followers_count']
        processed_columns.append(verified_followers.rename('verified_followers'))
        if verbose:
            print(f"  - verified_followers: mean={verified_followers.mean():.4f}")
    
    if 'verified' in X_temp.columns and 'statuses_count' in X_temp.columns:
        verified_statuses = X_temp['verified'] * X_temp['statuses_count']
        processed_columns.append(verified_statuses.rename('verified_statuses'))
        if verbose:
            print(f"  - verified_statuses: mean={verified_statuses.mean():.4f}")
    
    # Length ratios
    if 'description_len' in X_temp.columns and 'screen_name_len' in X_temp.columns:
        desc_to_screenname_len = X_temp['description_len'] / (X_temp['screen_name_len'] + 1)
        processed_columns.append(desc_to_screenname_len.rename('desc_to_screenname_len'))
        if verbose:
            print(f"  - desc_to_screenname_len: mean={desc_to_screenname_len.mean():.4f}")
    
    if 'description_len' in X_temp.columns and 'location_len' in X_temp.columns:
        desc_to_location_len = X_temp['description_len'] / (X_temp['location_len'] + 1)
        processed_columns.append(desc_to_location_len.rename('desc_to_location_len'))
        if verbose:
            print(f"  - desc_to_location_len: mean={desc_to_location_len.mean():.4f}")
    
    # Profile quality score
    profile_quality_score = pd.Series(0, index=X.index, name='profile_quality_score')
    
    if 'description_len' in X_temp.columns:
        profile_quality_score += (X_temp['description_len'] > 0).astype(int)
    if 'location_len' in X_temp.columns:
        profile_quality_score += (X_temp['location_len'] > 0).astype(int)
    if 'default_profile' in X_temp.columns:
        profile_quality_score += (1 - X_temp['default_profile'].fillna(0))
    if 'default_profile_image' in X_temp.columns:
        profile_quality_score += (1 - X_temp['default_profile_image'].fillna(0))
    if 'geo_enabled' in X_temp.columns:
        profile_quality_score += X_temp['geo_enabled'].fillna(0)
    
    processed_columns.append(profile_quality_score)
    if verbose:
        print(f"  - profile_quality_score: mean={profile_quality_score.mean():.4f}")
    
    # =============================================================================
    # COMBINE ALL FEATURES
    # =============================================================================
    
    X_processed = pd.concat(processed_columns, axis=1) if processed_columns else pd.DataFrame(index=X.index)
    
    if verbose:
        missing_values = X_processed.isnull().sum().sum()
        print(f"\n=== Pre-CV Preprocessing Complete ===")
        print(f"Shape: {X_processed.shape}")
        print(f"Missing values: {missing_values} (will be imputed in-fold)")
    
    return X_processed


def in_fold_process_data(X_train, X_val, 
                         imputation_strategy='median',
                         categorical_encoding='onehot',
                         lang_frequency_threshold=0,
                         scale_numerical=False,
                         verbose=True):
    """
    Preprocessing that must be done WITHIN each CV fold to avoid data leakage.
    This includes imputation and encoding that depend on training fold statistics.
    
    This function should be called separately for each CV fold, fitting on the
    training fold and transforming both training and validation folds.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training fold data (output from preprocess_data)
    X_val : pd.DataFrame
        Validation fold data (output from preprocess_data)
    imputation_strategy : str, default='median'
        Strategy for numerical imputation ('mean', 'median', 'constant')
    categorical_encoding : str, default='onehot'
        Encoding method for categorical features ('onehot' or 'label')
    lang_frequency_threshold : int, default=0
        Minimum frequency for language categories (lower frequency -> 'other')
    scale_numerical : bool, default=False
        Whether to apply StandardScaler to numerical features (can help some models)
    verbose : bool, default=True
        Whether to print progress information
    
    Returns:
    --------
    X_train_processed : pd.DataFrame
        Fully processed training fold
    X_val_processed : pd.DataFrame
        Fully processed validation fold
    imputation_values : dict
        Dictionary of imputation values used (for later use on test data)
    """
    
    if verbose:
        print("=== In-Fold Processing (Fit on Train, Transform Train & Val) ===")
        print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
        if scale_numerical:
            print("Numerical scaling: ENABLED (StandardScaler)")
        else:
            print("Numerical scaling: DISABLED")
    
    X_train = X_train.copy()
    X_val = X_val.copy()
    
    imputation_values = {}
    
    # =============================================================================
    # NUMERICAL IMPUTATION (fit on train, apply to both)
    # =============================================================================
    
    if verbose:
        print("\nImputing numerical features...")
    
    numeric_features = ['favourites_count', 'followers_count', 'friends_count',
                       'average_tweets_per_day', 'account_age_days', 'statuses_count',
                       'followers_per_day', 'friends_per_day', 'statuses_per_day', 
                       'favourites_per_day', 'followers_to_friends', 'friends_to_followers',
                       'favourites_to_statuses', 'default_profile_image_followers',
                       'verified_followers', 'verified_statuses', 'desc_to_screenname_len',
                       'desc_to_location_len']
    
    for feat in numeric_features:
        if feat in X_train.columns and X_train[feat].isnull().any():
            # Compute imputation value from TRAINING fold only
            if imputation_strategy == 'median':
                impute_val = X_train[feat].median()
            elif imputation_strategy == 'mean':
                impute_val = X_train[feat].mean()
            else:
                impute_val = 0
            
            imputation_values[feat] = impute_val
            
            # Apply to both train and val
            X_train[feat] = X_train[feat].fillna(impute_val)
            if feat in X_val.columns:
                X_val[feat] = X_val[feat].fillna(impute_val)
            
            if verbose:
                print(f"  - {feat}: imputed with {impute_val:.4f}")
    
    # =============================================================================
    # BINARY IMPUTATION (fit on train, apply to both)
    # =============================================================================
    
    if verbose:
        print("\nImputing binary features...")
    
    binary_features = ['default_profile', 'default_profile_image', 'geo_enabled', 'verified']
    
    for feat in binary_features:
        if feat in X_train.columns and X_train[feat].isnull().any():
            # Compute mode from TRAINING fold only
            mode_val = X_train[feat].mode()
            mode_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
            
            imputation_values[feat] = mode_val
            
            # Apply to both train and val
            X_train[feat] = X_train[feat].fillna(mode_val).astype(int)
            if feat in X_val.columns:
                X_val[feat] = X_val[feat].fillna(mode_val).astype(int)
            
            if verbose:
                print(f"  - {feat}: imputed with {mode_val}")
    
    # =============================================================================
    # CATEGORICAL ENCODING (fit on train, apply to both)
    # =============================================================================
    
    if verbose:
        print("\nEncoding categorical features...")
    
    if 'lang' in X_train.columns:
        # Group low-frequency languages based on TRAINING fold only
        lang_counts = X_train['lang'].value_counts()
        low_freq_langs = lang_counts[lang_counts < lang_frequency_threshold].index.tolist()
        
        imputation_values['low_freq_langs'] = low_freq_langs
        
        # Apply grouping to both train and val
        X_train['lang'] = X_train['lang'].replace(low_freq_langs, 'other')
        X_val['lang'] = X_val['lang'].replace(low_freq_langs, 'other')
        
        if verbose:
            print(f"  - lang: grouped {len(low_freq_langs)} categories with freq < {lang_frequency_threshold}")
        
        # Encode
        if categorical_encoding == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
            # Fit on TRAINING fold only
            lang_train_encoded = encoder.fit_transform(X_train['lang'].values.reshape(-1, 1))
            lang_val_encoded = encoder.transform(X_val['lang'].values.reshape(-1, 1))
            
            # Create new columns
            lang_categories = encoder.categories_[0]
            imputation_values['lang_categories'] = lang_categories
            imputation_values['lang_encoder'] = encoder
            
            # Drop original lang column
            X_train = X_train.drop(columns=['lang'])
            X_val = X_val.drop(columns=['lang'])
            
            # Add encoded columns
            for i, category in enumerate(lang_categories):
                col_name = f'lang_{category}'
                X_train[col_name] = lang_train_encoded[:, i]
                X_val[col_name] = lang_val_encoded[:, i]
            
            if verbose:
                print(f"  - lang: one-hot encoded into {len(lang_categories)} columns")
        
        elif categorical_encoding == 'label':
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            
            # Fit on TRAINING fold only
            X_train['lang'] = encoder.fit_transform(X_train['lang'])
            
            # Transform validation, handling unknown categories
            X_val['lang'] = X_val['lang'].apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )
            
            imputation_values['lang_encoder'] = encoder
            
            if verbose:
                print(f"  - lang: label encoded into {len(encoder.classes_)} labels")
    
    # =============================================================================
    # NUMERICAL SCALING (optional, fit on train, apply to both)
    # =============================================================================
    
    if scale_numerical:
        from sklearn.preprocessing import StandardScaler
        
        if verbose:
            print("\nScaling numerical features...")
        
        # Identify numerical columns (exclude binary and one-hot encoded)
        numerical_cols = []
        for col in X_train.columns:
            if col.startswith('lang_'):  # Skip one-hot encoded language features
                continue
            # Check if column has more than 2 unique values (not binary)
            if X_train[col].nunique() > 2:
                numerical_cols.append(col)
        
        if numerical_cols:
            scaler = StandardScaler()
            
            # Fit on TRAINING fold only
            X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
            X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
            
            imputation_values['scaler'] = scaler
            imputation_values['scaled_columns'] = numerical_cols
            
            if verbose:
                print(f"  - Scaled {len(numerical_cols)} numerical features")
                print(f"  - Scaled features: {numerical_cols[:5]}{'...' if len(numerical_cols) > 5 else ''}")
        else:
            if verbose:
                print("  - No numerical features to scale")
    
    # =============================================================================
    # FINAL CHECK: ENSURE ALL COLUMNS ARE NUMERIC
    # =============================================================================
    
    if verbose:
        print("\nFinal data validation...")
    
    # Check for any remaining non-numeric columns in train
    non_numeric_train = X_train.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_train:
        if verbose:
            print(f"  WARNING: Non-numeric columns in train: {non_numeric_train}")
            print("  Converting to numeric...")
        for col in non_numeric_train:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_train[col] = X_train[col].fillna(0)
    
    # Check for any remaining non-numeric columns in val
    non_numeric_val = X_val.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_val:
        if verbose:
            print(f"  WARNING: Non-numeric columns in val: {non_numeric_val}")
            print("  Converting to numeric...")
        for col in non_numeric_val:
            X_val[col] = pd.to_numeric(X_val[col], errors='coerce')
            X_val[col] = X_val[col].fillna(0)
    
    # Check for NaN/Inf values
    if np.isnan(X_train.values).any():
        if verbose:
            print("  WARNING: NaN values detected in train, filling with 0...")
        X_train = X_train.fillna(0)
    
    if np.isnan(X_val.values).any():
        if verbose:
            print("  WARNING: NaN values detected in val, filling with 0...")
        X_val = X_val.fillna(0)
    
    if verbose:
        print(f"  ✓ All columns are numeric")
        print(f"\n=== In-Fold Processing Complete ===")
        print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    
    return X_train, X_val, imputation_values
