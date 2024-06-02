# Data Processing and Feature Engineering Tools Library

# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, Normalizer
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

# Function to load data
def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

# Function to handle missing values
def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Args:
        data (pd.DataFrame): The data with missing values.
        strategy (str): Strategy to handle missing values ('mean', 'median', 'most_frequent', 'drop').
        
    Returns:
        pd.DataFrame: Data with missing values handled.
    """
    if strategy == 'drop':
        return data.dropna()
    else:
        imputer = SimpleImputer(strategy=strategy)
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Function to drop columns with zero variance
def drop_zero_variance_columns(data):
    """
    Drop columns with zero variance.
    
    Args:
        data (pd.DataFrame): The data with columns to check for zero variance.
        
    Returns:
        pd.DataFrame: Data with zero variance columns dropped.
    """
    return data.loc[:, data.var() != 0]

# Function to scale features
def scale_features(data, method='standard'):
    """
    Scale features in the dataset.
    
    Args:
        data (pd.DataFrame): The data with features to scale.
        method (str): Scaling method ('standard', 'minmax').
        
    Returns:
        np.ndarray: Scaled data.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    return scaler.fit_transform(data)

# Function to normalize features
def normalize_features(data, norm='l2'):
    """
    Normalize features in the dataset.
    
    Args:
        data (pd.DataFrame): The data with features to normalize.
        norm (str): Normalization method ('l1', 'l2', 'max').
        
    Returns:
        np.ndarray: Normalized data.
    """
    normalizer = Normalizer(norm=norm)
    return normalizer.fit_transform(data)

# Function to scale specific columns
def scale_specific_columns(data, columns, method='standard'):
    """
    Scale specific columns in the dataset.
    
    Args:
        data (pd.DataFrame): The data with columns to scale.
        columns (list): List of columns to scale.
        method (str): Scaling method ('standard', 'minmax').
        
    Returns:
        pd.DataFrame: Data with scaled columns.
    """
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

# Function to encode categorical features
def encode_categorical_features(data, columns, method='onehot'):
    """
    Encode categorical features in the dataset.
    
    Args:
        data (pd.DataFrame): The data with categorical features to encode.
        columns (list): List of columns to encode.
        method (str): Encoding method ('onehot', 'label').
        
    Returns:
        pd.DataFrame: Data with encoded categorical features.
    """
    if method == 'onehot':
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded = encoder.fit_transform(data[columns])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns))
        return pd.concat([data.drop(columns, axis=1), encoded_df], axis=1)
    elif method == 'label':
        encoder = LabelEncoder()
        for col in columns:
            data[col] = encoder.fit_transform(data[col])
        return data

# Function to detect and remove outliers
def remove_outliers(data, contamination=0.01):
    """
    Detect and remove outliers using Isolation Forest.
    
    Args:
        data (pd.DataFrame): The data with potential outliers.
        contamination (float): The proportion of outliers in the data set.
        
    Returns:
        pd.DataFrame: Data with outliers removed.
    """
    iso = IsolationForest(contamination=contamination)
    yhat = iso.fit_predict(data)
    mask = yhat != -1
    return data[mask]

# Function to detect multicollinearity
def detect_multicollinearity(data, threshold=0.8):
    """
    Detect multicollinearity by checking the correlation matrix.
    
    Args:
        data (pd.DataFrame): The data to check for multicollinearity.
        threshold (float): Correlation threshold to identify multicollinearity.
        
    Returns:
        list: List of column pairs with correlation above the threshold.
    """
    corr_matrix = data.corr().abs()
    high_corr_var = np.where(corr_matrix > threshold)
    high_corr_var = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*high_corr_var) if x != y and x < y]
    return high_corr_var

# Function to create interaction terms
def create_interaction_terms(data, columns):
    """
    Create interaction terms between specified columns.
    
    Args:
        data (pd.DataFrame): The data with columns to create interaction terms.
        columns (list of tuple): List of tuples specifying pairs of columns to create interaction terms.
        
    Returns:
        pd.DataFrame: Data with interaction terms added.
    """
    for col1, col2 in columns:
        data[f'{col1}_{col2}'] = data[col1] * data[col2]
    return data

# Function to select best features
def select_best_features(X, y, k=10, score_func='f_classif'):
    """
    Select the best features based on a scoring function.
    
    Args:
        X (pd.DataFrame or np.ndarray): Feature data.
        y (pd.Series or np.ndarray): Target data.
        k (int): Number of top features to select.
        score_func (str): Scoring function ('f_classif', 'chi2').
        
    Returns:
        np.ndarray: Selected top features.
    """
    if score_func == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=k)
    elif score_func == 'chi2':
        selector = SelectKBest(score_func=chi2, k=k)
    return selector.fit_transform(X, y)

# Function for recursive feature elimination
def recursive_feature_elimination(X, y, n_features_to_select=10):
    """
    Perform recursive feature elimination to select the best features.
    
    Args:
        X (pd.DataFrame or np.ndarray): Feature data.
        y (pd.Series or np.ndarray): Target data.
        n_features_to_select (int): Number of features to select.
        
    Returns:
        np.ndarray: Selected top features.
    """
    model = LogisticRegression()
    selector = RFE(model, n_features_to_select=n_features_to_select)
    return selector.fit_transform(X, y)

# Function for principal component analysis (PCA)
def perform_pca(data, n_components=2):
    """
    Perform principal component analysis (PCA) on the data.
    
    Args:
        data (pd.DataFrame or np.ndarray): Data to perform PCA on.
        n_components (int): Number of principal components to keep.
        
    Returns:
        np.ndarray: Principal components.
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

# Function to generate polynomial features
def generate_polynomial_features(data, degree=2):
    """
    Generate polynomial features from the existing features.
    
    Args:
        data (pd.DataFrame or np.ndarray): Original data.
        degree (int): Degree of the polynomial features.
        
    Returns:
        np.ndarray: Data with polynomial features.
    """
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(data)

# Function to bin continuous data
def bin_continuous_data(data, columns, bins, labels=None):
    """
    Bin continuous data into discrete intervals.
    
    Args:
        data (pd.DataFrame): The data with continuous columns to bin.
        columns (list): List of columns to bin.
        bins (int or list): Number of bins or list of bin edges.
        labels (list, optional): Labels for the bins.
        
    Returns:
        pd.DataFrame: Data with binned columns.
    """
    for col in columns:
        data[col] = pd.cut(data[col], bins=bins, labels=labels)
    return data

# Function to process time series data
def process_time_series_data(data, column, lags):
    """
    Process time series data by creating lag features.
    
    Args:
        data (pd.DataFrame): The time series data.
        column (str): The column to create lag features.
        lags (int): Number of lag features to create.
        
    Returns:
        pd.DataFrame: Data with lag features.
    """
    for lag in range(1, lags + 1):
        data[f'{column}_lag{lag}'] = data[column].shift(lag)
    return data.dropna()

# Function to perform target encoding
def target_encode(data, column, target):
    """
    Perform target encoding on a categorical column.
    
    Args:
        data (pd.DataFrame): The data with the categorical column.
        column (str): The column to target encode.
        target (str): The target variable.
        
    Returns:
        pd.DataFrame: Data with target encoded column.
    """
    target_mean = data.groupby(column)[target].mean()
    data[f'{column}_target_enc'] = data[column].map(target_mean)
    return data.drop(column, axis=1)

# Example usage:
# data = load_data('data.csv')
# data = handle_missing_values(data, strategy='mean')
# data = drop_zero_variance_columns(data)
# scaled_data = scale_features(data, method='standard')
# normalized_data = normalize_features(data, norm='l2')
# data = scale_specific_columns(data, columns=['col1', 'col2'], method='minmax')
# encoded_data = encode_categorical_features(data, columns=['col1', 'col2'], method='onehot')
# cleaned_data = remove_outliers(data, contamination=0.01)
# multicollinearity = detect_multicollinearity(data, threshold=0.8)
# interaction_data = create_interaction_terms(data, columns=[('col1', 'col2'), ('col3', 'col4')])
# selected_features = select_best_features(X, y, k=10, score_func='f_classif')
# rfe_features = recursive_feature_elimination(X, y, n_features_to_select=10)
# pca_data = perform_pca(data, n_components=2)
# poly_data = generate_polynomial_features(data, degree=3)
# binned_data = bin_continuous_data(data, columns=['col1', 'col2'], bins=5, labels=['low', 'medium', 'high'])
# time_series_data = process_time_series_data(data, column='time_series_col', lags=3)
# target_encoded_data = target_encode(data, column='categorical_col', target='target_col')
