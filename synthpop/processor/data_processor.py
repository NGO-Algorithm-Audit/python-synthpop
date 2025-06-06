import pandas as pd
import numpy as np
import warnings
import logging
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

# Set up logging
LOGGER = logging.getLogger(__name__)

class InvalidDataError(Exception):
    """Custom exception for invalid data errors."""
    pass

class DataProcessor:
    """Preprocess and post-process data before and after synthetic data generation.

    Handles:
    - Type conversions (categorical ↔ numerical).
    - Feature transformations for Gaussian Copula.
    - Reverse transformations to restore original data types.
    """

    def __init__(self, metadata, enforce_rounding=True, enforce_min_max_values=True, model_kwargs=None, table_name=None, locales=['en_US']):
        self.metadata = metadata
        self.enforce_rounding = enforce_rounding
        self.enforce_min_max_values = enforce_min_max_values
        self.model_kwargs = model_kwargs or {}
        self.table_name = table_name
        self.locales = locales
        self._fitted = False
        self._prepared_for_fitting = False
        self.encoders = {}  # Stores encoders for categorical columns
        self.scalers = {}  # Stores scalers for numerical columns
        self.original_columns = None  # To restore column order
        self._original_dtypes = None  # Store original dtypes

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the raw data into numerical space."""
        if self._fitted:
            warnings.warn(
                "This model has already been fitted. To use new preprocessed data, "
                "please refit the model using 'fit'."
            )

        self.validate(data)
        self.original_columns = data.columns  # Store original column order
        self._original_dtypes = data.dtypes  # Store original dtypes
        processed_data = self._preprocess(data)

        return processed_data

    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handles encoding, scaling."""
        data = data.copy()

        for col, dtype in self.metadata.items():
            if dtype == "categorical":
                # Choose encoder based on cardinality
                n_unique = len(data[col].unique())
                if n_unique < 10:
                    encoder = LabelEncoder()
                elif n_unique < 50:
                    encoder = OneHotEncoder(sparse=False, drop="first")
                else:
                    # Frequency encoding
                    value_counts = data[col].value_counts(normalize=True)
                    encoder = {'type': 'frequency', 'mapping': value_counts.to_dict()}
                
                transformed_data = self._encode_categorical(data[col], encoder)
                self.encoders[col] = encoder
                data.drop(columns=[col], inplace=True)
                data = pd.concat([data, transformed_data], axis=1)

            elif dtype == "numerical":
                scaler = StandardScaler(with_mean=False, with_std=False)
                data[col] = scaler.fit_transform(data[[col]])
                self.scalers[col] = scaler

            elif dtype == "boolean":
                data[col] = data[col].astype(int)  # Convert True/False to 1/0

            elif dtype == "datetime":
                data[col] = data[col].apply(
                    lambda x: x.timestamp() if pd.notnull(x) else np.nan
                )  # Convert to Unix timestamp
            
            elif dtype == "timedelta": 
                data[col] = pd.to_timedelta(data[col]).dt.total_seconds()

        return data[self.original_columns]

    def postprocess(self, synthetic_data: pd.DataFrame) -> pd.DataFrame:
        """Transform numerical synthetic data back to its original format."""
        synthetic_data = synthetic_data.copy()

        for col, dtype in self.metadata.items():
            if dtype == "categorical" and col in self.encoders:
                encoder = self.encoders[col]
                synthetic_data[col] = self._decode_categorical(synthetic_data[col], encoder)

            elif dtype == "numerical" and col in self.scalers:
                scaler = self.scalers[col]
                synthetic_data[col] = scaler.inverse_transform(synthetic_data[[col]])
                
                # Restore original dtype for numerical columns
                if self._original_dtypes is not None:
                    original_dtype = self._original_dtypes[col]
                    if np.issubdtype(original_dtype, np.integer):
                        synthetic_data[col] = synthetic_data[col].round().astype(original_dtype)

            elif dtype == "boolean":
                synthetic_data[col] = synthetic_data[col].round().astype(bool)

            elif dtype == "datetime":
                synthetic_data[col] = pd.to_datetime(synthetic_data[col], unit='s')

            elif dtype == "timedelta":
                synthetic_data[col] = pd.to_timedelta(synthetic_data[col], unit='s')

        return synthetic_data[self.original_columns]  # Restore original column order

    def validate(self, data: pd.DataFrame):
        """Validate input data."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        missing_columns = set(self.metadata.keys()) - set(data.columns)
        if missing_columns:
            raise InvalidDataError(f"Missing columns: {missing_columns}")

        primary_keys = [col for col, dtype in self.metadata.items() if dtype == "primary_key"]
        for key in primary_keys:
            if data[key].duplicated().any():
                raise InvalidDataError(f"Primary key '{key}' is not unique.")

    def _encode_categorical(self, series: pd.Series, encoder):
        """Encode categorical columns."""
        if isinstance(encoder, LabelEncoder):
            return pd.DataFrame(
                encoder.fit_transform(series), 
                columns=[series.name]
            )
        elif isinstance(encoder, OneHotEncoder):
            encoded_array = encoder.fit_transform(series.values.reshape(-1, 1))
            encoded_df = pd.DataFrame(
                encoded_array, 
                columns=encoder.get_feature_names_out([series.name])
            )
            return encoded_df
        elif isinstance(encoder, dict) and encoder['type'] == 'frequency':
            # Frequency encoding
            encoded_values = series.map(encoder['mapping'])
            return pd.DataFrame(encoded_values, columns=[series.name])
        else:
            raise TypeError(f"Unsupported encoder type: {type(encoder)}")

    def _decode_categorical(self, encoded: pd.Series or pd.DataFrame, encoder):
        """
        Decode categorical columns, snapping any out‐of‐range codes back to the nearest
        valid category (or to NaN), so novel copula values won't blow up.
        """
        # LABEL ENCODER CASE
        if isinstance(encoder, LabelEncoder):
            # Pull out the raw numeric codes (may be floats from copula)
            codes = np.rint(encoded.astype(float)).astype(int)
            max_idx = len(encoder.classes_) - 1

            # Any code outside [0, max_idx] → -1 sentinel
            safe_codes = np.where((codes >= 0) & (codes <= max_idx), codes, -1)

            # Map valid codes back to labels, sentinel→NaN
            decoded = [
                encoder.classes_[c] if c >= 0 else np.nan
                for c in safe_codes
            ]
            return pd.Series(decoded, index=getattr(encoded, "index", None))

        # ONE-HOT ENCODER CASE
        elif isinstance(encoder, OneHotEncoder):
            # Ensure a 2D array of one-hot "scores"
            arr = encoded.values if isinstance(encoded, pd.DataFrame) else np.asarray(encoded)
            if arr.ndim == 1:
                # If someone passed a flat Series, assume the first category axis:
                n_cat = len(encoder.categories_[0])
                arr = arr.reshape(-1, n_cat)

            # Argmax and clip into [0, n_cat-1]
            idx = np.argmax(arr, axis=1)
            max_idx = len(encoder.categories_[0]) - 1
            idx = np.clip(idx, 0, max_idx)

            # Look up the category labels
            cats = encoder.categories_[0]
            return pd.Series(cats[idx], index=getattr(encoded, "index", None))

        # FREQUENCY ENCODER CASE
        elif isinstance(encoder, dict) and encoder['type'] == 'frequency':
            # For frequency encoding, we need to map the encoded values back to categories
            # We'll use the inverse mapping (frequency -> category)
            inverse_mapping = {v: k for k, v in encoder['mapping'].items()}
            # Find the closest frequency for each encoded value
            encoded_values = encoded.values.flatten()
            decoded = []
            for val in encoded_values:
                if pd.isna(val):
                    decoded.append(np.nan)
                else:
                    # Find the category with the closest frequency
                    closest_freq = min(inverse_mapping.keys(), key=lambda x: abs(x - val))
                    decoded.append(inverse_mapping[closest_freq])
            return pd.Series(decoded, index=getattr(encoded, "index", None))

        else:
            raise TypeError(f"Unsupported encoder type: {type(encoder)}")

    def _handle_missing_values(self, series: pd.Series):
        """Handle missing values based on column type."""
        if series.dtype in ["float64", "int64"]:
            return series.fillna(series.median())
        elif series.dtype == "object":
            return series.fillna(series.mode()[0])
        else:
            return series.fillna(0)
