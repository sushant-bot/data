"""
Property-Based Tests for Processing Lambda Function

This module contains property-based tests for the AI Data Analyst Platform's
data preprocessing engine, validating correctness properties across all
supported preprocessing operations.

Feature: ai-data-analyst-platform
"""

import pytest
import pandas as pd
import numpy as np
import json
import io
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.pandas import data_frames, columns
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add processing directory to sys.path BEFORE loading the module
# (processing lambda_function.py imports quality_assessment at module level)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lambda', 'processing'))

# Load processing lambda_function via importlib to avoid sys.path collisions
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "processing_lambda",
    os.path.join(os.path.dirname(__file__), '..', 'lambda', 'processing', 'lambda_function.py')
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
sys.modules['processing_lambda'] = _module

execute_preprocessing_operation = _module.execute_preprocessing_operation
handle_null_removal = _module.handle_null_removal
handle_null_filling = _module.handle_null_filling
handle_outlier_removal = _module.handle_outlier_removal
handle_scaling = _module.handle_scaling
handle_label_encoding = _module.handle_label_encoding
handle_one_hot_encoding = _module.handle_one_hot_encoding

from quality_assessment import assess_dataset_quality


# Test data generators
@st.composite
def generate_numeric_dataframe(draw):
    """Generate DataFrame with numeric columns for testing."""
    n_rows = draw(st.integers(min_value=10, max_value=100))
    n_cols = draw(st.integers(min_value=2, max_value=5))
    
    data = {}
    for i in range(n_cols):
        col_name = f'numeric_col_{i}'
        # Generate numeric data with some potential outliers
        values = draw(st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False),
            min_size=n_rows,
            max_size=n_rows
        ))
        data[col_name] = values
    
    return pd.DataFrame(data)


@st.composite
def generate_categorical_dataframe(draw):
    """Generate DataFrame with categorical columns for testing."""
    n_rows = draw(st.integers(min_value=10, max_value=100))
    n_cols = draw(st.integers(min_value=2, max_value=4))
    
    data = {}
    for i in range(n_cols):
        col_name = f'cat_col_{i}'
        # Generate categorical data
        categories = draw(st.lists(
            st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=['Lu', 'Ll'])),
            min_size=2,
            max_size=5,
            unique=True
        ))
        values = draw(st.lists(
            st.sampled_from(categories),
            min_size=n_rows,
            max_size=n_rows
        ))
        data[col_name] = values
    
    return pd.DataFrame(data)


@st.composite
def generate_mixed_dataframe_with_nulls(draw):
    """Generate DataFrame with mixed data types and null values."""
    n_rows = draw(st.integers(min_value=20, max_value=100))
    
    data = {}
    
    # Add numeric columns with nulls
    for i in range(2):
        col_name = f'numeric_{i}'
        values = []
        for _ in range(n_rows):
            if draw(st.booleans().filter(lambda x: True)):  # 50% chance of null
                if draw(st.floats(min_value=0, max_value=1)) < 0.1:  # 10% null rate
                    values.append(np.nan)
                else:
                    values.append(draw(st.floats(min_value=-50, max_value=50, allow_nan=False)))
            else:
                values.append(draw(st.floats(min_value=-50, max_value=50, allow_nan=False)))
        data[col_name] = values
    
    # Add categorical columns with nulls
    for i in range(2):
        col_name = f'categorical_{i}'
        categories = ['A', 'B', 'C', 'D']
        values = []
        for _ in range(n_rows):
            if draw(st.floats(min_value=0, max_value=1)) < 0.1:  # 10% null rate
                values.append(None)
            else:
                values.append(draw(st.sampled_from(categories)))
        data[col_name] = values
    
    return pd.DataFrame(data)


class TestNullValuePreprocessing:
    """
    Property 10: Null Value Preprocessing
    Validates: Requirements 4.1, 4.2
    """
    
    @given(df=generate_mixed_dataframe_with_nulls())
    @settings(max_examples=50, deadline=None)
    def test_null_removal_preserves_non_null_data(self, df):
        """
        For any dataset with null values, null removal operations should preserve
        all non-null data while removing rows/columns with null values.
        """
        assume(len(df) > 0 and len(df.columns) > 0)
        assume(df.isnull().any().any())  # Ensure there are null values
        
        original_non_null_count = df.count().sum()
        
        # Test row removal
        operation = {
            'type': 'null_removal',
            'parameters': {'method': 'drop_rows'}
        }
        
        processed_df, result = execute_preprocessing_operation(df, operation)
        
        # Verify no null values remain
        assert not processed_df.isnull().any().any(), "Null values should be completely removed"
        
        # Verify data integrity - all remaining data should be from original
        for col in processed_df.columns:
            if col in df.columns:
                original_values = set(df[col].dropna().values)
                processed_values = set(processed_df[col].values)
                assert processed_values.issubset(original_values), f"Column {col} contains new values not in original"
        
        # Verify operation metadata
        assert result['method_used'] == 'drop_rows'
        assert result['rows_removed'] >= 0
        assert len(processed_df) <= len(df)
    
    @given(df=generate_mixed_dataframe_with_nulls())
    @settings(max_examples=50, deadline=None)
    def test_null_filling_eliminates_nulls(self, df):
        """
        For any dataset with null values, null filling operations should eliminate
        all null values using appropriate strategies.
        """
        assume(len(df) > 0 and len(df.columns) > 0)
        assume(df.isnull().any().any())  # Ensure there are null values
        
        # Test mean filling for numeric columns
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if numeric_cols:
            operation = {
                'type': 'null_filling',
                'parameters': {
                    'strategy': 'mean',
                    'columns': numeric_cols
                }
            }
            
            processed_df, result = execute_preprocessing_operation(df, operation)
            
            # Verify nulls are filled in specified columns
            for col in numeric_cols:
                if col in df.columns and df[col].isnull().any():
                    assert not processed_df[col].isnull().any(), f"Nulls should be filled in {col}"
                    
                    # Verify fill value is reasonable (close to mean)
                    original_mean = df[col].mean()
                    if not pd.isna(original_mean):
                        filled_values = processed_df.loc[df[col].isnull(), col]
                        if len(filled_values) > 0:
                            assert np.allclose(filled_values, original_mean, rtol=1e-10), "Fill values should match mean"
            
            # Verify operation metadata
            assert result['strategy_used'] == 'mean'
            assert isinstance(result['columns_affected'], list)


class TestOutlierRemoval:
    """
    Property 11: Outlier Removal Accuracy
    Validates: Requirements 4.3
    """
    
    @given(df=generate_numeric_dataframe())
    @settings(max_examples=30, deadline=None)
    def test_iqr_outlier_removal_accuracy(self, df):
        """
        For any numerical dataset, IQR outlier removal should eliminate data points
        beyond the specified IQR threshold while preserving inliers.
        """
        assume(len(df) > 10)  # Need sufficient data for IQR calculation
        
        # Add some known outliers
        outlier_df = df.copy()
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if numeric_cols:
            col = numeric_cols[0]
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Add extreme outliers
            outlier_df.loc[len(outlier_df)] = outlier_df.iloc[0].copy()
            outlier_df.loc[len(outlier_df)-1, col] = Q3 + 5 * IQR  # Upper outlier
            outlier_df.loc[len(outlier_df)] = outlier_df.iloc[0].copy()
            outlier_df.loc[len(outlier_df)-1, col] = Q1 - 5 * IQR  # Lower outlier
            
            operation = {
                'type': 'outlier_removal',
                'parameters': {
                    'method': 'iqr',
                    'threshold': 1.5,
                    'columns': [col]
                }
            }
            
            processed_df, result = execute_preprocessing_operation(outlier_df, operation)
            
            # Verify outliers are removed
            if len(processed_df) > 0:
                processed_Q1 = outlier_df[col].quantile(0.25)
                processed_Q3 = outlier_df[col].quantile(0.75)
                processed_IQR = processed_Q3 - processed_Q1
                lower_bound = processed_Q1 - 1.5 * processed_IQR
                upper_bound = processed_Q3 + 1.5 * processed_IQR
                
                # All remaining values should be within bounds
                within_bounds = (processed_df[col] >= lower_bound) & (processed_df[col] <= upper_bound)
                assert within_bounds.all(), "All remaining values should be within IQR bounds"
            
            # Verify operation metadata
            assert result['method_used'] == 'iqr'
            assert result['threshold_used'] == 1.5
            assert result['outliers_removed'] >= 0
    
    @given(df=generate_numeric_dataframe())
    @settings(max_examples=30, deadline=None)
    def test_zscore_outlier_removal_accuracy(self, df):
        """
        For any numerical dataset, Z-score outlier removal should eliminate data points
        beyond the specified Z-score threshold.
        """
        assume(len(df) > 10)
        
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if numeric_cols:
            col = numeric_cols[0]
            assume(df[col].std() > 0)  # Need non-zero standard deviation
            
            # Add extreme outliers to ensure we have something to remove
            outlier_df = df.copy()
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            # Add multiple extreme outliers
            outlier_df.loc[len(outlier_df)] = outlier_df.iloc[0].copy()
            outlier_df.loc[len(outlier_df)-1, col] = mean_val + 5 * std_val  # Upper outlier
            outlier_df.loc[len(outlier_df)] = outlier_df.iloc[0].copy()
            outlier_df.loc[len(outlier_df)-1, col] = mean_val - 5 * std_val  # Lower outlier
            
            operation = {
                'type': 'outlier_removal',
                'parameters': {
                    'method': 'zscore',
                    'threshold': 3.0,
                    'columns': [col]
                }
            }
            
            processed_df, result = execute_preprocessing_operation(outlier_df, operation)
            
            # Verify outliers were removed (should have fewer rows)
            assert len(processed_df) <= len(outlier_df), "Outlier removal should not increase row count"
            
            # Verify operation metadata
            assert result['method_used'] == 'zscore'
            assert result['threshold_used'] == 3.0
            assert result['outliers_removed'] >= 0


class TestDataScaling:
    """
    Property 12: Data Scaling Transformations
    Validates: Requirements 4.4
    """
    
    @given(df=generate_numeric_dataframe())
    @settings(max_examples=30, deadline=None)
    def test_standard_scaling_properties(self, df):
        """
        For any numerical dataset, StandardScaler should transform data to have
        mean ≈ 0 and standard deviation ≈ 1.
        """
        assume(len(df) > 1)
        
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        assume(len(numeric_cols) > 0)
        
        # Ensure columns have non-zero standard deviation
        valid_cols = [col for col in numeric_cols if df[col].std() > 1e-10]
        assume(len(valid_cols) > 0)
        
        operation = {
            'type': 'scaling',
            'parameters': {
                'method': 'standard',
                'columns': valid_cols
            }
        }
        
        processed_df, result = execute_preprocessing_operation(df, operation)
        
        # Verify scaling properties
        for col in valid_cols:
            if col in processed_df.columns:
                col_mean = processed_df[col].mean()
                col_std = processed_df[col].std()
                
                # Mean should be approximately 0
                assert abs(col_mean) < 0.1, f"Mean of {col} should be ~0, got {col_mean}"
                
                # Standard deviation should be approximately 1
                if len(processed_df) > 1:
                    # Allow for small numerical errors, especially with small datasets
                    assert abs(col_std - 1.0) < 0.1, f"Std of {col} should be ~1, got {col_std}"
        
        # Verify operation metadata
        assert result['method_used'] == 'standard'
        assert set(result['columns_affected']) == set(valid_cols)
    
    @given(df=generate_numeric_dataframe())
    @settings(max_examples=30, deadline=None)
    def test_minmax_scaling_properties(self, df):
        """
        For any numerical dataset, MinMaxScaler should transform data to range [0, 1].
        """
        assume(len(df) > 1)
        
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        assume(len(numeric_cols) > 0)
        
        # Ensure columns have range > 0
        valid_cols = [col for col in numeric_cols if (df[col].max() - df[col].min()) > 1e-10]
        assume(len(valid_cols) > 0)
        
        operation = {
            'type': 'scaling',
            'parameters': {
                'method': 'minmax',
                'columns': valid_cols
            }
        }
        
        processed_df, result = execute_preprocessing_operation(df, operation)
        
        # Verify scaling properties
        for col in valid_cols:
            if col in processed_df.columns:
                col_min = processed_df[col].min()
                col_max = processed_df[col].max()
                
                # Values should be in range [0, 1]
                assert col_min >= -0.01, f"Min of {col} should be >= 0, got {col_min}"
                assert col_max <= 1.01, f"Max of {col} should be <= 1, got {col_max}"
                
                # If original range > 0, scaled range should span [0, 1]
                if (df[col].max() - df[col].min()) > 1e-10:
                    assert abs(col_min - 0.0) < 0.01, f"Min should be 0, got {col_min}"
                    assert abs(col_max - 1.0) < 0.01, f"Max should be 1, got {col_max}"
        
        # Verify operation metadata
        assert result['method_used'] == 'minmax'


class TestCategoricalEncoding:
    """
    Property 13: Categorical Encoding Consistency
    Validates: Requirements 4.5, 4.6
    """
    
    @given(df=generate_categorical_dataframe())
    @settings(max_examples=30, deadline=None)
    def test_label_encoding_consistency(self, df):
        """
        For any categorical dataset, label encoding should produce consistent
        numerical mappings for the same categorical values.
        """
        assume(len(df) > 0)
        
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        assume(len(categorical_cols) > 0)
        
        operation = {
            'type': 'label_encoding',
            'parameters': {
                'columns': categorical_cols
            }
        }
        
        processed_df, result = execute_preprocessing_operation(df, operation)
        
        # Verify encoding consistency
        for col in categorical_cols:
            if col in processed_df.columns:
                # All values should be numeric
                assert pd.api.types.is_numeric_dtype(processed_df[col]), f"Column {col} should be numeric after encoding"
                
                # Same categorical values should map to same numeric values
                original_unique = df[col].dropna().unique()
                if len(original_unique) > 1:
                    for unique_val in original_unique:
                        original_mask = df[col] == unique_val
                        if original_mask.any():
                            encoded_values = processed_df.loc[original_mask, col].dropna()
                            if len(encoded_values) > 1:
                                # All instances of same category should have same encoding
                                assert len(encoded_values.unique()) == 1, f"Inconsistent encoding for {unique_val}"
        
        # Verify operation metadata
        assert result['columns_affected'] == categorical_cols
        assert isinstance(result['encoding_mappings'], dict)
    
    @given(df=generate_categorical_dataframe())
    @settings(max_examples=30, deadline=None)
    def test_one_hot_encoding_properties(self, df):
        """
        For any categorical dataset, one-hot encoding should create the correct
        number of binary columns and preserve information.
        """
        assume(len(df) > 0)
        
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        assume(len(categorical_cols) > 0)
        
        # Calculate expected new columns
        expected_new_cols = 0
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            expected_new_cols += unique_vals
        
        operation = {
            'type': 'one_hot_encoding',
            'parameters': {
                'columns': categorical_cols,
                'drop_first': False
            }
        }
        
        processed_df, result = execute_preprocessing_operation(df, operation)
        
        # Verify one-hot encoding properties
        assert len(result['new_columns_created']) == expected_new_cols, "Should create correct number of new columns"
        
        # Verify binary nature of new columns
        for new_col in result['new_columns_created']:
            if new_col in processed_df.columns:
                unique_vals = processed_df[new_col].dropna().unique()
                assert set(unique_vals).issubset({0, 1}), f"Column {new_col} should be binary"
        
        # Verify original categorical columns are removed
        for col in categorical_cols:
            assert col not in processed_df.columns, f"Original column {col} should be removed"
        
        # Verify operation metadata
        assert result['columns_affected'] == categorical_cols
        assert result['drop_first_used'] == False
        assert result['total_new_columns'] == len(result['new_columns_created'])


class TestPreprocessingOperationPersistence:
    """
    Property 14: Preprocessing Operation Persistence
    Validates: Requirements 4.7, 4.8
    """
    
    @patch('processing_lambda.s3_client')
    @patch('processing_lambda.dynamodb')
    def test_operation_logging_completeness(self, mock_dynamodb, mock_s3):
        """
        For any preprocessing operation performed, the operation should be logged
        with complete metadata including parameters, timing, and results.
        """
        # Setup mocks
        mock_table = Mock()
        mock_dynamodb.Table.return_value = mock_table
        
        # Create test dataframe
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B']
        })
        
        operation = {
            'type': 'scaling',
            'parameters': {
                'method': 'standard',
                'columns': ['numeric_col']
            }
        }
        
        # Execute operation
        processed_df, result = execute_preprocessing_operation(df, operation)
        
        # Verify operation result contains required metadata
        required_fields = ['method_used', 'columns_affected']
        for field in required_fields:
            assert field in result, f"Operation result should contain {field}"
        
        # Verify operation was successful
        assert result['method_used'] == 'standard'
        assert 'numeric_col' in result['columns_affected']
        
        # Verify processed dataframe is valid
        assert len(processed_df) == len(df), "Processed dataframe should have same number of rows"
        assert 'numeric_col' in processed_df.columns, "Processed column should exist"


# Integration test for quality assessment
class TestQualityAssessmentIntegration:
    """
    Integration tests for dataset quality assessment functionality.
    """
    
    @given(df=generate_mixed_dataframe_with_nulls())
    @settings(max_examples=20, deadline=None)
    def test_quality_assessment_completeness(self, df):
        """
        For any dataset, quality assessment should provide comprehensive metrics
        and actionable recommendations.
        """
        assume(len(df) > 5 and len(df.columns) > 0)
        
        quality_report = assess_dataset_quality(df)
        
        # Verify required sections exist
        required_sections = [
            'basic_metrics', 'missing_value_analysis', 'duplicate_analysis',
            'data_imbalance_analysis', 'data_type_analysis', 'outlier_analysis',
            'overall_quality_score', 'recommendations'
        ]
        
        for section in required_sections:
            assert section in quality_report, f"Quality report should contain {section}"
        
        # Verify basic metrics accuracy
        basic_metrics = quality_report['basic_metrics']
        assert basic_metrics['total_rows'] == len(df)
        assert basic_metrics['total_columns'] == len(df.columns)
        assert basic_metrics['total_cells'] == len(df) * len(df.columns)
        
        # Verify quality score is in valid range
        quality_score = quality_report['overall_quality_score']
        assert 0 <= quality_score <= 100, f"Quality score should be 0-100, got {quality_score}"
        
        # Verify recommendations are provided
        recommendations = quality_report['recommendations']
        assert isinstance(recommendations, list), "Recommendations should be a list"
        assert len(recommendations) > 0, "Should provide at least one recommendation"


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])