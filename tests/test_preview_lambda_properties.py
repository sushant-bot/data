"""
Property-based tests for Preview Lambda function.

These tests validate the correctness properties defined in the design document
for the AI Data Analyst Platform dataset analysis functionality.
"""

import pytest
import json
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Load preview lambda_function via importlib to avoid sys.path collisions
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "preview_lambda",
    os.path.join(os.path.dirname(__file__), '..', 'lambda', 'preview', 'lambda_function.py')
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
sys.modules['preview_lambda'] = _module

lambda_handler = _module.lambda_handler
generate_dataset_preview = _module.generate_dataset_preview
calculate_detailed_statistics = _module.calculate_detailed_statistics
calculate_column_statistics = _module.calculate_column_statistics
calculate_overall_numerical_stats = _module.calculate_overall_numerical_stats
find_highly_correlated_pairs = _module.find_highly_correlated_pairs
create_error_response = _module.create_error_response


# Test data generators
@st.composite
def valid_dataframe(draw):
    """Generate valid DataFrame for testing."""
    num_rows = draw(st.integers(min_value=10, max_value=100))
    num_cols = draw(st.integers(min_value=1, max_value=8))
    
    # Generate column names
    columns = [f"col_{i}" for i in range(num_cols)]
    
    # Generate data with different types
    data = {}
    for col in columns:
        col_type = draw(st.sampled_from(['int', 'float', 'str', 'bool']))
        if col_type == 'int':
            data[col] = draw(st.lists(st.integers(min_value=-1000, max_value=1000), 
                                    min_size=num_rows, max_size=num_rows))
        elif col_type == 'float':
            data[col] = draw(st.lists(st.floats(min_value=-1000.0, max_value=1000.0, 
                                               allow_nan=False, allow_infinity=False), 
                                    min_size=num_rows, max_size=num_rows))
        elif col_type == 'bool':
            data[col] = draw(st.lists(st.booleans(), min_size=num_rows, max_size=num_rows))
        else:
            data[col] = draw(st.lists(st.text(min_size=1, max_size=20), 
                                    min_size=num_rows, max_size=num_rows))
    
    df = pd.DataFrame(data)
    return df


@st.composite
def dataframe_with_missing_values(draw):
    """Generate DataFrame with missing values."""
    df = draw(valid_dataframe())
    
    # Randomly introduce missing values
    for col in df.columns:
        missing_ratio = draw(st.floats(min_value=0.0, max_value=0.5))
        num_missing = int(len(df) * missing_ratio)
        if num_missing > 0:
            missing_indices = draw(st.lists(st.integers(min_value=0, max_value=len(df)-1), 
                                          min_size=num_missing, max_size=num_missing, unique=True))
            for idx in missing_indices:
                df.loc[idx, col] = np.nan
    
    return df


@st.composite
def numerical_dataframe(draw):
    """Generate DataFrame with only numerical columns."""
    num_rows = draw(st.integers(min_value=10, max_value=50))
    num_cols = draw(st.integers(min_value=2, max_value=6))
    
    data = {}
    for i in range(num_cols):
        col_name = f"num_col_{i}"
        col_type = draw(st.sampled_from(['int', 'float']))
        if col_type == 'int':
            data[col_name] = draw(st.lists(st.integers(min_value=-100, max_value=100), 
                                         min_size=num_rows, max_size=num_rows))
        else:
            data[col_name] = draw(st.lists(st.floats(min_value=-100.0, max_value=100.0, 
                                                   allow_nan=False, allow_infinity=False), 
                                         min_size=num_rows, max_size=num_rows))
    
    return pd.DataFrame(data)


def create_mock_event(session_id: str) -> dict:
    """Create a mock API Gateway event for testing."""
    return {
        'pathParameters': {'sessionId': session_id},
        'queryStringParameters': {}
    }


def create_mock_session_data(session_id: str, s3_key: str) -> dict:
    """Create mock session data for DynamoDB."""
    return {
        'session_id': session_id,
        'dataset_name': 'test_data.csv',
        'file_size': 1024,
        'timestamp': '2024-01-15T10:30:00Z',
        's3_key': s3_key
    }


class TestPreviewLambdaProperties:
    """Property-based tests for Preview Lambda functionality."""
    
    def setup_method(self):
        """Set up mocks for each test."""
        self.s3_mock = Mock()
        self.dynamodb_mock = Mock()
        self.sessions_table_mock = Mock()

        # Configure DynamoDB mock
        self.dynamodb_mock.Table.return_value = self.sessions_table_mock

        # Patch AWS clients using module reference
        self.s3_patcher = patch.object(_module, 's3_client', self.s3_mock)
        self.dynamodb_patcher = patch.object(_module, 'dynamodb', self.dynamodb_mock)

        self.s3_patcher.start()
        self.dynamodb_patcher.start()

    def teardown_method(self):
        """Clean up mocks after each test."""
        self.s3_patcher.stop()
        self.dynamodb_patcher.stop()

    @given(df=valid_dataframe())
    @settings(max_examples=30)
    def test_property_4_dataset_preview_accuracy(self, df):
        """
        **Property 4: Dataset Preview Accuracy**
        **Validates: Requirements 2.1, 2.2**

        For any uploaded dataset, the preview should display exactly the first 10 rows
        and accurately report the total row and column counts.
        """
        # Arrange
        session_id = "test-session-123"
        s3_key = f"datasets/{session_id}/original.csv"

        # Reset mocks for this test iteration
        self.s3_mock.reset_mock()
        self.sessions_table_mock.reset_mock()

        # Mock DynamoDB response
        self.sessions_table_mock.get_item.return_value = {
            'Item': create_mock_session_data(session_id, s3_key)
        }

        # Mock S3 response
        csv_content = df.to_csv(index=False).encode()
        self.s3_mock.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=csv_content))
        }

        event = create_mock_event(session_id)

        # Act
        response = lambda_handler(event, {})

        # Assert
        assert response['statusCode'] == 200

        response_body = json.loads(response['body'])
        preview_data = response_body['preview']
        metadata = response_body['metadata']

        # Verify row and column counts
        assert metadata['total_rows'] == len(df)
        assert metadata['total_columns'] == len(df.columns)

        # Verify preview shows exactly first 10 rows (or all rows if less than 10)
        expected_preview_rows = min(10, len(df))
        assert preview_data['total_rows_shown'] == expected_preview_rows
        assert len(preview_data['rows']) == expected_preview_rows

        # Verify column names match
        assert preview_data['columns'] == list(df.columns)
    
    @given(df=dataframe_with_missing_values())
    @settings(max_examples=25, suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow])
    def test_property_5_missing_value_detection(self, df):
        """
        **Property 5: Missing Value Detection**
        **Validates: Requirements 2.3**
        
        For any dataset with missing values, the system should accurately count 
        and report the number of missing values per column.
        """
        # Act
        stats = calculate_detailed_statistics(df)
        
        # Assert
        column_stats = stats['column_statistics']
        
        for col in df.columns:
            expected_missing = int(df[col].isnull().sum())
            actual_missing = column_stats[col]['missing_count']
            
            assert actual_missing == expected_missing, \
                f"Missing value count mismatch for column {col}: expected {expected_missing}, got {actual_missing}"
            
            # Verify missing percentage calculation
            expected_percentage = round((expected_missing / len(df)) * 100, 2)
            actual_percentage = column_stats[col]['missing_percentage']
            
            assert abs(actual_percentage - expected_percentage) < 0.01, \
                f"Missing percentage mismatch for column {col}: expected {expected_percentage}, got {actual_percentage}"
    
    @given(df=valid_dataframe())
    @settings(max_examples=25)
    def test_property_6_data_type_inference(self, df):
        """
        **Property 6: Data Type Inference**
        **Validates: Requirements 2.4**

        For any dataset column, the system should correctly identify and display
        the appropriate data type (numeric, categorical, datetime, etc.).
        """
        # Act
        stats = calculate_detailed_statistics(df)

        # Assert
        column_stats = stats['column_statistics']

        for col in df.columns:
            inferred_type = column_stats[col]['data_type']

            # Verify data type is one of the expected types
            valid_types = ['integer', 'float', 'categorical', 'datetime', 'boolean']
            assert inferred_type in valid_types, \
                f"Invalid data type '{inferred_type}' inferred for column {col}"

            # Verify type inference logic matches what the function does
            col_data = df[col]
            if pd.api.types.is_bool_dtype(col_data):
                assert inferred_type == 'boolean'
            elif pd.api.types.is_integer_dtype(col_data):
                assert inferred_type == 'integer'
            elif pd.api.types.is_float_dtype(col_data):
                assert inferred_type == 'float'
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                assert inferred_type == 'datetime'
            else:
                # Booleans stored as object dtype will be detected as categorical
                assert inferred_type == 'categorical'
    
    @given(df=numerical_dataframe())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
    def test_property_7_statistical_summary_accuracy(self, df):
        """
        **Property 7: Statistical Summary Accuracy**
        **Validates: Requirements 2.5**

        For any numerical column in a dataset, the generated statistical summary
        (mean, median, std, etc.) should match the expected mathematical calculations.
        """
        # Act
        stats = calculate_detailed_statistics(df)

        # Assert
        column_stats = stats['column_statistics']

        for col in df.columns:
            col_stats = column_stats[col]
            col_data = df[col].dropna()  # Remove NaN for calculations

            if col_stats['data_type'] in ['integer', 'float'] and len(col_data) > 0:
                # Verify mean
                expected_mean = float(col_data.mean())
                actual_mean = col_stats['mean']
                assert abs(actual_mean - expected_mean) < 1e-6, \
                    f"Mean mismatch for column {col}: expected {expected_mean}, got {actual_mean}"

                # Verify median
                expected_median = float(col_data.median())
                actual_median = col_stats['median']
                assert abs(actual_median - expected_median) < 1e-6, \
                    f"Median mismatch for column {col}: expected {expected_median}, got {actual_median}"

                # Verify min and max
                expected_min = float(col_data.min())
                expected_max = float(col_data.max())
                actual_min = col_stats['min']
                actual_max = col_stats['max']

                assert abs(actual_min - expected_min) < 1e-6, \
                    f"Min mismatch for column {col}: expected {expected_min}, got {actual_min}"
                assert abs(actual_max - expected_max) < 1e-6, \
                    f"Max mismatch for column {col}: expected {expected_max}, got {actual_max}"

                # Verify standard deviation (if more than one value)
                if len(col_data) > 1:
                    expected_std = float(col_data.std())
                    actual_std = col_stats['std']
                    assert abs(actual_std - expected_std) < 1e-6, \
                        f"Std mismatch for column {col}: expected {expected_std}, got {actual_std}"


class TestDatasetPreviewGeneration:
    """Tests for dataset preview generation functionality."""
    
    @given(df=valid_dataframe())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.data_too_large])
    def test_preview_generation_structure(self, df):
        """Test that preview generation returns correct structure."""
        # Act
        preview = generate_dataset_preview(df)
        
        # Assert
        required_keys = ['columns', 'rows', 'total_rows_shown', 'total_rows_available']
        for key in required_keys:
            assert key in preview, f"Missing key '{key}' in preview data"
        
        assert preview['columns'] == list(df.columns)
        assert preview['total_rows_available'] == len(df)
        assert preview['total_rows_shown'] == min(10, len(df))
        assert len(preview['rows']) == min(10, len(df))
    
    def test_preview_handles_empty_dataframe(self):
        """Test preview generation with empty DataFrame."""
        df = pd.DataFrame()
        preview = generate_dataset_preview(df)
        
        assert preview['columns'] == []
        assert preview['rows'] == []
        assert preview['total_rows_shown'] == 0
        assert preview['total_rows_available'] == 0


class TestCorrelationAnalysis:
    """Tests for correlation analysis functionality."""
    
    @given(df=numerical_dataframe())
    @settings(max_examples=15)
    def test_correlation_matrix_accuracy(self, df):
        """Test that correlation matrix calculation is accurate."""
        assume(len(df.columns) >= 2)  # Need at least 2 columns for correlation
        
        # Act
        numerical_cols = list(df.columns)
        overall_stats = calculate_overall_numerical_stats(df, numerical_cols)
        
        # Assert
        correlation_matrix = overall_stats['correlation_matrix']
        
        # Verify matrix structure
        assert len(correlation_matrix) == len(numerical_cols)
        for col in numerical_cols:
            assert col in correlation_matrix
            assert len(correlation_matrix[col]) == len(numerical_cols)
        
        # Verify diagonal elements are 1.0 (self-correlation)
        for col in numerical_cols:
            if correlation_matrix[col][col] is not None:
                assert abs(correlation_matrix[col][col] - 1.0) < 1e-10
        
        # Verify symmetry
        for col1 in numerical_cols:
            for col2 in numerical_cols:
                val1 = correlation_matrix[col1][col2]
                val2 = correlation_matrix[col2][col1]
                if val1 is not None and val2 is not None:
                    assert abs(val1 - val2) < 1e-10
    
    def test_highly_correlated_pairs_detection(self):
        """Test detection of highly correlated pairs."""
        # Create DataFrame with known correlations
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.1  # Highly correlated with x
        z = np.random.randn(100)  # Independent
        
        df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        correlation_matrix = df.corr()
        
        # Act
        highly_correlated = find_highly_correlated_pairs(correlation_matrix, threshold=0.8)
        
        # Assert
        # Should find x-y pair as highly correlated
        pair_found = False
        for pair in highly_correlated:
            if (pair['column1'] == 'x' and pair['column2'] == 'y') or \
               (pair['column1'] == 'y' and pair['column2'] == 'x'):
                pair_found = True
                assert abs(pair['correlation']) >= 0.8
                break
        
        assert pair_found, "Expected highly correlated pair (x, y) not found"


class TestErrorHandling:
    """Tests for error handling in preview functionality."""
    
    def setup_method(self):
        """Set up mocks for each test."""
        self.s3_mock = Mock()
        self.dynamodb_mock = Mock()
        self.sessions_table_mock = Mock()
        
        self.dynamodb_mock.Table.return_value = self.sessions_table_mock
        
        self.s3_patcher = patch.object(_module, 's3_client', self.s3_mock)
        self.dynamodb_patcher = patch.object(_module, 'dynamodb', self.dynamodb_mock)

        self.s3_patcher.start()
        self.dynamodb_patcher.start()

    def teardown_method(self):
        """Clean up mocks after each test."""
        self.s3_patcher.stop()
        self.dynamodb_patcher.stop()

    def test_missing_session_id_handling(self):
        """Test handling of missing session ID."""
        event = {'pathParameters': {}}

        response = lambda_handler(event, {})
        assert response['statusCode'] == 400

        response_body = json.loads(response['body'])
        assert 'error' in response_body

    def test_session_not_found_handling(self):
        """Test handling of non-existent session."""
        event = create_mock_event("non-existent-session")

        # Mock DynamoDB to return no item
        self.sessions_table_mock.get_item.return_value = {}

        response = lambda_handler(event, {})
        assert response['statusCode'] == 404

        response_body = json.loads(response['body'])
        assert 'error' in response_body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])