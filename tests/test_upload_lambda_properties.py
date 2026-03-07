"""
Property-based tests for Upload Lambda function.

These tests validate the correctness properties defined in the design document
for the AI Data Analyst Platform upload functionality.
"""

import pytest
import json
import base64
import io
import pandas as pd
import uuid
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Load upload lambda_function via importlib to avoid sys.path collisions
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "upload_lambda",
    os.path.join(os.path.dirname(__file__), '..', 'lambda', 'upload', 'lambda_function.py')
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
sys.modules['upload_lambda'] = _module

lambda_handler = _module.lambda_handler
calculate_dataset_statistics = _module.calculate_dataset_statistics
detect_pii_with_guardrails = _module.detect_pii_with_guardrails
detect_pii_patterns = _module.detect_pii_patterns
create_error_response = _module.create_error_response


# Test data generators
@st.composite
def valid_csv_data(draw):
    """Generate valid CSV data for testing."""
    num_rows = draw(st.integers(min_value=1, max_value=100))
    num_cols = draw(st.integers(min_value=1, max_value=10))
    
    # Generate column names
    columns = [f"col_{i}" for i in range(num_cols)]
    
    # Generate data
    data = {}
    for col in columns:
        col_type = draw(st.sampled_from(['int', 'float', 'str']))
        if col_type == 'int':
            data[col] = draw(st.lists(st.integers(min_value=-1000, max_value=1000), 
                                    min_size=num_rows, max_size=num_rows))
        elif col_type == 'float':
            data[col] = draw(st.lists(st.floats(min_value=-1000.0, max_value=1000.0, 
                                               allow_nan=False, allow_infinity=False), 
                                    min_size=num_rows, max_size=num_rows))
        else:
            data[col] = draw(st.lists(st.text(min_size=1, max_size=20), 
                                    min_size=num_rows, max_size=num_rows))
    
    df = pd.DataFrame(data)
    return df


@st.composite
def csv_with_missing_values(draw):
    """Generate CSV data with missing values."""
    df = draw(valid_csv_data())
    
    # Randomly introduce missing values
    for col in df.columns:
        missing_indices = draw(st.lists(st.integers(min_value=0, max_value=len(df)-1), 
                                      max_size=len(df)//2, unique=True))
        for idx in missing_indices:
            df.loc[idx, col] = None
    
    return df


@st.composite
def pii_containing_data(draw):
    """Generate CSV data containing PII patterns."""
    base_df = draw(valid_csv_data())
    
    # Add PII columns
    num_rows = len(base_df)
    
    # Add email column
    emails = [f"user{i}@example.com" for i in range(num_rows)]
    base_df['email'] = emails
    
    # Add phone column
    phones = [f"555-{i:03d}-{(i*7)%10000:04d}" for i in range(num_rows)]
    base_df['phone'] = phones
    
    return base_df


def create_mock_event(file_content: str, file_name: str) -> dict:
    """Create a mock API Gateway event for testing."""
    encoded_content = base64.b64encode(file_content.encode()).decode()
    
    return {
        'body': json.dumps({
            'file_content': encoded_content,
            'file_name': file_name
        }),
        'pathParameters': {},
        'queryStringParameters': {}
    }


def df_to_csv_string(df: pd.DataFrame) -> str:
    """Convert DataFrame to CSV string."""
    return df.to_csv(index=False)


class TestUploadLambdaProperties:
    """Property-based tests for Upload Lambda functionality."""
    
    def setup_method(self):
        """Set up mocks for each test."""
        self.s3_mock = Mock()
        self.dynamodb_mock = Mock()
        self.sessions_table_mock = Mock()
        self.sessions_table_mock.put_item.return_value = {}

        # Configure DynamoDB mock
        self.dynamodb_mock.Table.return_value = self.sessions_table_mock

        # Patch AWS clients using the module reference
        self.s3_patcher = patch.object(_module, 's3_client', self.s3_mock)
        self.dynamodb_patcher = patch.object(_module, 'dynamodb', self.dynamodb_mock)

        self.s3_patcher.start()
        self.dynamodb_patcher.start()

    def teardown_method(self):
        """Clean up mocks after each test."""
        self.s3_patcher.stop()
        self.dynamodb_patcher.stop()
    
    @given(csv_data=valid_csv_data())
    @settings(max_examples=10)  # Reduced for faster testing
    def test_property_1_csv_file_upload_validation(self, csv_data):
        """
        **Property 1: CSV File Upload Validation**
        **Validates: Requirements 1.1, 1.4**
        
        For any file uploaded to the system, if the file has a .csv extension 
        and valid CSV format, it should be successfully stored in S3 with a 
        unique session identifier generated.
        """
        # Reset mocks for this test iteration
        self.s3_mock.reset_mock()
        self.sessions_table_mock.reset_mock()
        
        # Arrange
        csv_string = df_to_csv_string(csv_data)
        event = create_mock_event(csv_string, "test_data.csv")
        
        # Act
        response = lambda_handler(event, {})
        
        # Assert
        assert response['statusCode'] == 200
        
        response_body = json.loads(response['body'])
        assert 'session_id' in response_body
        assert response_body['session_id'] is not None
        
        # Verify session ID is a valid UUID
        try:
            uuid.UUID(response_body['session_id'])
        except ValueError:
            pytest.fail("Generated session_id is not a valid UUID")
        
        # Verify S3 upload was called
        self.s3_mock.put_object.assert_called_once()
        call_args = self.s3_mock.put_object.call_args
        assert call_args[1]['ContentType'] == 'text/csv'
        assert call_args[1]['Key'].startswith('datasets/')
        assert call_args[1]['Key'].endswith('/original.csv')
        
        # Verify DynamoDB storage was called
        self.sessions_table_mock.put_item.assert_called_once()
        stored_item = self.sessions_table_mock.put_item.call_args[1]['Item']
        assert stored_item['session_id'] == response_body['session_id']
        assert stored_item['status'] == 'uploaded'
    
    @given(file_extension=st.sampled_from(['.txt', '.json', '.xlsx', '.pdf', '.doc']))
    def test_property_2_file_format_validation(self, file_extension):
        """
        **Property 2: File Format Validation**
        **Validates: Requirements 1.2, 1.3**
        
        For any non-CSV file uploaded to the system, the upload should be 
        rejected with a descriptive error message indicating the invalid format.
        """
        # Arrange
        file_name = f"test_file{file_extension}"
        event = create_mock_event("some content", file_name)
        
        # Act
        response = lambda_handler(event, {})
        
        # Assert
        assert response['statusCode'] == 400
        
        response_body = json.loads(response['body'])
        assert 'error' in response_body
        assert 'CSV' in response_body['error'] or 'csv' in response_body['error']
        
        # Verify no S3 upload occurred
        self.s3_mock.put_object.assert_not_called()
        
        # Verify no DynamoDB storage occurred
        self.sessions_table_mock.put_item.assert_not_called()
    
    @given(csv_data=valid_csv_data())
    @settings(max_examples=10)  # Reduced for faster testing
    def test_property_3_session_metadata_persistence(self, csv_data):
        """
        **Property 3: Session Metadata Persistence**
        **Validates: Requirements 1.5**
        
        For any successful dataset upload, the corresponding session metadata 
        should be stored in DynamoDB with all required fields populated.
        """
        # Reset mocks for this test iteration
        self.s3_mock.reset_mock()
        self.sessions_table_mock.reset_mock()
        
        # Arrange
        csv_string = df_to_csv_string(csv_data)
        event = create_mock_event(csv_string, "test_data.csv")
        
        # Act
        response = lambda_handler(event, {})
        
        # Assert
        assert response['statusCode'] == 200
        
        # Verify DynamoDB storage
        self.sessions_table_mock.put_item.assert_called_once()
        stored_item = self.sessions_table_mock.put_item.call_args[1]['Item']
        
        # Check required fields
        required_fields = [
            'session_id', 'timestamp', 'dataset_name', 'file_size',
            'row_count', 'column_count', 'status', 'pii_detected',
            's3_key', 'data_types', 'missing_values'
        ]
        
        for field in required_fields:
            assert field in stored_item, f"Required field '{field}' missing from stored metadata"
        
        # Verify field values
        assert stored_item['dataset_name'] == "test_data.csv"
        assert stored_item['row_count'] == len(csv_data)
        assert stored_item['column_count'] == len(csv_data.columns)
        assert stored_item['status'] == 'uploaded'
        assert isinstance(stored_item['pii_detected'], bool)
        assert isinstance(stored_item['data_types'], dict)
        assert isinstance(stored_item['missing_values'], dict)


class TestDatasetStatisticsProperties:
    """Property-based tests for dataset statistics calculation."""
    
    @given(csv_data=valid_csv_data())
    @settings(max_examples=30)
    def test_dataset_statistics_accuracy(self, csv_data):
        """
        Test that calculated dataset statistics match expected values.
        """
        # Act
        stats = calculate_dataset_statistics(csv_data)
        
        # Assert basic counts
        assert stats['row_count'] == len(csv_data)
        assert stats['column_count'] == len(csv_data.columns)
        
        # Verify data types are calculated for all columns
        assert len(stats['data_types']) == len(csv_data.columns)
        for col in csv_data.columns:
            assert col in stats['data_types']
            assert stats['data_types'][col] in ['integer', 'float', 'categorical', 'datetime', 'boolean']
        
        # Verify missing values are calculated for all columns
        assert len(stats['missing_values']) == len(csv_data.columns)
        for col in csv_data.columns:
            assert col in stats['missing_values']
            expected_missing = csv_data[col].isnull().sum()
            assert stats['missing_values'][col] == expected_missing
    
    @given(csv_data=csv_with_missing_values())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.data_too_large])
    def test_missing_value_detection_accuracy(self, csv_data):
        """
        Test that missing value detection is accurate.
        """
        # Act
        stats = calculate_dataset_statistics(csv_data)
        
        # Assert
        for col in csv_data.columns:
            expected_missing = int(csv_data[col].isnull().sum())
            actual_missing = stats['missing_values'][col]
            assert actual_missing == expected_missing, \
                f"Missing value count mismatch for column {col}: expected {expected_missing}, got {actual_missing}"


class TestPIIDetectionProperties:
    """Property-based tests for PII detection functionality."""
    
    @given(csv_data=pii_containing_data())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.data_too_large])
    def test_pii_detection_accuracy(self, csv_data):
        """
        Test that PII detection correctly identifies PII patterns.
        """
        # Act
        pii_results = detect_pii_with_guardrails(csv_data)
        
        # Assert
        assert isinstance(pii_results, dict)
        assert 'pii_detected' in pii_results
        assert 'pii_details' in pii_results
        
        # Since we added email and phone columns, PII should be detected
        assert pii_results['pii_detected'] is True
        assert len(pii_results['pii_details']) > 0
    
    def test_pii_pattern_detection(self):
        """
        Test PII pattern detection with known patterns.
        """
        # Test email detection
        text_with_email = "Contact us at john.doe@example.com for more info"
        patterns = detect_pii_patterns(text_with_email)
        assert 'email' in patterns
        assert patterns['email'] == 1
        
        # Test phone detection
        text_with_phone = "Call us at 555-123-4567 or 555.987.6543"
        patterns = detect_pii_patterns(text_with_phone)
        assert 'phone' in patterns
        assert patterns['phone'] == 2
        
        # Test SSN detection
        text_with_ssn = "SSN: 123-45-6789"
        patterns = detect_pii_patterns(text_with_ssn)
        assert 'ssn' in patterns
        assert patterns['ssn'] == 1


class TestErrorHandlingProperties:
    """Property-based tests for error handling."""
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON in request body."""
        event = {
            'body': 'invalid json',
            'pathParameters': {},
            'queryStringParameters': {}
        }
        
        response = lambda_handler(event, {})
        assert response['statusCode'] in [400, 500]
    
    def test_missing_file_content_handling(self):
        """Test handling of missing file content."""
        event = {
            'body': json.dumps({'file_name': 'test.csv'}),
            'pathParameters': {},
            'queryStringParameters': {}
        }
        
        response = lambda_handler(event, {})
        assert response['statusCode'] == 400
        
        response_body = json.loads(response['body'])
        assert 'error' in response_body
    
    def test_invalid_csv_handling(self):
        """Test handling of completely invalid CSV content."""
        # Use binary data that cannot be parsed as CSV at all
        invalid_content = "\x00\x01\x02\x03\x04\x05"
        event = create_mock_event(invalid_content, "test.csv")

        response = lambda_handler(event, {})
        # Should either reject as bad CSV (400) or handle gracefully (500)
        assert response['statusCode'] in [400, 500]

        response_body = json.loads(response['body'])
        assert 'error' in response_body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])