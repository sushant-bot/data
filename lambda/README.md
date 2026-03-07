# AI Data Analyst Platform - Lambda Functions

This directory contains the core data processing Lambda functions for the AI Data Analyst Platform.

## Functions

### Upload Lambda (`upload/`)

**Purpose**: Handles dataset upload and initial processing

**Features**:
- CSV file validation and format checking
- S3 upload with unique session ID generation
- Bedrock Guardrails integration for PII detection
- Basic dataset statistics calculation (rows, columns, data types, missing values)
- Session metadata storage in DynamoDB

**API Endpoint**: `POST /upload`

**Request Format**:
```json
{
  "file_content": "base64-encoded-csv-content",
  "file_name": "dataset.csv"
}
```

**Response Format**:
```json
{
  "session_id": "uuid",
  "dataset_name": "dataset.csv",
  "statistics": {
    "row_count": 1000,
    "column_count": 5,
    "data_types": {...},
    "missing_values": {...}
  },
  "pii_detection": {
    "pii_detected": false,
    "pii_details": {},
    "warning_message": null
  },
  "message": "Dataset uploaded successfully"
}
```

### Preview Lambda (`preview/`)

**Purpose**: Generates dataset preview and detailed statistics

**Features**:
- Dataset preview generation (first 10 rows)
- Missing value detection and counting per column
- Statistical summaries for numerical columns
- Correlation analysis for numerical data
- Data type inference and validation

**API Endpoint**: `GET /preview/{sessionId}`

**Response Format**:
```json
{
  "session_id": "uuid",
  "dataset_name": "dataset.csv",
  "preview": {
    "columns": ["col1", "col2", "col3"],
    "rows": [[val1, val2, val3], ...],
    "total_rows_shown": 10,
    "total_rows_available": 1000
  },
  "statistics": {
    "column_statistics": {...},
    "missing_value_summary": {...},
    "data_type_summary": {...},
    "numerical_columns_summary": {...}
  },
  "metadata": {
    "total_rows": 1000,
    "total_columns": 5,
    "file_size": 1048576,
    "upload_timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Dependencies

Both functions require:
- `pandas==2.1.4` - Data manipulation and analysis
- `boto3==1.34.0` - AWS SDK for Python
- `numpy==1.24.3` - Numerical computing

## Environment Variables

- `DATA_BUCKET` - S3 bucket name for dataset storage
- `SESSIONS_TABLE` - DynamoDB table name for session metadata
- `AWS_DEFAULT_REGION` - AWS region (default: us-east-1)

## IAM Permissions

### Upload Lambda
- `s3:GetObject`, `s3:PutObject` - S3 data bucket access
- `dynamodb:PutItem` - Session metadata storage
- `bedrock:InvokeModel`, `bedrock:ApplyGuardrail` - PII detection
- CloudWatch Logs permissions

### Preview Lambda
- `s3:GetObject` - S3 data bucket read access
- `dynamodb:GetItem` - Session metadata retrieval
- CloudWatch Logs permissions

## Testing

Property-based tests are available in the `tests/` directory:

- `test_upload_lambda_properties.py` - Upload Lambda property tests
- `test_preview_lambda_properties.py` - Preview Lambda property tests

Run tests with:
```bash
pytest tests/ -v
```

## Deployment

Use the `deployment.yaml` configuration file for CloudFormation deployment or reference the individual function configurations for manual deployment.

## Error Handling

Both functions implement comprehensive error handling:

- Input validation errors (400 status codes)
- Resource not found errors (404 status codes)
- Internal processing errors (500 status codes)
- Structured error responses with timestamps
- CloudWatch logging for debugging

## Performance Considerations

- Upload Lambda: Optimized for files up to 10MB
- Preview Lambda: Efficient preview generation for large datasets
- Memory allocation: 1024MB for both functions
- Timeout: 5 minutes for processing operations
- Correlation analysis limited to numerical columns for performance