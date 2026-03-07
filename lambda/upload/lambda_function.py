import json
import boto3
import pandas as pd
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import io
import base64

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
import os
aws_region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')

s3_client = boto3.client('s3', region_name=aws_region)
dynamodb = boto3.resource('dynamodb', region_name=aws_region)
bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)

# Environment variables
DATA_BUCKET = os.environ.get('DATA_BUCKET', 'ai-data-analyst-platform-data-dev-077437903006')
SESSIONS_TABLE = os.environ.get('SESSIONS_TABLE', 'ai-data-analyst-platform-sessions-dev')

def lambda_handler(event, context):
    """
    Main Lambda handler for dataset upload and initial processing.
    
    Handles:
    - CSV file validation and format checking
    - S3 upload with unique session ID generation
    - Bedrock Guardrails integration for PII detection
    - Basic dataset statistics calculation
    - Session metadata storage in DynamoDB
    """
    try:
        logger.info(f"Upload Lambda invoked with event: {json.dumps(event, default=str)}")
        
        # Parse the request
        body = json.loads(event.get('body', '{}'))
        
        # Extract file data
        file_content = body.get('file_content')
        file_name = body.get('file_name')
        
        if not file_content or not file_name:
            return create_error_response(400, "Missing file_content or file_name")
        
        # Validate file format
        if not file_name.lower().endswith('.csv'):
            return create_error_response(400, "Invalid file format. Only CSV files are supported.")
        
        # Decode base64 file content
        try:
            file_data = base64.b64decode(file_content)
        except Exception as e:
            logger.error(f"Failed to decode file content: {str(e)}")
            return create_error_response(400, "Invalid file encoding")
        
        # Validate CSV format and load dataset
        try:
            df = pd.read_csv(io.BytesIO(file_data))
        except Exception as e:
            logger.error(f"Failed to parse CSV: {str(e)}")
            return create_error_response(400, f"Invalid CSV format: {str(e)}")
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Calculate basic dataset statistics
        stats = calculate_dataset_statistics(df)
        
        # Detect PII using Bedrock Guardrails
        pii_results = detect_pii_with_guardrails(df)
        
        # Store dataset in S3
        s3_key = f"datasets/{session_id}/original.csv"
        try:
            s3_client.put_object(
                Bucket=DATA_BUCKET,
                Key=s3_key,
                Body=file_data,
                ContentType='text/csv'
            )
            logger.info(f"Dataset uploaded to S3: {s3_key}")
        except Exception as e:
            logger.error(f"Failed to upload to S3: {str(e)}")
            return create_error_response(500, "Failed to store dataset")
        
        # Store session metadata in DynamoDB
        session_metadata = {
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat(),
            'dataset_name': file_name,
            'file_size': len(file_data),
            'row_count': stats['row_count'],
            'column_count': stats['column_count'],
            'status': 'uploaded',
            'pii_detected': pii_results['pii_detected'],
            'pii_details': pii_results['pii_details'],
            's3_key': s3_key,
            'data_types': stats['data_types'],
            'missing_values': stats['missing_values']
        }
        
        try:
            sessions_table = dynamodb.Table(SESSIONS_TABLE)
            sessions_table.put_item(Item=session_metadata)
            logger.info(f"Session metadata stored: {session_id}")
        except Exception as e:
            logger.error(f"Failed to store session metadata: {str(e)}")
            return create_error_response(500, "Failed to store session metadata")
        
        # Return success response
        response_data = {
            'session_id': session_id,
            'dataset_name': file_name,
            'statistics': stats,
            'pii_detection': pii_results,
            'message': 'Dataset uploaded successfully'
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            },
            'body': json.dumps(response_data)
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in upload handler: {str(e)}")
        return create_error_response(500, "Internal server error")


def calculate_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate basic statistics for the uploaded dataset.
    
    Args:
        df: Pandas DataFrame containing the dataset
        
    Returns:
        Dictionary containing dataset statistics
    """
    try:
        stats = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'data_types': {},
            'missing_values': {},
            'numerical_summary': {}
        }
        
        # Calculate data types and missing values for each column
        for column in df.columns:
            # Infer data type
            if pd.api.types.is_numeric_dtype(df[column]):
                if pd.api.types.is_integer_dtype(df[column]):
                    data_type = 'integer'
                else:
                    data_type = 'float'
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                data_type = 'datetime'
            elif pd.api.types.is_bool_dtype(df[column]):
                data_type = 'boolean'
            else:
                data_type = 'categorical'
            
            stats['data_types'][column] = data_type
            
            # Count missing values
            missing_count = df[column].isnull().sum()
            stats['missing_values'][column] = int(missing_count)
            
            # Calculate numerical summaries for numeric columns
            if data_type in ['integer', 'float']:
                try:
                    numeric_stats = {
                        'mean': float(df[column].mean()) if not df[column].isnull().all() else None,
                        'median': float(df[column].median()) if not df[column].isnull().all() else None,
                        'std': float(df[column].std()) if not df[column].isnull().all() else None,
                        'min': float(df[column].min()) if not df[column].isnull().all() else None,
                        'max': float(df[column].max()) if not df[column].isnull().all() else None
                    }
                    stats['numerical_summary'][column] = numeric_stats
                except Exception as e:
                    logger.warning(f"Failed to calculate numerical summary for {column}: {str(e)}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to calculate dataset statistics: {str(e)}")
        return {
            'row_count': 0,
            'column_count': 0,
            'data_types': {},
            'missing_values': {},
            'numerical_summary': {}
        }


def detect_pii_with_guardrails(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect PII in the dataset using Amazon Bedrock Guardrails.
    
    Args:
        df: Pandas DataFrame containing the dataset
        
    Returns:
        Dictionary containing PII detection results
    """
    try:
        pii_results = {
            'pii_detected': False,
            'pii_details': {},
            'warning_message': None
        }
        
        # Sample first few rows for PII detection to avoid large payloads
        sample_size = min(10, len(df))
        sample_df = df.head(sample_size)
        
        # Convert sample to string for analysis
        sample_text = sample_df.to_string()
        
        # Use Bedrock Guardrails for PII detection
        # Note: This is a simplified implementation - in production, you'd use the actual Guardrails API
        pii_patterns = detect_pii_patterns(sample_text)
        
        if pii_patterns:
            pii_results['pii_detected'] = True
            pii_results['pii_details'] = pii_patterns
            pii_results['warning_message'] = "Personally Identifiable Information (PII) detected in dataset. Please review data privacy compliance."
        
        return pii_results
        
    except Exception as e:
        logger.error(f"Failed to detect PII: {str(e)}")
        return {
            'pii_detected': False,
            'pii_details': {},
            'warning_message': None,
            'error': f"PII detection failed: {str(e)}"
        }


def detect_pii_patterns(text: str) -> Dict[str, int]:
    """
    Detect common PII patterns in text using regex.
    This is a simplified implementation - production would use Bedrock Guardrails.
    
    Args:
        text: Text to analyze for PII patterns
        
    Returns:
        Dictionary with PII pattern counts
    """
    import re
    
    patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    }
    
    detected = {}
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            detected[pattern_name] = len(matches)
    
    return detected


def create_error_response(status_code: int, message: str) -> Dict[str, Any]:
    """
    Create standardized error response.
    
    Args:
        status_code: HTTP status code
        message: Error message
        
    Returns:
        API Gateway response dictionary
    """
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'POST, OPTIONS'
        },
        'body': json.dumps({
            'error': message,
            'timestamp': datetime.utcnow().isoformat()
        })
    }