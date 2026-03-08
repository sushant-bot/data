import json
import boto3
import pandas as pd
import logging
from typing import Dict, Any, List
import io

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
import os
aws_region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')

s3_client = boto3.client('s3', region_name=aws_region)
dynamodb = boto3.resource('dynamodb', region_name=aws_region)

# Environment variables
DATA_BUCKET = os.environ.get('DATA_BUCKET', 'ai-data-analyst-platform-data-dev-672627895253')
SESSIONS_TABLE = os.environ.get('SESSIONS_TABLE', 'ai-data-analyst-platform-sessions-dev')

def lambda_handler(event, context):
    """
    Main Lambda handler for dataset preview and statistics generation.

    Handles:
    - Paginated dataset preview (page / page_size query params)
    - Original or processed dataset selection (dataset_type query param)
    - Presigned download URL generation (download query param: 'original' | 'processed')
    - Missing value detection and counting per column
    - Statistical summaries for numerical columns
    """
    try:
        logger.info(f"Preview Lambda invoked with event: {json.dumps(event, default=str)}")

        # Extract session ID from path parameters
        session_id = event.get('pathParameters', {}).get('sessionId')

        if not session_id:
            return create_error_response(400, "Missing session ID")

        # Parse query string parameters
        query_params = event.get('queryStringParameters') or {}
        page = int(query_params.get('page', 1))
        page_size = min(int(query_params.get('page_size', 15)), 100)  # cap at 100
        dataset_type = query_params.get('dataset_type', 'original')   # 'original' | 'processed'
        download_target = query_params.get('download')                 # 'original' | 'processed' | None
        download_format = query_params.get('format', 'csv')           # 'csv' | 'json'

        # Get session metadata from DynamoDB
        try:
            sessions_table = dynamodb.Table(SESSIONS_TABLE)
            response = sessions_table.get_item(Key={'session_id': session_id})

            if 'Item' not in response:
                return create_error_response(404, "Session not found")

            session_data = response['Item']

        except Exception as e:
            logger.error(f"Failed to retrieve session data: {str(e)}")
            return create_error_response(500, "Failed to retrieve session data")

        original_s3_key = session_data.get('s3_key')
        processed_s3_key = session_data.get('processed_s3_key')
        has_processed = processed_s3_key is not None

        # ── Download request ──────────────────────────────────────────────────
        if download_target in ('original', 'processed'):
            key = original_s3_key if download_target == 'original' else processed_s3_key
            if not key:
                return create_error_response(404, f"No {download_target} dataset found")

            if download_format == 'json':
                # Stream CSV, convert to JSON records, return inline
                try:
                    obj = s3_client.get_object(Bucket=DATA_BUCKET, Key=key)
                    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
                    json_body = df.to_json(orient='records', default_handler=str)
                    return {
                        'statusCode': 200,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Content-Disposition': f'attachment; filename="{download_target}_data.json"',
                            'Access-Control-Allow-Origin': '*',
                        },
                        'body': json_body,
                    }
                except Exception as e:
                    logger.error(f"JSON conversion failed: {str(e)}")
                    return create_error_response(500, "Failed to convert dataset to JSON")
            else:
                # Generate presigned URL for CSV download
                try:
                    presigned_url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={
                            'Bucket': DATA_BUCKET,
                            'Key': key,
                            'ResponseContentDisposition': f'attachment; filename="{download_target}_data.csv"',
                            'ResponseContentType': 'text/csv',
                        },
                        ExpiresIn=3600,
                    )
                    return {
                        'statusCode': 200,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*',
                        },
                        'body': json.dumps({'download_url': presigned_url}),
                    }
                except Exception as e:
                    logger.error(f"Failed to generate presigned URL: {str(e)}")
                    return create_error_response(500, "Failed to generate download URL")

        # ── Normal preview request ────────────────────────────────────────────
        # Determine which file to load
        if dataset_type == 'processed' and processed_s3_key:
            s3_key = processed_s3_key
        else:
            s3_key = original_s3_key

        if not s3_key:
            return create_error_response(404, "Dataset not found for this session")

        # Load dataset from S3
        try:
            obj = s3_client.get_object(Bucket=DATA_BUCKET, Key=s3_key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        except Exception as e:
            logger.error(f"Failed to load dataset from S3: {str(e)}")
            return create_error_response(500, "Failed to load dataset")

        # Generate paginated preview
        preview_data = generate_dataset_preview(df, page=page, page_size=page_size)

        # Statistics are computed on the full df (not paginated)
        detailed_stats = calculate_detailed_statistics(df)

        # Generate presigned download URLs to include in every preview response
        download_urls = {}
        try:
            download_urls['original_csv'] = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': DATA_BUCKET,
                    'Key': original_s3_key,
                    'ResponseContentDisposition': 'attachment; filename="original_data.csv"',
                    'ResponseContentType': 'text/csv',
                },
                ExpiresIn=3600,
            )
            if processed_s3_key:
                download_urls['processed_csv'] = s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': DATA_BUCKET,
                        'Key': processed_s3_key,
                        'ResponseContentDisposition': 'attachment; filename="processed_data.csv"',
                        'ResponseContentType': 'text/csv',
                    },
                    ExpiresIn=3600,
                )
        except Exception as e:
            logger.warning(f"Could not generate presigned download URLs: {str(e)}")

        response_data = {
            'session_id': session_id,
            'dataset_name': session_data.get('dataset_name'),
            'preview': preview_data,
            'statistics': detailed_stats,
            'metadata': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'file_size': session_data.get('file_size'),
                'upload_timestamp': session_data.get('timestamp'),
                'has_processed_data': has_processed,
            },
            'download_urls': download_urls,
        }

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
            },
            'body': json.dumps(response_data, default=str),
        }

    except Exception as e:
        logger.error(f"Unexpected error in preview handler: {str(e)}")
        return create_error_response(500, "Internal server error")


def generate_dataset_preview(df: pd.DataFrame, page: int = 1, page_size: int = 15) -> Dict[str, Any]:
    """
    Generate a paginated dataset preview.

    Args:
        df: Pandas DataFrame containing the dataset
        page: 1-based page number
        page_size: Number of rows per page

    Returns:
        Dictionary containing preview data with pagination metadata
    """
    try:
        total_rows = len(df)
        total_pages = max(1, -(-total_rows // page_size))  # ceiling division
        page = max(1, min(page, total_pages))              # clamp to valid range
        offset = (page - 1) * page_size

        preview_df = df.iloc[offset: offset + page_size]

        preview_data = {
            'columns': list(df.columns),
            'rows': [],
            'total_rows_shown': len(preview_df),
            'total_rows_available': total_rows,
            'current_page': page,
            'total_pages': total_pages,
            'page_size': page_size,
        }

        for _, row in preview_df.iterrows():
            row_data = []
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    row_data.append(None)
                else:
                    row_data.append(value.item() if hasattr(value, 'item') else value)
            preview_data['rows'].append(row_data)

        return preview_data

    except Exception as e:
        logger.error(f"Failed to generate dataset preview: {str(e)}")
        return {
            'columns': [],
            'rows': [],
            'total_rows_shown': 0,
            'total_rows_available': 0,
            'current_page': 1,
            'total_pages': 1,
            'page_size': page_size,
            'error': f"Preview generation failed: {str(e)}",
        }


def calculate_detailed_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate detailed statistics including missing values and numerical summaries.
    
    Args:
        df: Pandas DataFrame containing the dataset
        
    Returns:
        Dictionary containing detailed statistics
    """
    try:
        stats = {
            'column_statistics': {},
            'missing_value_summary': {
                'total_missing_values': 0,
                'columns_with_missing': 0,
                'missing_percentage': 0.0
            },
            'data_type_summary': {},
            'numerical_columns_summary': {}
        }
        
        total_missing = 0
        columns_with_missing = 0
        
        # Calculate statistics for each column
        for column in df.columns:
            col_stats = calculate_column_statistics(df, column)
            stats['column_statistics'][column] = col_stats
            
            # Track missing values
            missing_count = col_stats['missing_count']
            if missing_count > 0:
                columns_with_missing += 1
                total_missing += missing_count
        
        # Calculate missing value summary
        total_cells = len(df) * len(df.columns)
        stats['missing_value_summary'] = {
            'total_missing_values': total_missing,
            'columns_with_missing': columns_with_missing,
            'missing_percentage': round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0
        }
        
        # Calculate data type summary
        data_types = {}
        for column in df.columns:
            data_type = stats['column_statistics'][column]['data_type']
            data_types[data_type] = data_types.get(data_type, 0) + 1
        stats['data_type_summary'] = data_types
        
        # Calculate numerical columns summary
        numerical_columns = [col for col in df.columns 
                           if stats['column_statistics'][col]['data_type'] in ['integer', 'float']]
        
        if numerical_columns:
            stats['numerical_columns_summary'] = {
                'count': len(numerical_columns),
                'columns': numerical_columns,
                'overall_statistics': calculate_overall_numerical_stats(df, numerical_columns)
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to calculate detailed statistics: {str(e)}")
        return {
            'column_statistics': {},
            'missing_value_summary': {'total_missing_values': 0, 'columns_with_missing': 0, 'missing_percentage': 0.0},
            'data_type_summary': {},
            'numerical_columns_summary': {},
            'error': f"Statistics calculation failed: {str(e)}"
        }


def calculate_column_statistics(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Calculate statistics for a single column.
    
    Args:
        df: Pandas DataFrame
        column: Column name
        
    Returns:
        Dictionary containing column statistics
    """
    try:
        col_data = df[column]
        
        # Determine data type (check boolean before numeric since bools are numeric in pandas)
        if pd.api.types.is_bool_dtype(col_data):
            data_type = 'boolean'
        elif pd.api.types.is_numeric_dtype(col_data):
            if pd.api.types.is_integer_dtype(col_data):
                data_type = 'integer'
            else:
                data_type = 'float'
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            data_type = 'datetime'
        else:
            data_type = 'categorical'
        
        # Basic statistics
        stats = {
            'data_type': data_type,
            'missing_count': int(col_data.isnull().sum()),
            'missing_percentage': round((col_data.isnull().sum() / len(col_data)) * 100, 2),
            'unique_count': int(col_data.nunique()),
            'total_count': len(col_data)
        }
        
        # Type-specific statistics
        if data_type in ['integer', 'float']:
            # Numerical statistics
            non_null_data = col_data.dropna()
            if len(non_null_data) > 0:
                stats.update({
                    'mean': float(non_null_data.mean()),
                    'median': float(non_null_data.median()),
                    'std': float(non_null_data.std()) if len(non_null_data) > 1 else 0.0,
                    'min': float(non_null_data.min()),
                    'max': float(non_null_data.max()),
                    'q25': float(non_null_data.quantile(0.25)),
                    'q75': float(non_null_data.quantile(0.75))
                })
        elif data_type == 'boolean':
            # Boolean statistics
            non_null_data = col_data.dropna()
            if len(non_null_data) > 0:
                true_count = int(non_null_data.sum())
                false_count = len(non_null_data) - true_count
                stats.update({
                    'true_count': true_count,
                    'false_count': false_count,
                    'true_percentage': round((true_count / len(non_null_data)) * 100, 2)
                })
        elif data_type == 'categorical':
            # Categorical statistics
            value_counts = col_data.value_counts().head(10)  # Top 10 values
            stats.update({
                'most_frequent_values': {
                    str(k): int(v) for k, v in value_counts.items()
                },
                'cardinality': int(col_data.nunique())
            })
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to calculate statistics for column {column}: {str(e)}")
        return {
            'data_type': 'unknown',
            'missing_count': 0,
            'missing_percentage': 0.0,
            'unique_count': 0,
            'total_count': 0,
            'error': f"Column statistics failed: {str(e)}"
        }


def calculate_overall_numerical_stats(df: pd.DataFrame, numerical_columns: List[str]) -> Dict[str, Any]:
    """
    Calculate overall statistics for all numerical columns.
    
    Args:
        df: Pandas DataFrame
        numerical_columns: List of numerical column names
        
    Returns:
        Dictionary containing overall numerical statistics
    """
    try:
        numerical_df = df[numerical_columns]
        
        # Calculate correlation matrix
        correlation_matrix = numerical_df.corr()
        
        # Convert correlation matrix to serializable format
        correlation_dict = {}
        for col1 in correlation_matrix.columns:
            correlation_dict[col1] = {}
            for col2 in correlation_matrix.columns:
                corr_value = correlation_matrix.loc[col1, col2]
                if pd.isna(corr_value):
                    correlation_dict[col1][col2] = None
                else:
                    correlation_dict[col1][col2] = float(corr_value)
        
        return {
            'correlation_matrix': correlation_dict,
            'highly_correlated_pairs': find_highly_correlated_pairs(correlation_matrix),
            'summary': {
                'total_numerical_columns': len(numerical_columns),
                'columns': numerical_columns
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate overall numerical statistics: {str(e)}")
        return {
            'correlation_matrix': {},
            'highly_correlated_pairs': [],
            'summary': {'total_numerical_columns': 0, 'columns': []},
            'error': f"Overall numerical statistics failed: {str(e)}"
        }


def find_highly_correlated_pairs(correlation_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Dict[str, Any]]:
    """
    Find pairs of columns with high correlation.
    
    Args:
        correlation_matrix: Pandas DataFrame containing correlation matrix
        threshold: Correlation threshold for identifying high correlation
        
    Returns:
        List of highly correlated column pairs
    """
    try:
        highly_correlated = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if not pd.isna(corr_value) and abs(corr_value) >= threshold:
                    highly_correlated.append({
                        'column1': col1,
                        'column2': col2,
                        'correlation': float(corr_value),
                        'strength': 'strong' if abs(corr_value) >= 0.9 else 'moderate'
                    })
        
        return highly_correlated
        
    except Exception as e:
        logger.error(f"Failed to find highly correlated pairs: {str(e)}")
        return []


def create_error_response(status_code: int, message: str) -> Dict[str, Any]:
    """
    Create standardized error response.
    
    Args:
        status_code: HTTP status code
        message: Error message
        
    Returns:
        API Gateway response dictionary
    """
    from datetime import datetime
    
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET, OPTIONS'
        },
        'body': json.dumps({
            'error': message,
            'timestamp': datetime.utcnow().isoformat()
        })
    }