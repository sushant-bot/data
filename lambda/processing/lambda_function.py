import json
import boto3
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import uuid

# Import quality assessment module
from quality_assessment import assess_dataset_quality

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
import os
aws_region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')

s3_client = boto3.client('s3', region_name=aws_region)
dynamodb = boto3.resource('dynamodb', region_name=aws_region)

# Environment variables
DATA_BUCKET = os.environ.get('DATA_BUCKET', 'ai-data-analyst-platform-data-dev-077437903006')
SESSIONS_TABLE = os.environ.get('SESSIONS_TABLE', 'ai-data-analyst-platform-sessions-dev')
OPERATIONS_TABLE = os.environ.get('OPERATIONS_TABLE', 'ai-data-analyst-platform-operations-dev')

def lambda_handler(event, context):
    """
    Main Lambda handler for data preprocessing operations and quality assessment.
    
    Handles:
    - Data preprocessing operations (POST /preprocess)
    - Dataset quality assessment (GET /quality/{sessionId})
    """
    try:
        logger.info(f"Processing Lambda invoked with event: {json.dumps(event, default=str)}")
        
        # Determine the operation based on HTTP method and path
        http_method = event.get('httpMethod', 'POST')
        path = event.get('path', '')
        
        if http_method == 'GET' and '/quality/' in path:
            return handle_quality_assessment(event, context)
        else:
            return handle_preprocessing_operations(event, context)
            
    except Exception as e:
        logger.error(f"Unexpected error in processing handler: {str(e)}")
        return create_error_response(500, "Internal server error")


def handle_quality_assessment(event, context):
    """
    Handle dataset quality assessment requests.
    """
    try:
        # Extract session ID from path parameters
        session_id = event.get('pathParameters', {}).get('sessionId')
        
        if not session_id:
            return create_error_response(400, "Missing session ID")
        
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
        
        # Determine which dataset to analyze (processed if available, otherwise original)
        s3_key = session_data.get('processed_s3_key', session_data.get('s3_key'))
        
        if not s3_key:
            return create_error_response(404, "No dataset found for session")
        
        # Load dataset from S3
        try:
            response = s3_client.get_object(Bucket=DATA_BUCKET, Key=s3_key)
            df = pd.read_csv(io.BytesIO(response['Body'].read()))
            logger.info(f"Loaded dataset with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load dataset from S3: {str(e)}")
            return create_error_response(500, "Failed to load dataset")
        
        # Perform quality assessment
        quality_report = assess_dataset_quality(df)
        
        # Return quality assessment results
        response_data = {
            'session_id': session_id,
            'dataset_name': session_data.get('dataset_name'),
            'dataset_type': 'processed' if 'processed_s3_key' in session_data else 'original',
            'quality_report': quality_report,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET, OPTIONS'
            },
            'body': json.dumps(response_data, default=str)
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in quality assessment handler: {str(e)}")
        return create_error_response(500, "Internal server error")


def handle_preprocessing_operations(event, context):
    """
    Handle data preprocessing operations.
    
    Handles:
    - Null value handling (removal and filling strategies)
    - Outlier detection and removal using IQR and Z-score methods
    - Data scaling (StandardScaler and MinMaxScaler)
    - Categorical encoding (label encoding and one-hot encoding)
    - Store processed datasets in S3 and log operations in DynamoDB
    """
    try:
        logger.info(f"Processing Lambda invoked with event: {json.dumps(event, default=str)}")
        
        # Parse the request
        body = json.loads(event.get('body', '{}'))
        
        # Extract required parameters
        session_id = body.get('session_id')
        operations = body.get('operations', [])
        
        if not session_id:
            return create_error_response(400, "Missing session_id")
        
        if not operations:
            return create_error_response(400, "No preprocessing operations specified")
        
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
        
        # Load original dataset from S3
        try:
            s3_key = session_data['s3_key']
            response = s3_client.get_object(Bucket=DATA_BUCKET, Key=s3_key)
            df = pd.read_csv(io.BytesIO(response['Body'].read()))
            logger.info(f"Loaded dataset with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load dataset from S3: {str(e)}")
            return create_error_response(500, "Failed to load dataset")
        
        # Process operations sequentially
        processed_df = df.copy()
        operation_results = []
        
        for operation in operations:
            try:
                start_time = datetime.utcnow()
                
                # Normalize operation format for logging
                normalized = normalize_operation(operation)
                action_name = normalized.get('type', operation.get('operation', 'unknown'))
                
                # Execute the preprocessing operation
                processed_df, operation_result = execute_preprocessing_operation(
                    processed_df, operation
                )
                
                end_time = datetime.utcnow()
                duration_ms = int((end_time - start_time).total_seconds() * 1000)
                
                # Log operation details
                operation_log = {
                    'session_id': session_id,
                    'timestamp': start_time.isoformat(),
                    'operation_type': 'preprocessing',
                    'action': action_name,
                    'parameters': normalized.get('parameters', {}),
                    'status': 'completed',
                    'duration_ms': duration_ms,
                    'rows_before': len(df),
                    'rows_after': len(processed_df),
                    'columns_before': len(df.columns),
                    'columns_after': len(processed_df.columns)
                }
                operation_log.update(operation_result)
                
                operation_results.append(operation_log)
                
                # Store operation log in DynamoDB
                operations_table = dynamodb.Table(OPERATIONS_TABLE)
                operations_table.put_item(Item=operation_log)
                
                logger.info(f"Completed operation: {action_name}")
                
            except Exception as e:
                logger.error(f"Failed to execute operation {operation.get('type', operation.get('operation'))}: {str(e)}")
                
                # Log failed operation
                normalized = normalize_operation(operation)
                operation_log = {
                    'session_id': session_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'operation_type': 'preprocessing',
                    'action': normalized.get('type', operation.get('operation', 'unknown')),
                    'parameters': normalized.get('parameters', {}),
                    'status': 'failed',
                    'error': str(e),
                    'rows_before': len(df),
                    'rows_after': len(processed_df)
                }
                
                operation_results.append(operation_log)
                
                # Continue with other operations even if one fails
                continue
        
        # Store processed dataset in S3
        try:
            processed_s3_key = f"datasets/{session_id}/processed.csv"
            csv_buffer = io.StringIO()
            processed_df.to_csv(csv_buffer, index=False)
            
            s3_client.put_object(
                Bucket=DATA_BUCKET,
                Key=processed_s3_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv'
            )
            
            logger.info(f"Processed dataset stored at: {processed_s3_key}")
            
        except Exception as e:
            logger.error(f"Failed to store processed dataset: {str(e)}")
            return create_error_response(500, "Failed to store processed dataset")
        
        # Calculate dataset quality metrics using enhanced assessment
        quality_metrics = assess_dataset_quality(processed_df)
        
        # Update session with processed dataset info
        try:
            sessions_table.update_item(
                Key={'session_id': session_id},
                UpdateExpression='SET processed_s3_key = :key, quality_score = :score, last_updated = :timestamp',
                ExpressionAttributeValues={
                    ':key': processed_s3_key,
                    ':score': quality_metrics['overall_quality_score'],
                    ':timestamp': datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Failed to update session metadata: {str(e)}")
        
        # Generate download URL for processed dataset
        download_url = generate_presigned_url(processed_s3_key)

        # Return success response
        operations_applied = len([op for op in operation_results if op['status'] == 'completed'])
        response_data = {
            'session_id': session_id,
            'operations_applied': operations_applied,
            'operations_completed': operations_applied,
            'operations_failed': len([op for op in operation_results if op['status'] == 'failed']),
            'processed_dataset_location': processed_s3_key,
            'download_url': download_url,
            'processed_dataset': {
                's3_key': processed_s3_key,
                'shape': {
                    'rows': len(processed_df),
                    'columns': len(processed_df.columns)
                }
            },
            'quality_metrics': quality_metrics,
            'operation_results': operation_results,
            'message': 'Preprocessing operations completed'
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            },
            'body': json.dumps(response_data, default=str)
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in processing handler: {str(e)}")
        return create_error_response(500, "Internal server error")


def normalize_operation(operation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize frontend operation format to backend format.
    
    Frontend sends: { operation: "handle_missing", method: "fill", columns: [...] }
    Backend expects: { type: "null_filling", parameters: { strategy: "mean", columns: [...] } }
    """
    # If already in backend format, return as-is
    if 'type' in operation and 'parameters' in operation:
        return operation

    op = operation.get('operation', '')
    method = operation.get('method', '')
    columns = operation.get('columns', [])
    remove = operation.get('remove', False)

    if op == 'handle_missing':
        if method == 'drop':
            return {'type': 'null_removal', 'parameters': {'method': 'drop_rows', 'columns': columns}}
        else:
            return {'type': 'null_filling', 'parameters': {'strategy': 'mean', 'columns': columns}}
    elif op == 'detect_outliers':
        return {'type': 'outlier_removal', 'parameters': {'method': method or 'iqr', 'remove': remove}}
    elif op == 'scale_features':
        return {'type': 'scaling', 'parameters': {'method': method or 'standard', 'columns': columns}}
    elif op == 'encode_categorical':
        if method == 'onehot':
            return {'type': 'one_hot_encoding', 'parameters': {'columns': columns}}
        else:
            return {'type': 'label_encoding', 'parameters': {'columns': columns}}
    else:
        return operation


def execute_preprocessing_operation(df: pd.DataFrame, operation: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute a single preprocessing operation on the dataset.
    
    Args:
        df: Input DataFrame
        operation: Operation configuration dictionary
        
    Returns:
        Tuple of (processed DataFrame, operation result details)
    """
    operation = normalize_operation(operation)
    operation_type = operation.get('type')
    parameters = operation.get('parameters', {})
    
    if operation_type == 'null_removal':
        return handle_null_removal(df, parameters)
    elif operation_type == 'null_filling':
        return handle_null_filling(df, parameters)
    elif operation_type == 'outlier_removal':
        return handle_outlier_removal(df, parameters)
    elif operation_type == 'scaling':
        return handle_scaling(df, parameters)
    elif operation_type == 'label_encoding':
        return handle_label_encoding(df, parameters)
    elif operation_type == 'one_hot_encoding':
        return handle_one_hot_encoding(df, parameters)
    else:
        raise ValueError(f"Unknown operation type: {operation_type}")


def handle_null_removal(df: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle null value removal operations.
    
    Args:
        df: Input DataFrame
        parameters: Operation parameters
        
    Returns:
        Tuple of (processed DataFrame, operation details)
    """
    method = parameters.get('method', 'drop_rows')  # 'drop_rows' or 'drop_columns'
    columns = parameters.get('columns')  # Specific columns to process
    threshold = parameters.get('threshold', 0.0)  # Threshold for dropping columns
    
    original_shape = df.shape
    
    if method == 'drop_rows':
        if columns:
            # Drop rows with null values in specific columns
            processed_df = df.dropna(subset=columns)
        else:
            # Drop rows with any null values
            processed_df = df.dropna()
    elif method == 'drop_columns':
        if columns:
            # Drop specific columns
            processed_df = df.drop(columns=columns)
        else:
            # Drop columns with null percentage above threshold
            null_percentages = df.isnull().sum() / len(df)
            columns_to_drop = null_percentages[null_percentages > threshold].index.tolist()
            processed_df = df.drop(columns=columns_to_drop)
    else:
        raise ValueError(f"Unknown null removal method: {method}")
    
    result_details = {
        'method_used': method,
        'columns_affected': columns or 'all',
        'threshold_used': threshold,
        'rows_removed': original_shape[0] - processed_df.shape[0],
        'columns_removed': original_shape[1] - processed_df.shape[1]
    }
    
    return processed_df, result_details


def handle_null_filling(df: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle null value filling operations.
    
    Args:
        df: Input DataFrame
        parameters: Operation parameters
        
    Returns:
        Tuple of (processed DataFrame, operation details)
    """
    strategy = parameters.get('strategy', 'mean')  # 'mean', 'median', 'mode', 'constant'
    columns = parameters.get('columns')  # Specific columns to process
    fill_value = parameters.get('fill_value')  # For constant strategy
    
    processed_df = df.copy()
    columns_affected = []
    
    # Determine columns to process
    if columns:
        target_columns = [col for col in columns if col in df.columns]
    else:
        target_columns = df.columns.tolist()
    
    for column in target_columns:
        if df[column].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                fill_val = df[column].mean()
                processed_df[column] = processed_df[column].fillna(fill_val)
                columns_affected.append(column)
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                fill_val = df[column].median()
                processed_df[column] = processed_df[column].fillna(fill_val)
                columns_affected.append(column)
            elif strategy == 'mode':
                mode_val = df[column].mode()
                if len(mode_val) > 0:
                    processed_df[column] = processed_df[column].fillna(mode_val[0])
                    columns_affected.append(column)
            elif strategy == 'constant' and fill_value is not None:
                processed_df[column] = processed_df[column].fillna(fill_value)
                columns_affected.append(column)
    
    result_details = {
        'strategy_used': strategy,
        'fill_value_used': fill_value,
        'columns_affected': columns_affected,
        'total_nulls_filled': df.isnull().sum().sum() - processed_df.isnull().sum().sum()
    }
    
    return processed_df, result_details


def handle_outlier_removal(df: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle outlier detection and removal operations.
    
    Args:
        df: Input DataFrame
        parameters: Operation parameters
        
    Returns:
        Tuple of (processed DataFrame, operation details)
    """
    method = parameters.get('method', 'iqr')  # 'iqr' or 'zscore'
    columns = parameters.get('columns')  # Specific columns to process
    threshold = parameters.get('threshold', 3.0)  # Z-score threshold or IQR multiplier
    
    processed_df = df.copy()
    outliers_removed = 0
    columns_affected = []
    
    # Determine numerical columns to process
    if columns:
        target_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    else:
        target_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if method == 'iqr':
        for column in target_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            outliers_in_column = outlier_mask.sum()
            
            if outliers_in_column > 0:
                processed_df = processed_df[~outlier_mask]
                outliers_removed += outliers_in_column
                columns_affected.append(column)
                
    elif method == 'zscore':
        for column in target_columns:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outlier_mask = z_scores > threshold
            outliers_in_column = outlier_mask.sum()
            
            if outliers_in_column > 0:
                processed_df = processed_df[~outlier_mask]
                outliers_removed += outliers_in_column
                columns_affected.append(column)
    else:
        raise ValueError(f"Unknown outlier removal method: {method}")
    
    result_details = {
        'method_used': method,
        'threshold_used': threshold,
        'columns_affected': columns_affected,
        'outliers_removed': outliers_removed
    }
    
    return processed_df, result_details


def handle_scaling(df: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle data scaling operations.
    
    Args:
        df: Input DataFrame
        parameters: Operation parameters
        
    Returns:
        Tuple of (processed DataFrame, operation details)
    """
    method = parameters.get('method', 'standard')  # 'standard' or 'minmax'
    columns = parameters.get('columns')  # Specific columns to process
    
    processed_df = df.copy()
    columns_affected = []
    
    # Determine numerical columns to process
    if columns:
        target_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    else:
        target_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if target_columns:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Apply scaling
        scaled_data = scaler.fit_transform(processed_df[target_columns])
        processed_df[target_columns] = scaled_data
        columns_affected = target_columns
    
    result_details = {
        'method_used': method,
        'columns_affected': columns_affected,
        'scaler_parameters': {
            'feature_names': target_columns
        }
    }
    
    return processed_df, result_details


def handle_label_encoding(df: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle label encoding operations for categorical variables.
    
    Args:
        df: Input DataFrame
        parameters: Operation parameters
        
    Returns:
        Tuple of (processed DataFrame, operation details)
    """
    columns = parameters.get('columns')  # Specific columns to process
    
    processed_df = df.copy()
    columns_affected = []
    encoding_mappings = {}
    
    # Determine categorical columns to process
    if columns:
        target_columns = [col for col in columns if col in df.columns]
    else:
        target_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    
    for column in target_columns:
        if df[column].dtype == 'object' or df[column].dtype.name == 'category':
            encoder = LabelEncoder()
            # Handle NaN values by filling with a placeholder
            temp_series = df[column].fillna('__MISSING__')
            encoded_values = encoder.fit_transform(temp_series)
            
            # Replace back the NaN values
            nan_mask = df[column].isnull()
            processed_df[column] = encoded_values
            processed_df.loc[nan_mask, column] = np.nan
            
            columns_affected.append(column)
            encoding_mappings[column] = {
                'classes': encoder.classes_.tolist(),
                'unique_values': len(encoder.classes_)
            }
    
    result_details = {
        'columns_affected': columns_affected,
        'encoding_mappings': encoding_mappings
    }
    
    return processed_df, result_details


def handle_one_hot_encoding(df: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle one-hot encoding operations for categorical variables.
    
    Args:
        df: Input DataFrame
        parameters: Operation parameters
        
    Returns:
        Tuple of (processed DataFrame, operation details)
    """
    columns = parameters.get('columns')  # Specific columns to process
    drop_first = parameters.get('drop_first', False)  # Drop first category to avoid multicollinearity
    
    processed_df = df.copy()
    columns_affected = []
    new_columns_created = []
    
    # Determine categorical columns to process
    if columns:
        target_columns = [col for col in columns if col in df.columns]
    else:
        target_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    
    for column in target_columns:
        if df[column].dtype == 'object' or df[column].dtype.name == 'category':
            # Create one-hot encoded columns
            dummies = pd.get_dummies(df[column], prefix=column, drop_first=drop_first)
            
            # Add to processed dataframe
            processed_df = pd.concat([processed_df, dummies], axis=1)
            
            # Remove original column
            processed_df = processed_df.drop(columns=[column])
            
            columns_affected.append(column)
            new_columns_created.extend(dummies.columns.tolist())
    
    result_details = {
        'columns_affected': columns_affected,
        'new_columns_created': new_columns_created,
        'drop_first_used': drop_first,
        'total_new_columns': len(new_columns_created)
    }
    
    return processed_df, result_details


def generate_presigned_url(s3_key: str, expiration: int = 3600) -> str:
    """Generate a presigned URL for downloading the processed dataset."""
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': DATA_BUCKET, 'Key': s3_key},
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
        return ""


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