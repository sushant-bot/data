import json
import boto3
import boto3.dynamodb.conditions
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import logging
import os
from typing import Dict, List, Any, Optional
import io
import base64
import sys
import zipfile

# ---------------------------------------------------------------------------
# S3 bootstrap: download all ML packages to /tmp on first cold start.
# Uses a self-contained zip (pandas+numpy+scipy+sklearn+matplotlib+seaborn)
# so numpy ABI is consistent across all packages.
# ---------------------------------------------------------------------------
_ML_CACHE = '/tmp/ml_cache'
if not os.path.exists(os.path.join(_ML_CACHE, 'matplotlib')):
    _s3_boot = boto3.client('s3', region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'))
    _bucket = os.environ.get('BUCKET_NAME', 'ai-data-analyst-platform-data-dev-672627895253')
    _data = _s3_boot.get_object(Bucket=_bucket, Key='lambda-layers/ml-all-packages.zip')['Body'].read()
    os.makedirs(_ML_CACHE, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(_data)) as _zf:
        for _m in _zf.namelist():
            if _m.startswith('python/') and not _m.endswith('/'):
                _t = os.path.join(_ML_CACHE, _m[7:])
                os.makedirs(os.path.dirname(_t), exist_ok=True)
                with open(_t, 'wb') as _f:
                    _f.write(_zf.read(_m))
    del _s3_boot, _bucket, _data, _zf, _m, _t, _f
sys.path.insert(0, _ML_CACHE)
del _ML_CACHE
# ---------------------------------------------------------------------------

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
def confusion_matrix(y_true, y_pred, labels=None):
    """Pure-numpy confusion matrix (replaces sklearn.metrics.confusion_matrix)."""
    if labels is None:
        labels = sorted(list(set(list(y_true) + list(y_pred))))
    idx = {lbl: i for i, lbl in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Environment variables
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'ai-data-analyst-platform-data-dev-672627895253')
OPERATIONS_TABLE = os.environ.get('OPERATIONS_TABLE', 'ai-data-analyst-platform-operations-dev')

CORS_HEADERS = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
}

def lambda_handler(event, context):
    """
    Main Lambda handler for visualization generation.
    Supports correlation heatmaps, confusion matrices, ROC curves, cluster plots, and feature importance.
    """
    try:
        # Parse request
        body = json.loads(event['body']) if isinstance(event.get('body'), str) else event.get('body', {})
        session_id = body.get('session_id')
        visualization_type = body.get('visualization_type')
        parameters = body.get('parameters', {})
        
        if not session_id or not visualization_type:
            return {
                'statusCode': 400,
                'headers': CORS_HEADERS,
                'body': json.dumps({
                    'error': 'Missing required parameters: session_id, visualization_type'
                })
            }
        
        logger.info(f"Generating visualization for session {session_id}, type: {visualization_type}")
        
        # Load dataset
        dataset = load_dataset(session_id, parameters.get('dataset_type', 'processed'))
        if dataset is None:
            return {
                'statusCode': 404,
                'headers': CORS_HEADERS,
                'body': json.dumps({'error': 'Dataset not found'})
            }
        
        # Generate visualization based on type
        visualization_key = None
        
        if visualization_type == 'correlation_heatmap':
            visualization_key = generate_correlation_heatmap(session_id, dataset, parameters)
        elif visualization_type == 'confusion_matrix':
            visualization_key = generate_confusion_matrix_viz(session_id, parameters)
        elif visualization_type == 'roc_curve':
            visualization_key = generate_roc_curve_viz(session_id, parameters)
        elif visualization_type == 'cluster_plot':
            visualization_key = generate_cluster_plot_viz(session_id, dataset, parameters)
        elif visualization_type == 'feature_importance':
            visualization_key = generate_feature_importance_viz(session_id, parameters)
        else:
            return {
                'statusCode': 400,
                'headers': CORS_HEADERS,
                'body': json.dumps({'error': f'Unsupported visualization type: {visualization_type}'})
            }
        
        if not visualization_key:
            return {
                'statusCode': 500,
                'headers': CORS_HEADERS,
                'body': json.dumps({'error': 'Failed to generate visualization'})
            }
        
        # Log operation
        log_visualization_operation(session_id, visualization_type, visualization_key)
        
        # Generate presigned URL for frontend access
        presigned_url = generate_presigned_url(visualization_key)
        
        logger.info(f"Visualization generated successfully: {visualization_key}")
        
        return {
            'statusCode': 200,
            'headers': CORS_HEADERS,
            'body': json.dumps({
                'session_id': session_id,
                'visualization_type': visualization_type,
                'visualization_key': visualization_key,
                'presigned_url': presigned_url
            })
        }
        
    except Exception as e:
        logger.error(f"Error in visualization generation: {str(e)}")
        return {
            'statusCode': 500,
            'headers': CORS_HEADERS,
            'body': json.dumps({'error': f'Internal server error: {str(e)}'})
        }

def load_dataset(session_id: str, dataset_type: str = 'processed') -> Optional[pd.DataFrame]:
    """Load dataset from S3."""
    try:
        if dataset_type == 'original':
            key = f"datasets/{session_id}/original.csv"
        else:
            key = f"datasets/{session_id}/processed.csv"
            
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        return pd.read_csv(io.StringIO(csv_content))
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def generate_correlation_heatmap(session_id: str, dataset: pd.DataFrame, 
                               parameters: Dict[str, Any]) -> Optional[str]:
    """Generate correlation heatmap visualization."""
    try:
        # Select only numeric columns
        numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            logger.warning("Not enough numeric columns for correlation heatmap")
            return None
        
        # Calculate correlation matrix
        correlation_matrix = dataset[numeric_columns].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8}
        )
        
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save to S3
        visualization_key = save_plot_to_s3(session_id, 'correlation_heatmap')
        plt.close()
        
        return visualization_key
        
    except Exception as e:
        logger.error(f"Error generating correlation heatmap: {str(e)}")
        return None

def generate_confusion_matrix_viz(session_id: str, parameters: Dict[str, Any]) -> Optional[str]:
    """Return confusion matrix visualization from stored ML training results."""
    try:
        ml_results = load_ml_results(session_id)
        if not ml_results:
            logger.warning("No ML results found for confusion matrix")
            return None

        # Find the pre-generated confusion matrix S3 key stored by ml_training Lambda
        for key in ml_results.get('visualizations', []):
            if 'confusion_matrix' in str(key):
                return str(key)

        logger.warning("No confusion matrix visualization found in stored ML results")
        return None
    except Exception as e:
        logger.error(f"Error getting confusion matrix: {str(e)}")
        return None

def generate_roc_curve_viz(session_id: str, parameters: Dict[str, Any]) -> Optional[str]:
    """Return ROC curve visualization from stored ML training results."""
    try:
        ml_results = load_ml_results(session_id)
        if not ml_results:
            logger.warning("No ML results found for ROC curve")
            return None

        for key in ml_results.get('visualizations', []):
            if 'roc_curve' in str(key):
                return str(key)

        logger.warning("No ROC curve visualization found in stored ML results")
        return None
    except Exception as e:
        logger.error(f"Error getting ROC curve: {str(e)}")
        return None

def generate_cluster_plot_viz(session_id: str, dataset: pd.DataFrame, 
                            parameters: Dict[str, Any]) -> Optional[str]:
    """Return cluster plot from stored ML training results, or re-generate using KMeans."""
    try:
        # First try to return stored cluster_plot from ml_training results
        ml_results = load_ml_results(session_id)
        if ml_results:
            for key in ml_results.get('visualizations', []):
                if 'cluster_plot' in str(key):
                    return str(key)

        # Fallback: re-generate with KMeans on the dataset's numeric columns
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric features for cluster plot")
            return None

        x_col, y_col = numeric_cols[0], numeric_cols[1]
        from sklearn.cluster import KMeans as _KMeans
        km = _KMeans(n_clusters=3, random_state=42, n_init='auto')
        labels = km.fit_predict(dataset[numeric_cols].fillna(0))

        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(dataset[x_col], dataset[y_col], c=labels, cmap='viridis', alpha=0.7, s=40)
        plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                    c='red', marker='x', s=200, linewidths=3, label='Centroids')
        plt.xlabel(x_col, fontsize=11)
        plt.ylabel(y_col, fontsize=11)
        plt.title('Cluster Analysis (KMeans, k=3)', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        key = save_plot_to_s3(session_id, 'cluster_plot')
        plt.close()
        return key

    except Exception as e:
        logger.error(f"Error generating cluster plot: {str(e)}")
        return None
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to S3
        visualization_key = save_plot_to_s3(session_id, 'cluster_plot')
        plt.close()
        
        return visualization_key
        
    except Exception as e:
        logger.error(f"Error generating cluster plot: {str(e)}")
        return None

def generate_feature_importance_viz(session_id: str, parameters: Dict[str, Any]) -> Optional[str]:
    """Return feature importance visualization from stored ML training results."""
    try:
        ml_results = load_ml_results(session_id)
        if not ml_results:
            logger.warning("No ML results found for feature importance")
            return None

        for key in ml_results.get('visualizations', []):
            if 'feature_importance' in str(key):
                return str(key)

        logger.warning("No feature importance visualization found in stored ML results")
        return None
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return None


def save_plot_to_s3(session_id: str, chart_type: str) -> Optional[str]:
    """Save the current matplotlib figure to S3 as a PNG image."""
    try:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)

        visualization_id = str(uuid.uuid4())[:8]
        key = f"visualizations/{session_id}/{chart_type}_{visualization_id}.png"

        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=buf.getvalue(),
            ContentType='image/png'
        )

        logger.info(f"Visualization saved to S3: {key}")
        return key

    except Exception as e:
        logger.error(f"Error saving plot to S3: {str(e)}")
        return None


def load_ml_results(session_id: str) -> Optional[Dict[str, Any]]:
    """Load the most recent ML training results from the Operations DynamoDB table."""
    try:
        table = dynamodb.Table(OPERATIONS_TABLE)
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key('session_id').eq(session_id),
            ScanIndexForward=False,
            Limit=20
        )

        for item in response.get('Items', []):
            # ml_training Lambda stores results with operation_type = 'ml_results'
            if item.get('operation_type') == 'ml_results':
                return item

        return None

    except Exception as e:
        logger.error(f"Error loading ML results: {str(e)}")
        return None


def log_visualization_operation(session_id: str, visualization_type: str, visualization_key: str) -> None:
    """Log visualization generation operation to DynamoDB."""
    try:
        table = dynamodb.Table(OPERATIONS_TABLE)
        table.put_item(
            Item={
                'session_id': session_id,
                'timestamp': datetime.utcnow().isoformat(),
                'operation_type': 'visualization',
                'visualization_type': visualization_type,
                'visualization_key': visualization_key,
                'status': 'completed'
            }
        )
        logger.info(f"Visualization operation logged for session {session_id}")

    except Exception as e:
        logger.error(f"Error logging visualization operation: {str(e)}")


def generate_presigned_url(key: str, expiration: int = 3600) -> str:
    """Generate a presigned URL for accessing a visualization in S3."""
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': key},
            ExpiresIn=expiration
        )
        return url

    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
        return ""