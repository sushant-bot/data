import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import logging
import os
from typing import Dict, List, Any, Tuple, Optional
import io
import base64

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, silhouette_score
)
from sklearn.preprocessing import LabelEncoder

# Lazy imports for matplotlib/seaborn (loaded only when generating visualizations)
plt = None
sns = None

def _ensure_matplotlib():
    """Lazily import matplotlib and seaborn to avoid import errors when not in layer."""
    global plt, sns
    if plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt
        import seaborn as _sns
        plt = _plt
        sns = _sns

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')

# Environment variables
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'ai-data-analyst-platform-data-dev-077437903006')
SESSIONS_TABLE = os.environ.get('SESSIONS_TABLE', 'ai-data-analyst-platform-sessions-dev')
OPERATIONS_TABLE = os.environ.get('OPERATIONS_TABLE', 'ai-data-analyst-platform-operations-dev')

CORS_HEADERS = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS'
}

def lambda_handler(event, context):
    """
    Main Lambda handler for ML training operations.
    Supports async training pattern to avoid API Gateway 29s timeout.

    Three modes:
    1. Start training (default): validates input, stores pending status, invokes self async, returns operation_id
    2. Async execution (_async_training=True): performs actual training, stores results
    3. Check status (action='check_status'): returns current training status/results from DynamoDB
    """
    try:
        # Check if this is an async invocation (no 'body' key from API Gateway)
        if event.get('_async_training'):
            return _execute_training(event)

        # Parse request from API Gateway
        body = json.loads(event['body']) if isinstance(event.get('body'), str) else event.get('body', {})

        # Check if this is a status check request
        if body.get('action') == 'check_status':
            return _check_training_status(body)

        # Otherwise, start a new training job
        return _start_training(body, context)

    except Exception as e:
        logger.error(f"Error in ML training: {str(e)}")
        return {
            'statusCode': 500,
            'headers': CORS_HEADERS,
            'body': json.dumps({'error': f'Internal server error: {str(e)}'})
        }


def _start_training(body, context):
    """Validate request, store pending status, invoke self asynchronously, and return immediately."""
    session_id = body.get('session_id')
    model_type = body.get('model_type')
    algorithm = body.get('algorithm')
    target_column = body.get('target_column')
    feature_columns = body.get('feature_columns', [])
    parameters = body.get('parameters', {})

    if not session_id or not model_type or not algorithm:
        return {
            'statusCode': 400,
            'headers': CORS_HEADERS,
            'body': json.dumps({
                'error': 'Missing required parameters: session_id, model_type, algorithm'
            })
        }

    logger.info(f"Starting async ML training for session {session_id}, type: {model_type}, algorithm: {algorithm}")

    # Generate operation ID
    operation_id = str(uuid.uuid4())

    # Store initial "training" status in DynamoDB
    operations_table = dynamodb.Table(OPERATIONS_TABLE)
    operations_table.put_item(Item={
        'session_id': session_id,
        'timestamp': f"ml_training_{operation_id}",
        'operation_id': operation_id,
        'operation_type': 'ml_training',
        'status': 'training',
        'model_type': model_type,
        'algorithm': algorithm,
        'target_column': target_column,
        'feature_columns': feature_columns,
        'parameters': convert_floats_to_decimal(parameters),
        'created_at': datetime.now().isoformat(),
    })

    # Invoke self asynchronously for the actual training
    async_payload = {
        '_async_training': True,
        'session_id': session_id,
        'operation_id': operation_id,
        'model_type': model_type,
        'algorithm': algorithm,
        'target_column': target_column,
        'feature_columns': feature_columns,
        'parameters': parameters,
    }

    function_name = context.function_name
    lambda_client.invoke(
        FunctionName=function_name,
        InvocationType='Event',  # Async invocation
        Payload=json.dumps(async_payload),
    )

    logger.info(f"Async training invoked for operation {operation_id}")

    return {
        'statusCode': 202,
        'headers': CORS_HEADERS,
        'body': json.dumps({
            'session_id': session_id,
            'operation_id': operation_id,
            'status': 'training',
            'message': 'Training started. Poll with action=check_status to get results.',
        })
    }


def _check_training_status(body):
    """Check the status of a training operation from DynamoDB."""
    session_id = body.get('session_id')
    operation_id = body.get('operation_id')

    if not session_id or not operation_id:
        return {
            'statusCode': 400,
            'headers': CORS_HEADERS,
            'body': json.dumps({'error': 'Missing session_id or operation_id'})
        }

    operations_table = dynamodb.Table(OPERATIONS_TABLE)

    # Fetch the training record
    response = operations_table.get_item(Key={
        'session_id': session_id,
        'timestamp': f"ml_training_{operation_id}",
    })

    item = response.get('Item')
    if not item:
        return {
            'statusCode': 404,
            'headers': CORS_HEADERS,
            'body': json.dumps({'error': 'Training operation not found'})
        }

    status = item.get('status', 'unknown')

    result = {
        'session_id': session_id,
        'operation_id': operation_id,
        'status': status,
        'model_type': item.get('model_type'),
        'algorithm': item.get('algorithm'),
    }

    if status == 'completed':
        result['results'] = {
            'metrics': convert_decimals_to_floats(item.get('metrics', {})),
            'training_details': convert_decimals_to_floats(item.get('training_details', {})),
            'visualizations': item.get('visualizations', []),
            'feature_columns': item.get('feature_columns', []),
            'target_column': item.get('target_column'),
        }
    elif status == 'failed':
        result['error'] = item.get('error_message', 'Training failed')

    return {
        'statusCode': 200,
        'headers': CORS_HEADERS,
        'body': json.dumps(result, default=str)
    }


def _execute_training(event):
    """Perform the actual ML training (called asynchronously)."""
    session_id = event['session_id']
    operation_id = event['operation_id']
    model_type = event['model_type']
    algorithm = event['algorithm']
    target_column = event.get('target_column')
    feature_columns = event.get('feature_columns', [])
    parameters = event.get('parameters', {})

    operations_table = dynamodb.Table(OPERATIONS_TABLE)

    try:
        logger.info(f"Executing training for operation {operation_id}, session {session_id}")

        # Load processed dataset
        dataset = load_processed_dataset(session_id)
        if dataset is None:
            operations_table.update_item(
                Key={'session_id': session_id, 'timestamp': f"ml_training_{operation_id}"},
                UpdateExpression='SET #s = :s, error_message = :e',
                ExpressionAttributeNames={'#s': 'status'},
                ExpressionAttributeValues={':s': 'failed', ':e': 'Processed dataset not found'},
            )
            return

        # Perform ML training based on type
        if model_type == 'supervised':
            result = train_supervised_model(dataset, algorithm, target_column, feature_columns, parameters)
        elif model_type == 'unsupervised':
            result = train_unsupervised_model(dataset, algorithm, feature_columns, parameters)
        else:
            operations_table.update_item(
                Key={'session_id': session_id, 'timestamp': f"ml_training_{operation_id}"},
                UpdateExpression='SET #s = :s, error_message = :e',
                ExpressionAttributeNames={'#s': 'status'},
                ExpressionAttributeValues={':s': 'failed', ':e': f'Invalid model_type: {model_type}'},
            )
            return

        # Store results in the operation record
        operations_table.update_item(
            Key={'session_id': session_id, 'timestamp': f"ml_training_{operation_id}"},
            UpdateExpression='SET #s = :s, metrics = :m, visualizations = :v, training_details = :td, completed_at = :ca',
            ExpressionAttributeNames={'#s': 'status'},
            ExpressionAttributeValues={
                ':s': 'completed',
                ':m': convert_floats_to_decimal(result.get('metrics', {})),
                ':v': result.get('visualizations', []),
                ':td': convert_floats_to_decimal({
                    'model_type': result.get('model_type'),
                    'algorithm': result.get('algorithm'),
                    'feature_columns': result.get('feature_columns', []),
                    'target_column': result.get('target_column'),
                    'n_clusters': result.get('n_clusters'),
                }),
                ':ca': datetime.now().isoformat(),
            },
        )

        # Also store in the original ml_results format for backwards compatibility
        store_ml_results(session_id, result)

        logger.info(f"Training completed for operation {operation_id}")

    except Exception as e:
        logger.error(f"Training failed for operation {operation_id}: {str(e)}")
        try:
            operations_table.update_item(
                Key={'session_id': session_id, 'timestamp': f"ml_training_{operation_id}"},
                UpdateExpression='SET #s = :s, error_message = :e',
                ExpressionAttributeNames={'#s': 'status'},
                ExpressionAttributeValues={':s': 'failed', ':e': str(e)},
            )
        except Exception as update_err:
            logger.error(f"Failed to update operation status: {str(update_err)}")

def load_processed_dataset(session_id: str) -> Optional[pd.DataFrame]:
    """Load processed dataset from S3."""
    try:
        key = f"datasets/{session_id}/processed.csv"
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        return pd.read_csv(io.StringIO(csv_content))
    except Exception as e:
        logger.error(f"Error loading processed dataset: {str(e)}")
        return None

def train_supervised_model(dataset: pd.DataFrame, algorithm: str, target_column: str, 
                         feature_columns: List[str], parameters: Dict) -> Dict[str, Any]:
    """Train supervised learning models."""
    try:
        # Validate target column
        if target_column not in dataset.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Use all columns except target if no features specified
        if not feature_columns:
            feature_columns = [col for col in dataset.columns if col != target_column]
        
        # Prepare data
        X = dataset[feature_columns]
        y = dataset[target_column]
        
        # Handle categorical target variable
        label_encoder = None
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        # Split data
        test_size = parameters.get('test_size', 0.2)
        random_state = parameters.get('random_state', 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        model = create_supervised_model(algorithm, parameters)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = calculate_supervised_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate visualizations
        visualizations = generate_supervised_visualizations(
            y_test, y_pred, y_pred_proba, feature_columns, model, algorithm
        )
        
        return {
            'model_type': 'supervised',
            'algorithm': algorithm,
            'target_column': target_column,
            'feature_columns': feature_columns,
            'metrics': metrics,
            'visualizations': visualizations,
            'label_encoder_classes': label_encoder.classes_.tolist() if label_encoder else None
        }
        
    except Exception as e:
        logger.error(f"Error in supervised training: {str(e)}")
        raise

def train_unsupervised_model(dataset: pd.DataFrame, algorithm: str, 
                           feature_columns: List[str], parameters: Dict) -> Dict[str, Any]:
    """Train unsupervised learning models."""
    try:
        # Use all numeric columns if no features specified
        if not feature_columns:
            feature_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        
        # Prepare data
        X = dataset[feature_columns]
        
        # Train model
        model = create_unsupervised_model(algorithm, parameters)
        
        if algorithm in ['kmeans', 'k-means']:
            labels = model.fit_predict(X)
            cluster_centers = model.cluster_centers_
        elif algorithm == 'dbscan':
            labels = model.fit_predict(X)
            cluster_centers = None
        else:
            raise ValueError(f"Unsupported unsupervised algorithm: {algorithm}")
        
        # Calculate metrics
        metrics = calculate_unsupervised_metrics(X, labels)
        
        # Generate visualizations
        visualizations = generate_unsupervised_visualizations(
            X, labels, feature_columns, algorithm, cluster_centers
        )
        
        return {
            'model_type': 'unsupervised',
            'algorithm': algorithm,
            'feature_columns': feature_columns,
            'metrics': metrics,
            'visualizations': visualizations,
            'n_clusters': len(np.unique(labels[labels != -1])) if algorithm == 'dbscan' else len(np.unique(labels))
        }
        
    except Exception as e:
        logger.error(f"Error in unsupervised training: {str(e)}")
        raise

def create_supervised_model(algorithm: str, parameters: Dict):
    """Create supervised learning model based on algorithm."""
    if algorithm == 'logistic_regression':
        return LogisticRegression(
            random_state=parameters.get('random_state', 42),
            max_iter=parameters.get('max_iter', 1000)
        )
    elif algorithm == 'random_forest':
        return RandomForestClassifier(
            n_estimators=parameters.get('n_estimators', 100),
            random_state=parameters.get('random_state', 42),
            max_depth=parameters.get('max_depth', None)
        )
    elif algorithm == 'knn':
        return KNeighborsClassifier(
            n_neighbors=parameters.get('n_neighbors', 5)
        )
    elif algorithm == 'svm':
        return SVC(
            kernel=parameters.get('kernel', 'rbf'),
            random_state=parameters.get('random_state', 42),
            probability=True  # Enable probability estimates
        )
    else:
        raise ValueError(f"Unsupported supervised algorithm: {algorithm}")

def create_unsupervised_model(algorithm: str, parameters: Dict):
    """Create unsupervised learning model based on algorithm."""
    if algorithm in ['kmeans', 'k-means']:
        return KMeans(
            n_clusters=parameters.get('n_clusters', 3),
            random_state=parameters.get('random_state', 42)
        )
    elif algorithm == 'dbscan':
        return DBSCAN(
            eps=parameters.get('eps', 0.5),
            min_samples=parameters.get('min_samples', 5)
        )
    else:
        raise ValueError(f"Unsupported unsupervised algorithm: {algorithm}")

def calculate_supervised_metrics(y_true, y_pred, y_pred_proba=None) -> Dict[str, float]:
    """Calculate metrics for supervised learning."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }
    
    # Add AUC for binary classification
    if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        metrics['auc'] = float(auc(fpr, tpr))
    
    return metrics

def calculate_unsupervised_metrics(X, labels) -> Dict[str, float]:
    """Calculate metrics for unsupervised learning."""
    metrics = {}
    
    # Silhouette score (only if we have more than 1 cluster and less than n_samples)
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1 and len(unique_labels) < len(X):
        # Filter out noise points for DBSCAN
        if -1 in unique_labels:
            mask = labels != -1
            if np.sum(mask) > 1 and len(np.unique(labels[mask])) > 1:
                metrics['silhouette_score'] = float(silhouette_score(X[mask], labels[mask]))
        else:
            metrics['silhouette_score'] = float(silhouette_score(X, labels))
    
    # Number of clusters
    n_clusters = len(unique_labels)
    if -1 in unique_labels:  # DBSCAN noise points
        n_clusters -= 1
    metrics['n_clusters'] = n_clusters
    
    # Number of noise points (for DBSCAN)
    if -1 in unique_labels:
        metrics['n_noise_points'] = int(np.sum(labels == -1))
    
    return metrics
def generate_supervised_visualizations(y_true, y_pred, y_pred_proba, feature_columns, 
                                     model, algorithm) -> List[str]:
    """Generate visualizations for supervised learning results."""
    visualizations = []
    _ensure_matplotlib()

    try:
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {algorithm.replace("_", " ").title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        confusion_matrix_key = save_plot_to_s3('confusion_matrix', algorithm)
        visualizations.append(confusion_matrix_key)
        plt.close()
        
        # ROC Curve (for binary classification)
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {algorithm.replace("_", " ").title()}')
            plt.legend(loc="lower right")
            
            roc_curve_key = save_plot_to_s3('roc_curve', algorithm)
            visualizations.append(roc_curve_key)
            plt.close()
        
        # Feature Importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(f'Feature Importance - {algorithm.replace("_", " ").title()}')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), 
                      [feature_columns[i] for i in indices], rotation=45)
            plt.tight_layout()
            
            feature_importance_key = save_plot_to_s3('feature_importance', algorithm)
            visualizations.append(feature_importance_key)
            plt.close()
            
    except Exception as e:
        logger.error(f"Error generating supervised visualizations: {str(e)}")
    
    return visualizations

def generate_unsupervised_visualizations(X, labels, feature_columns, algorithm, 
                                       cluster_centers=None) -> List[str]:
    """Generate visualizations for unsupervised learning results."""
    visualizations = []
    _ensure_matplotlib()

    try:
        # Cluster Plot (2D projection using first two features)
        if len(feature_columns) >= 2:
            plt.figure(figsize=(10, 8))
            
            # Use first two features for 2D visualization
            x_col, y_col = feature_columns[0], feature_columns[1]
            scatter = plt.scatter(X[x_col], X[y_col], c=labels, cmap='viridis', alpha=0.7)
            
            # Plot cluster centers if available
            if cluster_centers is not None:
                plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                           c='red', marker='x', s=200, linewidths=3, label='Centroids')
                plt.legend()
            
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f'Cluster Plot - {algorithm.replace("_", " ").title()}')
            plt.colorbar(scatter)
            
            cluster_plot_key = save_plot_to_s3('cluster_plot', algorithm)
            visualizations.append(cluster_plot_key)
            plt.close()
        
        # Cluster Distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(range(len(unique_labels)), counts)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Points')
        plt.title(f'Cluster Distribution - {algorithm.replace("_", " ").title()}')
        plt.xticks(range(len(unique_labels)), 
                  [f'Cluster {i}' if i != -1 else 'Noise' for i in unique_labels])
        
        # Color bars
        for i, bar in enumerate(bars):
            if unique_labels[i] == -1:  # Noise points
                bar.set_color('red')
            else:
                bar.set_color(plt.cm.viridis(i / len(unique_labels)))
        
        cluster_distribution_key = save_plot_to_s3('cluster_distribution', algorithm)
        visualizations.append(cluster_distribution_key)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating unsupervised visualizations: {str(e)}")
    
    return visualizations

def save_plot_to_s3(plot_type: str, algorithm: str) -> str:
    """Save matplotlib plot to S3 and return the key."""
    _ensure_matplotlib()
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{plot_type}_{algorithm}_{timestamp}.png"
        key = f"visualizations/{filename}"
        
        # Save plot to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=buffer.getvalue(),
            ContentType='image/png'
        )
        
        logger.info(f"Saved visualization to S3: {key}")
        return key
        
    except Exception as e:
        logger.error(f"Error saving plot to S3: {str(e)}")
        return ""

def store_ml_results(session_id: str, results: Dict[str, Any]):
    """Store ML training results in DynamoDB."""
    try:
        operations_table = dynamodb.Table(OPERATIONS_TABLE)

        # Convert floats to Decimal for DynamoDB compatibility
        item = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'operation_type': 'ml_results',
            'model_type': results['model_type'],
            'algorithm': results['algorithm'],
            'metrics': json.loads(json.dumps(results['metrics'], default=str)),
            'visualizations': results['visualizations'],
            'feature_columns': results.get('feature_columns', []),
            'target_column': results.get('target_column'),
            'n_clusters': results.get('n_clusters')
        }
        item = convert_floats_to_decimal(item)

        operations_table.put_item(Item=item)

        logger.info(f"Stored ML results for session {session_id}")

    except Exception as e:
        logger.error(f"Error storing ML results: {str(e)}")
        raise


def convert_floats_to_decimal(obj):
    """Recursively convert float values to Decimal for DynamoDB."""
    from decimal import Decimal
    if isinstance(obj, float):
        if obj != obj:  # NaN check
            return None
        return Decimal(str(round(obj, 10)))
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimal(i) for i in obj]
    elif isinstance(obj, np.floating):
        return Decimal(str(round(float(obj), 10)))
    elif isinstance(obj, np.integer):
        return int(obj)
    return obj


def convert_decimals_to_floats(obj):
    """Recursively convert Decimal values back to floats for JSON serialization."""
    from decimal import Decimal
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimals_to_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals_to_floats(i) for i in obj]
    return obj