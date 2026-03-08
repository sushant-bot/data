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

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Environment variables
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'ai-data-analyst-platform-data-dev-077437903006')
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
            parameters['_session_id'] = session_id
            visualization_key = generate_feature_importance_viz(session_id, parameters)
        else:
            return {
                'statusCode': 400,
                'headers': CORS_HEADERS,
                'body': json.dumps({'error': f'Unsupported visualization type: {visualization_type}'})
            }
        
        # Types that strictly require ML training results
        ML_DEPENDENT_TYPES = {'confusion_matrix', 'roc_curve'}
        
        if not visualization_key and visualization_type in ML_DEPENDENT_TYPES:
            return {
                'statusCode': 400,
                'headers': CORS_HEADERS,
                'body': json.dumps({
                    'error': f'{visualization_type.replace("_", " ").title()} requires ML model training first. '
                             f'Please train a model in the ML Training tab before generating this visualization.'
                })
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
    """Load dataset from S3, falling back to original if processed is not found."""
    try:
        if dataset_type == 'original':
            key = f"datasets/{session_id}/original.csv"
        else:
            key = f"datasets/{session_id}/processed.csv"
            
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        return pd.read_csv(io.StringIO(csv_content))
    except Exception as e:
        error_str = str(e)
        is_not_found = 'NoSuchKey' in error_str or 'Not Found' in error_str or '404' in error_str
        if is_not_found and dataset_type != 'original':
            logger.info(f"Processed dataset not found, falling back to original")
            return load_dataset(session_id, 'original')
        logger.error(f"Error loading dataset: {error_str}")
        return None
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
    """Generate confusion matrix visualization from ML results."""
    try:
        # Try to retrieve stored visualization from ML training results
        stored_key = get_stored_visualization(session_id, 'confusion_matrix')
        if stored_key:
            return stored_key

        # Load ML results from DynamoDB
        ml_results = load_ml_results(session_id, 'supervised')
        if not ml_results:
            logger.warning("No supervised ML results found for confusion matrix")
            return None
        
        # Extract confusion matrix data from parameters or recreate
        y_true = parameters.get('y_true')
        y_pred = parameters.get('y_pred')
        labels = parameters.get('labels')
        
        if not y_true or not y_pred:
            logger.warning("Missing true/predicted values for confusion matrix")
            return None
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels if labels else range(len(cm)),
            yticklabels=labels if labels else range(len(cm))
        )
        
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save to S3
        visualization_key = save_plot_to_s3(session_id, 'confusion_matrix')
        plt.close()
        
        return visualization_key
        
    except Exception as e:
        logger.error(f"Error generating confusion matrix: {str(e)}")
        return None

def generate_roc_curve_viz(session_id: str, parameters: Dict[str, Any]) -> Optional[str]:
    """Generate ROC curve visualization."""
    try:
        # Try to retrieve stored visualization from ML training results
        stored_key = get_stored_visualization(session_id, 'roc_curve')
        if stored_key:
            return stored_key

        from sklearn.metrics import roc_curve, auc
        
        # Extract ROC curve data from parameters
        y_true = parameters.get('y_true')
        y_scores = parameters.get('y_scores')
        
        if not y_true or not y_scores:
            logger.warning("Missing data for ROC curve generation")
            return None
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to S3
        visualization_key = save_plot_to_s3(session_id, 'roc_curve')
        plt.close()
        
        return visualization_key
        
    except Exception as e:
        logger.error(f"Error generating ROC curve: {str(e)}")
        return None

def generate_cluster_plot_viz(session_id: str, dataset: pd.DataFrame, 
                            parameters: Dict[str, Any]) -> Optional[str]:
    """Generate cluster plot visualization."""
    try:
        # Try to retrieve stored visualization from ML training results
        stored_key = get_stored_visualization(session_id, 'cluster_plot')
        if stored_key:
            return stored_key

        # Get numeric columns for clustering
        numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) < 2:
            logger.warning("Not enough numeric features for cluster plot")
            return None

        # Extract clustering parameters or auto-generate
        labels = parameters.get('labels')
        feature_columns = parameters.get('feature_columns', [])
        cluster_centers = parameters.get('cluster_centers')

        if not feature_columns or len(feature_columns) < 2:
            feature_columns = numeric_columns[:2]

        # If no labels provided, auto-run KMeans clustering
        if labels is None:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            n_clusters = parameters.get('n_clusters', 3)
            X = dataset[numeric_columns].dropna()
            if len(X) < n_clusters:
                logger.warning("Not enough data points for clustering")
                return None
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            # Transform centers back using the first 2 feature columns
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
            # Use aligned index for plotting
            plot_data = dataset.loc[X.index]
        else:
            plot_data = dataset
        
        x_col, y_col = feature_columns[0], feature_columns[1]
        
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        scatter = plt.scatter(
            plot_data[x_col], plot_data[y_col], 
            c=labels, cmap='viridis', alpha=0.7, s=50
        )
        
        # Plot cluster centers if available
        if cluster_centers is not None:
            x_idx = numeric_columns.index(x_col) if x_col in numeric_columns else 0
            y_idx = numeric_columns.index(y_col) if y_col in numeric_columns else 1
            plt.scatter(
                [c[x_idx] for c in cluster_centers],
                [c[y_idx] for c in cluster_centers],
                c='red', marker='x', s=200, linewidths=3, 
                label='Centroids'
            )
            plt.legend()
        
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        plt.title('Cluster Analysis Results', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
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
    """Generate feature importance visualization from ML results or dataset variance."""
    try:
        # Try to retrieve stored visualization from ML training results
        stored_key = get_stored_visualization(session_id, 'feature_importance')
        if stored_key:
            return stored_key

        # Extract feature importance data from parameters
        importances = parameters.get('importances')
        feature_names = parameters.get('feature_names')
        
        # If no explicit importance data, compute variance-based importance from dataset
        if not importances or not feature_names:
            dataset_type = parameters.get('dataset_type', 'processed')
            dataset = load_dataset(parameters.get('_session_id', ''), dataset_type)
            if dataset is None:
                logger.warning("No dataset available for feature importance")
                return None
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 1:
                logger.warning("No numeric columns for feature importance")
                return None
            # Use normalized variance as a proxy for feature importance
            variances = dataset[numeric_cols].var()
            total_var = variances.sum()
            if total_var > 0:
                importances = (variances / total_var).tolist()
            else:
                importances = [1.0 / len(numeric_cols)] * len(numeric_cols)
            feature_names = numeric_cols
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        sorted_importances = [importances[i] for i in indices]
        sorted_features = [feature_names[i] for i in indices]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(sorted_importances)), sorted_importances)
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.title('Feature Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save to S3
        visualization_key = save_plot_to_s3(session_id, 'feature_importance')
        plt.close()
        
        return visualization_key
        
    except Exception as e:
        logger.error(f"Error generating feature importance plot: {str(e)}")
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


def load_ml_results(session_id: str, result_type: str) -> Optional[Dict[str, Any]]:
    """Load ML training results from DynamoDB Operations table."""
    try:
        table = dynamodb.Table(OPERATIONS_TABLE)
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key('session_id').eq(session_id),
            ScanIndexForward=False,
            Limit=10
        )

        for item in response.get('Items', []):
            if item.get('operation_type') == 'ml_results' and item.get('model_type') == result_type:
                return item

        return None

    except Exception as e:
        logger.error(f"Error loading ML results: {str(e)}")
        return None


def get_stored_visualization(session_id: str, viz_type: str) -> Optional[str]:
    """Retrieve a stored visualization key from ML training results."""
    try:
        # Check supervised results first
        ml_results = load_ml_results(session_id, 'supervised')
        if ml_results and ml_results.get('visualizations'):
            for key in ml_results['visualizations']:
                if viz_type in key:
                    # Verify the object exists in S3
                    try:
                        s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
                        logger.info(f"Found stored {viz_type} visualization: {key}")
                        return key
                    except Exception:
                        continue

        # Check unsupervised results
        ml_results = load_ml_results(session_id, 'unsupervised')
        if ml_results and ml_results.get('visualizations'):
            for key in ml_results['visualizations']:
                if viz_type in key:
                    try:
                        s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
                        logger.info(f"Found stored {viz_type} visualization: {key}")
                        return key
                    except Exception:
                        continue

        return None
    except Exception as e:
        logger.error(f"Error retrieving stored visualization: {str(e)}")
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