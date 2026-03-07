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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
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
    Supports both supervised and unsupervised learning.
    """
    try:
        # Parse request
        body = json.loads(event['body']) if isinstance(event.get('body'), str) else event.get('body', {})
        session_id = body.get('session_id')
        model_type = body.get('model_type')  # 'supervised' or 'unsupervised'
        algorithm = body.get('algorithm')
        target_column = body.get('target_column')  # For supervised learning
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
        
        logger.info(f"Starting ML training for session {session_id}, type: {model_type}, algorithm: {algorithm}")
        
        # Load processed dataset
        dataset = load_processed_dataset(session_id)
        if dataset is None:
            return {
                'statusCode': 404,
                'headers': CORS_HEADERS,
                'body': json.dumps({'error': 'Processed dataset not found'})
            }
        
        # Log operation start
        operation_id = str(uuid.uuid4())
        log_operation(session_id, operation_id, 'ml_training', 'started', {
            'model_type': model_type,
            'algorithm': algorithm,
            'target_column': target_column,
            'feature_columns': feature_columns
        })
        
        # Perform ML training based on type
        if model_type == 'supervised':
            result = train_supervised_model(dataset, algorithm, target_column, feature_columns, parameters)
        elif model_type == 'unsupervised':
            result = train_unsupervised_model(dataset, algorithm, feature_columns, parameters)
        else:
            return {
                'statusCode': 400,
                'headers': CORS_HEADERS,
                'body': json.dumps({'error': f'Invalid model_type: {model_type}'})
            }
        
        # Store results
        store_ml_results(session_id, result)
        
        # Log operation completion
        log_operation(session_id, operation_id, 'ml_training', 'completed', {
            'model_type': model_type,
            'algorithm': algorithm,
            'metrics': result.get('metrics', {})
        })
        
        logger.info(f"ML training completed for session {session_id}")
        
        return {
            'statusCode': 200,
            'headers': CORS_HEADERS,
            'body': json.dumps({
                'session_id': session_id,
                'model_type': model_type,
                'algorithm': algorithm,
                'results': result
            })
        }
        
    except Exception as e:
        logger.error(f"Error in ML training: {str(e)}")
        return {
            'statusCode': 500,
            'headers': CORS_HEADERS,
            'body': json.dumps({'error': f'Internal server error: {str(e)}'})
        }

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
        
        # Store results
        operations_table.put_item(
            Item={
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'operation_type': 'ml_results',
                'model_type': results['model_type'],
                'algorithm': results['algorithm'],
                'metrics': results['metrics'],
                'visualizations': results['visualizations'],
                'feature_columns': results.get('feature_columns', []),
                'target_column': results.get('target_column'),
                'n_clusters': results.get('n_clusters')
            }
        )
        
        logger.info(f"Stored ML results for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error storing ML results: {str(e)}")
        raise

def log_operation(session_id: str, operation_id: str, operation_type: str, 
                 status: str, details: Dict[str, Any]):
    """Log operation to DynamoDB."""
    try:
        operations_table = dynamodb.Table(OPERATIONS_TABLE)
        
        operations_table.put_item(
            Item={
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'operation_id': operation_id,
                'operation_type': operation_type,
                'status': status,
                'details': details
            }
        )
        
    except Exception as e:
        logger.error(f"Error logging operation: {str(e)}")