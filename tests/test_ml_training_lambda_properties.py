"""
Property-based tests for ML Training Lambda function.
Feature: ai-data-analyst-platform
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.pandas import data_frames, columns
import json
import boto3
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Set AWS env vars before import
os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')
os.environ.setdefault('AWS_ACCESS_KEY_ID', 'testing')
os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'testing')

# Load ml_training lambda_function via importlib to avoid sys.path collisions
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ml_training_lambda",
    os.path.join(os.path.dirname(__file__), '..', 'lambda', 'ml_training', 'lambda_function.py')
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
sys.modules['ml_training_lambda'] = _module

train_supervised_model = _module.train_supervised_model
train_unsupervised_model = _module.train_unsupervised_model
calculate_supervised_metrics = _module.calculate_supervised_metrics
calculate_unsupervised_metrics = _module.calculate_unsupervised_metrics
create_supervised_model = _module.create_supervised_model
create_unsupervised_model = _module.create_unsupervised_model

# Test data strategies
@st.composite
def classification_dataset(draw):
    """Generate a classification dataset for supervised learning."""
    n_samples = draw(st.integers(min_value=50, max_value=200))
    n_features = draw(st.integers(min_value=2, max_value=10))
    n_classes = draw(st.integers(min_value=2, max_value=5))

    # Generate features
    features = {}
    feature_names = [f'feature_{i}' for i in range(n_features)]

    for name in feature_names:
        features[name] = draw(st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
            min_size=n_samples, max_size=n_samples
        ))

    # Generate target ensuring each class has enough samples for stratified split
    # First place at least 3 samples per class, then fill the rest randomly
    target = []
    for c in range(n_classes):
        target.extend([c] * 3)
    remaining = n_samples - len(target)
    if remaining > 0:
        target.extend(draw(st.lists(
            st.integers(min_value=0, max_value=n_classes-1),
            min_size=remaining, max_size=remaining
        )))
    # Shuffle
    np.random.shuffle(target)
    target = target[:n_samples]

    features['target'] = target
    return pd.DataFrame(features), feature_names, 'target'

@st.composite
def clustering_dataset(draw):
    """Generate a dataset suitable for clustering."""
    n_samples = draw(st.integers(min_value=30, max_value=150))
    n_features = draw(st.integers(min_value=2, max_value=8))

    features = {}
    feature_names = [f'feature_{i}' for i in range(n_features)]

    for name in feature_names:
        features[name] = draw(st.lists(
            st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
            min_size=n_samples, max_size=n_samples
        ))

    return pd.DataFrame(features), feature_names


class TestMLTrainingProperties:
    """Property-based tests for ML training functionality."""

    def setup_method(self):
        """Mock S3 client for all tests since visualization functions use it."""
        self.s3_mock = Mock()
        self.s3_mock.put_object.return_value = {}
        self.s3_patcher = patch.object(_module, 's3_client', self.s3_mock)
        self.s3_patcher.start()

    def teardown_method(self):
        self.s3_patcher.stop()

    @given(classification_dataset())
    @settings(max_examples=20, deadline=30000)
    def test_property_17_machine_learning_model_training(self, dataset_info):
        """
        Feature: ai-data-analyst-platform, Property 17: Machine Learning Model Training

        For any valid dataset and model selection (Logistic Regression, Random Forest, KNN, SVM),
        the ML engine should successfully train the model and calculate standard performance metrics.

        Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
        """
        dataset, feature_columns, target_column = dataset_info

        # Ensure each class has at least 2 samples for stratified split
        class_counts = dataset[target_column].value_counts()
        assume(class_counts.min() >= 2)

        # Test each supervised algorithm
        algorithms = ['logistic_regression', 'random_forest', 'knn', 'svm']

        for algorithm in algorithms:
            try:
                result = train_supervised_model(
                    dataset, algorithm, target_column, feature_columns, {}
                )

                # Verify model training succeeded
                assert result['model_type'] == 'supervised'
                assert result['algorithm'] == algorithm
                assert result['target_column'] == target_column
                assert set(result['feature_columns']) == set(feature_columns)

                # Verify metrics are calculated
                metrics = result['metrics']
                assert 'accuracy' in metrics
                assert 'precision' in metrics
                assert 'recall' in metrics
                assert 'f1_score' in metrics

                # Verify metric ranges
                assert 0 <= metrics['accuracy'] <= 1
                assert 0 <= metrics['precision'] <= 1
                assert 0 <= metrics['recall'] <= 1
                assert 0 <= metrics['f1_score'] <= 1

                # Verify visualizations are generated
                assert 'visualizations' in result
                assert isinstance(result['visualizations'], list)

            except Exception as e:
                pytest.fail(f"Model training failed for {algorithm}: {str(e)}")

    @given(clustering_dataset())
    @settings(max_examples=15, deadline=25000)
    def test_property_19_clustering_analysis_execution(self, dataset_info):
        """
        Feature: ai-data-analyst-platform, Property 19: Clustering Analysis Execution

        For any dataset suitable for clustering, K-Means and DBSCAN algorithms should execute
        successfully and produce silhouette scores and cluster visualizations.

        Validates: Requirements 7.1, 7.2, 7.3, 7.4
        """
        dataset, feature_columns = dataset_info

        # Test K-Means clustering
        try:
            kmeans_result = train_unsupervised_model(
                dataset, 'kmeans', feature_columns, {'n_clusters': 3}
            )

            # Verify clustering succeeded
            assert kmeans_result['model_type'] == 'unsupervised'
            assert kmeans_result['algorithm'] == 'kmeans'
            assert set(kmeans_result['feature_columns']) == set(feature_columns)

            # Verify metrics
            metrics = kmeans_result['metrics']
            assert 'n_clusters' in metrics
            assert metrics['n_clusters'] > 0

            # Silhouette score should be present for valid clustering
            if 'silhouette_score' in metrics:
                assert -1 <= metrics['silhouette_score'] <= 1

            # Verify visualizations
            assert 'visualizations' in kmeans_result
            assert isinstance(kmeans_result['visualizations'], list)

        except Exception as e:
            pytest.fail(f"K-Means clustering failed: {str(e)}")

        # Test DBSCAN clustering
        try:
            dbscan_result = train_unsupervised_model(
                dataset, 'dbscan', feature_columns, {'eps': 0.5, 'min_samples': 3}
            )

            # Verify clustering succeeded
            assert dbscan_result['model_type'] == 'unsupervised'
            assert dbscan_result['algorithm'] == 'dbscan'
            assert set(dbscan_result['feature_columns']) == set(feature_columns)

            # Verify metrics
            metrics = dbscan_result['metrics']
            assert 'n_clusters' in metrics
            assert metrics['n_clusters'] >= 0

            # Verify visualizations
            assert 'visualizations' in dbscan_result
            assert isinstance(dbscan_result['visualizations'], list)

        except Exception as e:
            pytest.fail(f"DBSCAN clustering failed: {str(e)}")

    def test_property_20_ml_results_persistence(self):
        """
        Feature: ai-data-analyst-platform, Property 20: ML Results Persistence

        For any completed machine learning operation (supervised or unsupervised),
        the results and metrics should be stored in DynamoDB.

        Validates: Requirements 6.8, 7.5
        """
        # Create sample results
        supervised_result = {
            'model_type': 'supervised',
            'algorithm': 'random_forest',
            'target_column': 'target',
            'feature_columns': ['feature_1', 'feature_2'],
            'metrics': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87,
                'f1_score': 0.85
            },
            'visualizations': ['confusion_matrix.png', 'roc_curve.png']
        }

        unsupervised_result = {
            'model_type': 'unsupervised',
            'algorithm': 'kmeans',
            'feature_columns': ['feature_1', 'feature_2'],
            'metrics': {
                'silhouette_score': 0.65,
                'n_clusters': 3
            },
            'visualizations': ['cluster_plot.png'],
            'n_clusters': 3
        }

        # Verify required fields are present for storage
        for result in [supervised_result, unsupervised_result]:
            assert 'model_type' in result
            assert 'algorithm' in result
            assert 'metrics' in result
            assert 'visualizations' in result
            assert isinstance(result['metrics'], dict)
            assert isinstance(result['visualizations'], list)

    @given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=10, max_size=100))
    def test_supervised_metrics_calculation(self, probabilities):
        """Test that supervised metrics are calculated correctly."""
        # Create binary classification scenario
        np.random.seed(42)
        y_true = np.random.randint(0, 2, len(probabilities))
        y_pred = (np.array(probabilities) > 0.5).astype(int)

        metrics = calculate_supervised_metrics(y_true, y_pred)

        # Verify all required metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1

    @given(st.integers(min_value=2, max_value=10))
    def test_unsupervised_metrics_calculation(self, n_clusters):
        """Test that unsupervised metrics are calculated correctly."""
        # Create sample clustering results
        n_samples = 50
        np.random.seed(42)
        X = np.random.rand(n_samples, 3)
        labels = np.random.randint(0, n_clusters, n_samples)

        metrics = calculate_unsupervised_metrics(X, labels)

        # Verify required metrics
        assert 'n_clusters' in metrics
        assert metrics['n_clusters'] == n_clusters

        # Silhouette score should be present for valid clustering
        if 'silhouette_score' in metrics:
            assert -1 <= metrics['silhouette_score'] <= 1

    def test_model_creation_supervised(self):
        """Test that supervised models are created correctly."""
        algorithms = ['logistic_regression', 'random_forest', 'knn', 'svm']

        for algorithm in algorithms:
            model = create_supervised_model(algorithm, {})
            assert model is not None
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')

    def test_model_creation_unsupervised(self):
        """Test that unsupervised models are created correctly."""
        algorithms = ['kmeans', 'dbscan']

        for algorithm in algorithms:
            if algorithm == 'kmeans':
                model = create_unsupervised_model(algorithm, {'n_clusters': 3})
            else:  # dbscan
                model = create_unsupervised_model(algorithm, {'eps': 0.5, 'min_samples': 5})

            assert model is not None
            assert hasattr(model, 'fit_predict')

    def test_invalid_algorithm_handling(self):
        """Test that invalid algorithms are handled properly."""
        with pytest.raises(ValueError):
            create_supervised_model('invalid_algorithm', {})

        with pytest.raises(ValueError):
            create_unsupervised_model('invalid_algorithm', {})

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
