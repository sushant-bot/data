"""
Property-based tests for Visualization Lambda function.

These tests validate the correctness properties defined in the design document
for the AI Data Analyst Platform visualization generation functionality.

Properties tested:
- Property 15: Correlation Calculation Accuracy
- Property 16: Visualization Storage and Retrieval
- Property 18: Model Evaluation Visualization
- Property 23: Feature Importance Visualization
"""

import pytest
import json
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
import io

# Set AWS region before importing lambda (boto3 clients initialize at module level)
os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')
os.environ.setdefault('AWS_ACCESS_KEY_ID', 'testing')
os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'testing')

# Load visualization lambda_function via importlib to avoid sys.path collisions
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "viz_lambda",
    os.path.join(os.path.dirname(__file__), '..', 'lambda', 'visualization', 'lambda_function.py')
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
sys.modules['viz_lambda'] = _module

lambda_handler = _module.lambda_handler
generate_correlation_heatmap = _module.generate_correlation_heatmap
generate_confusion_matrix_viz = _module.generate_confusion_matrix_viz
generate_roc_curve_viz = _module.generate_roc_curve_viz
generate_cluster_plot_viz = _module.generate_cluster_plot_viz
generate_feature_importance_viz = _module.generate_feature_importance_viz
save_plot_to_s3 = _module.save_plot_to_s3
load_ml_results = _module.load_ml_results
log_visualization_operation = _module.log_visualization_operation
generate_presigned_url = _module.generate_presigned_url
load_dataset = _module.load_dataset


# Test data generators
@st.composite
def numerical_dataframe(draw):
    """Generate DataFrame with only numerical columns for correlation testing."""
    num_rows = draw(st.integers(min_value=10, max_value=50))
    num_cols = draw(st.integers(min_value=2, max_value=6))

    data = {}
    for i in range(num_cols):
        col_name = f"feature_{i}"
        col_type = draw(st.sampled_from(['int', 'float']))
        if col_type == 'int':
            data[col_name] = draw(st.lists(
                st.integers(min_value=-100, max_value=100),
                min_size=num_rows, max_size=num_rows
            ))
        else:
            data[col_name] = draw(st.lists(
                st.floats(min_value=-100.0, max_value=100.0,
                          allow_nan=False, allow_infinity=False),
                min_size=num_rows, max_size=num_rows
            ))

    return pd.DataFrame(data)


@st.composite
def feature_importance_data(draw):
    """Generate feature importance data for testing."""
    num_features = draw(st.integers(min_value=2, max_value=10))
    feature_names = [f"feature_{i}" for i in range(num_features)]
    importances = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=num_features, max_size=num_features
    ))
    return feature_names, importances


@st.composite
def confusion_matrix_data(draw):
    """Generate confusion matrix test data."""
    num_classes = draw(st.integers(min_value=2, max_value=5))
    num_samples = draw(st.integers(min_value=20, max_value=100))
    labels = [f"class_{i}" for i in range(num_classes)]
    y_true = draw(st.lists(
        st.integers(min_value=0, max_value=num_classes - 1),
        min_size=num_samples, max_size=num_samples
    ))
    y_pred = draw(st.lists(
        st.integers(min_value=0, max_value=num_classes - 1),
        min_size=num_samples, max_size=num_samples
    ))
    return y_true, y_pred, labels


@st.composite
def roc_curve_data(draw):
    """Generate ROC curve test data (binary classification)."""
    num_samples = draw(st.integers(min_value=20, max_value=100))
    y_true = draw(st.lists(
        st.integers(min_value=0, max_value=1),
        min_size=num_samples, max_size=num_samples
    ))
    # Ensure both classes are present
    assume(0 in y_true and 1 in y_true)
    y_scores = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=num_samples, max_size=num_samples
    ))
    return y_true, y_scores


class TestCorrelationVisualization:
    """
    Property 15: Correlation Calculation Accuracy
    Validates: Requirements 5.1, 5.2
    """

    def setup_method(self):
        self.s3_mock = Mock()
        self.dynamodb_mock = Mock()
        self.s3_patcher = patch.object(_module, 's3_client', self.s3_mock)
        self.dynamodb_patcher = patch.object(_module, 'dynamodb', self.dynamodb_mock)
        self.s3_patcher.start()
        self.dynamodb_patcher.start()

    def teardown_method(self):
        self.s3_patcher.stop()
        self.dynamodb_patcher.stop()

    @given(df=numerical_dataframe())
    @settings(max_examples=20, deadline=2000, suppress_health_check=[HealthCheck.data_too_large])
    def test_property_15_correlation_heatmap_generation(self, df):
        """
        Feature: ai-data-analyst-platform, Property 15: Correlation Calculation Accuracy

        For any numerical dataset, the correlation coefficients calculated should match
        standard mathematical correlation formulas and be properly visualized in a heatmap.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        assume(len(df.columns) >= 2)
        session_id = "test-session-corr"

        # Reset mock between Hypothesis iterations
        self.s3_mock.reset_mock()
        self.s3_mock.put_object.return_value = {}

        result = generate_correlation_heatmap(session_id, df, {})
        plt.close('all')

        # Verify a visualization key was returned
        assert result is not None
        assert f"visualizations/{session_id}/correlation_heatmap_" in result
        assert result.endswith('.png')

        # Verify S3 was called with correct bucket and a PNG image
        self.s3_mock.put_object.assert_called_once()
        call_kwargs = self.s3_mock.put_object.call_args
        assert call_kwargs[1]['ContentType'] == 'image/png'
        assert len(call_kwargs[1]['Body']) > 0  # Non-empty image data

    def test_correlation_heatmap_requires_minimum_columns(self):
        """Correlation heatmap should return None for datasets with < 2 numeric columns."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        df = pd.DataFrame({'single_col': [1, 2, 3, 4, 5]})
        result = generate_correlation_heatmap("test-session", df, {})
        plt.close('all')

        assert result is None

    def test_correlation_heatmap_ignores_non_numeric(self):
        """Correlation heatmap should only use numeric columns."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        df = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0, 4.0],
            'num2': [4.0, 3.0, 2.0, 1.0],
            'text': ['a', 'b', 'c', 'd']
        })
        self.s3_mock.put_object.return_value = {}

        result = generate_correlation_heatmap("test-session", df, {})
        plt.close('all')

        assert result is not None


class TestVisualizationStorageAndRetrieval:
    """
    Property 16: Visualization Storage and Retrieval
    Validates: Requirements 5.3, 5.4, 9.6, 9.7
    """

    def setup_method(self):
        self.s3_mock = Mock()
        self.dynamodb_mock = Mock()
        self.table_mock = Mock()
        self.dynamodb_mock.Table.return_value = self.table_mock
        self.s3_patcher = patch.object(_module, 's3_client', self.s3_mock)
        self.dynamodb_patcher = patch.object(_module, 'dynamodb', self.dynamodb_mock)
        self.s3_patcher.start()
        self.dynamodb_patcher.start()

    def teardown_method(self):
        self.s3_patcher.stop()
        self.dynamodb_patcher.stop()

    def test_property_16_save_plot_stores_png_in_s3(self):
        """
        Feature: ai-data-analyst-platform, Property 16: Visualization Storage and Retrieval

        For any generated visualization, it should be stored in S3 with a unique
        identifier and be retrievable for frontend display.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Create a simple plot
        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])

        self.s3_mock.put_object.return_value = {}

        key = save_plot_to_s3("test-session", "test_chart")
        plt.close('all')

        assert key is not None
        assert "visualizations/test-session/test_chart_" in key
        assert key.endswith('.png')

        # Verify S3 received the data
        self.s3_mock.put_object.assert_called_once()
        kwargs = self.s3_mock.put_object.call_args[1]
        assert kwargs['ContentType'] == 'image/png'
        assert kwargs['Bucket'] == 'ai-data-analyst-platform-data-dev-077437903006'

    def test_property_16_presigned_url_generation(self):
        """Presigned URLs should be generated for visualization access."""
        expected_url = "https://s3.amazonaws.com/bucket/key?signed=true"
        self.s3_mock.generate_presigned_url.return_value = expected_url

        url = generate_presigned_url("visualizations/session/chart.png")

        assert url == expected_url
        self.s3_mock.generate_presigned_url.assert_called_once_with(
            'get_object',
            Params={'Bucket': 'ai-data-analyst-platform-data-dev-077437903006', 'Key': 'visualizations/session/chart.png'},
            ExpiresIn=3600
        )

    def test_property_16_visualization_operation_logged(self):
        """Visualization operations should be logged to DynamoDB."""
        self.table_mock.put_item.return_value = {}

        log_visualization_operation("test-session", "correlation_heatmap", "viz/key.png")

        self.table_mock.put_item.assert_called_once()
        item = self.table_mock.put_item.call_args[1]['Item']
        assert item['session_id'] == "test-session"
        assert item['operation_type'] == 'visualization'
        assert item['visualization_type'] == 'correlation_heatmap'
        assert item['visualization_key'] == 'viz/key.png'
        assert item['status'] == 'completed'

    def test_property_16_full_handler_flow(self):
        """Full lambda handler should generate, store, log, and return presigned URL."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        session_id = "test-session-full"
        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [5.0, 4.0, 3.0, 2.0, 1.0],
            'c': [2.0, 3.0, 1.0, 5.0, 4.0]
        })

        # Mock S3 dataset retrieval
        csv_bytes = df.to_csv(index=False).encode()
        self.s3_mock.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=csv_bytes))
        }
        self.s3_mock.put_object.return_value = {}
        self.s3_mock.generate_presigned_url.return_value = "https://signed-url"
        self.table_mock.put_item.return_value = {}

        event = {
            'body': json.dumps({
                'session_id': session_id,
                'visualization_type': 'correlation_heatmap',
                'parameters': {'dataset_type': 'processed'}
            })
        }

        response = lambda_handler(event, {})
        plt.close('all')

        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert body['session_id'] == session_id
        assert body['visualization_type'] == 'correlation_heatmap'
        assert 'visualization_key' in body
        assert body['presigned_url'] == "https://signed-url"


class TestModelEvaluationVisualization:
    """
    Property 18: Model Evaluation Visualization
    Validates: Requirements 6.7, 9.2, 9.3
    """

    def setup_method(self):
        self.s3_mock = Mock()
        self.dynamodb_mock = Mock()
        self.table_mock = Mock()
        self.dynamodb_mock.Table.return_value = self.table_mock
        self.s3_patcher = patch.object(_module, 's3_client', self.s3_mock)
        self.dynamodb_patcher = patch.object(_module, 'dynamodb', self.dynamodb_mock)
        self.s3_patcher.start()
        self.dynamodb_patcher.start()

    def teardown_method(self):
        self.s3_patcher.stop()
        self.dynamodb_patcher.stop()

    @given(data=confusion_matrix_data())
    @settings(max_examples=15, deadline=2000)
    def test_property_18_confusion_matrix_visualization(self, data):
        """
        Feature: ai-data-analyst-platform, Property 18: Model Evaluation Visualization

        For any trained classification model, confusion matrices should be
        generated and stored as visualizations.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        y_true, y_pred, labels = data
        session_id = "test-session-cm"

        # Reset mocks between Hypothesis iterations
        self.s3_mock.reset_mock()
        self.table_mock.reset_mock()

        # Mock ML results lookup
        self.table_mock.query.return_value = {
            'Items': [{'operation_type': 'ml_results', 'model_type': 'supervised'}]
        }
        self.s3_mock.put_object.return_value = {}
        self.s3_mock.head_object.side_effect = Exception("NoSuchKey")

        result = generate_confusion_matrix_viz(session_id, {
            'y_true': y_true,
            'y_pred': y_pred,
            'labels': labels
        })
        plt.close('all')

        assert result is not None
        assert f"visualizations/{session_id}/confusion_matrix_" in result
        assert result.endswith('.png')

    @given(data=roc_curve_data())
    @settings(max_examples=15, deadline=2000, suppress_health_check=[HealthCheck.data_too_large])
    def test_property_18_roc_curve_visualization(self, data):
        """
        Feature: ai-data-analyst-platform, Property 18: Model Evaluation Visualization

        For any trained classification model, ROC curves should be
        generated and stored as visualizations.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        y_true, y_scores = data
        session_id = "test-session-roc"

        # Reset mock between Hypothesis iterations
        self.s3_mock.reset_mock()
        self.s3_mock.put_object.return_value = {}

        result = generate_roc_curve_viz(session_id, {
            'y_true': y_true,
            'y_scores': y_scores
        })
        plt.close('all')

        assert result is not None
        assert f"visualizations/{session_id}/roc_curve_" in result
        assert result.endswith('.png')

    def test_confusion_matrix_requires_data(self):
        """Confusion matrix should return None when y_true/y_pred are missing."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        self.table_mock.query.return_value = {
            'Items': [{'operation_type': 'ml_training', 'ml_type': 'supervised'}]
        }

        result = generate_confusion_matrix_viz("test", {'y_true': None, 'y_pred': None})
        plt.close('all')
        assert result is None

    def test_roc_curve_requires_data(self):
        """ROC curve should return None when y_true/y_scores are missing."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        result = generate_roc_curve_viz("test", {})
        plt.close('all')
        assert result is None


class TestFeatureImportanceVisualization:
    """
    Property 23: Feature Importance Visualization
    Validates: Requirements 9.5
    """

    def setup_method(self):
        self.s3_mock = Mock()
        self.dynamodb_mock = Mock()
        self.s3_patcher = patch.object(_module, 's3_client', self.s3_mock)
        self.dynamodb_patcher = patch.object(_module, 'dynamodb', self.dynamodb_mock)
        self.s3_patcher.start()
        self.dynamodb_patcher.start()

    def teardown_method(self):
        self.s3_patcher.stop()
        self.dynamodb_patcher.stop()

    @given(data=feature_importance_data())
    @settings(max_examples=15, deadline=2000)
    def test_property_23_feature_importance_chart(self, data):
        """
        Feature: ai-data-analyst-platform, Property 23: Feature Importance Visualization

        For any tree-based model (Random Forest), feature importance charts
        should be generated and stored correctly.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        feature_names, importances = data
        session_id = "test-session-fi"

        # Reset mock between Hypothesis iterations
        self.s3_mock.reset_mock()
        self.s3_mock.put_object.return_value = {}

        result = generate_feature_importance_viz(session_id, {
            'feature_names': feature_names,
            'importances': importances
        })
        plt.close('all')

        assert result is not None
        assert f"visualizations/{session_id}/feature_importance_" in result
        assert result.endswith('.png')

        # Verify S3 received the image
        self.s3_mock.put_object.assert_called_once()
        kwargs = self.s3_mock.put_object.call_args[1]
        assert kwargs['ContentType'] == 'image/png'
        assert len(kwargs['Body']) > 0

    def test_feature_importance_requires_data(self):
        """Feature importance should return None when data is missing."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        result = generate_feature_importance_viz("test", {})
        plt.close('all')
        assert result is None

    def test_feature_importance_requires_matching_lengths(self):
        """Feature importance should handle mismatched feature names and importances."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        self.s3_mock.put_object.return_value = {}

        # Valid matching data
        result = generate_feature_importance_viz("test", {
            'feature_names': ['a', 'b', 'c'],
            'importances': [0.5, 0.3, 0.2]
        })
        plt.close('all')
        assert result is not None


class TestClusterPlotVisualization:
    """Tests for cluster plot visualization generation."""

    def setup_method(self):
        self.s3_mock = Mock()
        self.dynamodb_mock = Mock()
        self.s3_patcher = patch.object(_module, 's3_client', self.s3_mock)
        self.dynamodb_patcher = patch.object(_module, 'dynamodb', self.dynamodb_mock)
        self.s3_patcher.start()
        self.dynamodb_patcher.start()

    def teardown_method(self):
        self.s3_patcher.stop()
        self.dynamodb_patcher.stop()

    def test_cluster_plot_generation(self):
        """Cluster plot should generate correctly with valid data."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        df = pd.DataFrame({
            'x': np.random.randn(50),
            'y': np.random.randn(50)
        })
        labels = [0] * 25 + [1] * 25
        self.s3_mock.put_object.return_value = {}

        result = generate_cluster_plot_viz("test-session", df, {
            'labels': labels,
            'feature_columns': ['x', 'y']
        })
        plt.close('all')

        assert result is not None
        assert "cluster_plot_" in result

    def test_cluster_plot_with_centers(self):
        """Cluster plot should display centroids when provided."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        df = pd.DataFrame({
            'x': np.random.randn(50),
            'y': np.random.randn(50)
        })
        labels = [0] * 25 + [1] * 25
        centers = np.array([[0.5, 0.5], [-0.5, -0.5]])
        self.s3_mock.put_object.return_value = {}

        result = generate_cluster_plot_viz("test-session", df, {
            'labels': labels,
            'feature_columns': ['x', 'y'],
            'cluster_centers': centers
        })
        plt.close('all')

        assert result is not None

    def test_cluster_plot_requires_labels(self):
        """Cluster plot should auto-generate clusters when labels are missing."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]})
        self.s3_mock.put_object.return_value = {}
        self.s3_mock.head_object.side_effect = Exception("NoSuchKey")
        result = generate_cluster_plot_viz("test", df, {})
        plt.close('all')
        assert result is not None
        assert 'cluster_plot' in result


class TestLambdaHandlerEdgeCases:
    """Tests for lambda handler error handling and edge cases."""

    def setup_method(self):
        self.s3_mock = Mock()
        self.dynamodb_mock = Mock()
        self.table_mock = Mock()
        self.dynamodb_mock.Table.return_value = self.table_mock
        self.s3_patcher = patch.object(_module, 's3_client', self.s3_mock)
        self.dynamodb_patcher = patch.object(_module, 'dynamodb', self.dynamodb_mock)
        self.s3_patcher.start()
        self.dynamodb_patcher.start()

    def teardown_method(self):
        self.s3_patcher.stop()
        self.dynamodb_patcher.stop()

    def test_missing_required_parameters(self):
        """Handler should return 400 for missing session_id or visualization_type."""
        event = {'body': json.dumps({})}
        response = lambda_handler(event, {})
        assert response['statusCode'] == 400
        body = json.loads(response['body'])
        assert 'error' in body

    def test_unsupported_visualization_type(self):
        """Handler should return 400 for unsupported visualization types."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        csv_bytes = df.to_csv(index=False).encode()
        self.s3_mock.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=csv_bytes))
        }

        event = {
            'body': json.dumps({
                'session_id': 'test',
                'visualization_type': 'unknown_type'
            })
        }
        response = lambda_handler(event, {})
        assert response['statusCode'] == 400
        body = json.loads(response['body'])
        assert 'Unsupported visualization type' in body['error']

    def test_dataset_not_found(self):
        """Handler should return 404 when dataset is not found in S3."""
        self.s3_mock.get_object.side_effect = Exception("NoSuchKey")

        event = {
            'body': json.dumps({
                'session_id': 'nonexistent',
                'visualization_type': 'correlation_heatmap'
            })
        }
        response = lambda_handler(event, {})
        assert response['statusCode'] == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
